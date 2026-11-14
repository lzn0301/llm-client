"""Unified resilient LLM client with tier routing, key-pool failover, timeout enforcement, and cross-provider backup."""

import base64
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from llm_client.env import resolve_credentials
from llm_client.registry import ModelRegistry, is_cross_provider_model, normalize_model_name
from llm_client.utils import (
    clean_json_text,
    extract_json_block,
    is_auth_or_billing_error,
    is_model_not_found_error,
    is_retriable_error,
    is_thinking_unsupported_error,
)

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Telemetry record emitted via usage_callback after each LLM invocation."""

    step_name: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    thinking_tokens: int = 0
    image_count: int = 0
    duration_s: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChatSessionWrapper:
    """Resilient wrapper around Google GenAI chat session with key-pool & model fallback and telemetry."""

    def __init__(
        self,
        llm_client: "LLMClient",
        candidates: List[str],
        config_kwargs: Dict[str, Any],
        step_name: str = "llm_chat",
    ):
        self._llm = llm_client
        self._candidates = candidates
        self._config_kwargs = config_kwargs
        self._step_name = step_name
        self._key_idx = 0
        self._candidate_idx = 0
        entries = self._llm._iter_gemini_clients()
        if not entries:
            raise RuntimeError("Gemini client is not initialized (missing API key).")
        self._current_key_tier = entries[0][1]
        self._chat = self._create_underlying_chat(entries[0][2], self._candidates[0], history=None)

    @property
    def model(self) -> str:
        return self._candidates[self._candidate_idx]

    @property
    def key_tier(self) -> str:
        return self._current_key_tier

    def _create_underlying_chat(self, client_obj: Any, model_name: str, history: Optional[List[Any]]) -> Any:
        from google.genai import types

        cfg_kwargs = dict(self._config_kwargs)
        # Drop thinking_config if caller passed None explicitly
        if "thinking_config" in cfg_kwargs and cfg_kwargs["thinking_config"] is None:
            cfg_kwargs.pop("thinking_config", None)
        config = types.GenerateContentConfig(**cfg_kwargs)
        create_kwargs: Dict[str, Any] = {"model": model_name, "config": config}
        if history is not None:
            create_kwargs["history"] = history
        return client_obj.chats.create(**create_kwargs)

    def get_history(self, curated: bool = False) -> List[Any]:
        return self._chat.get_history(curated=curated)

    def send_message(self, message: Any, **kwargs: Any) -> Any:
        """Send message with automatic retries, model fallback, and key-pool failover while preserving history."""
        last_err: Optional[Exception] = None
        current_history: Optional[List[Any]] = None
        try:
            current_history = list(self._chat.get_history())
        except Exception:
            current_history = None

        entries = self._llm._iter_gemini_clients()
        if not entries:
            raise RuntimeError("No active Gemini client available for chat session.")

        start_key_pos = min(self._key_idx, len(entries) - 1)
        for k_pos in range(start_key_pos, len(entries)):
            _, key_tier, client_obj = entries[k_pos]
            start_model_idx = self._candidate_idx if k_pos == start_key_pos else 0

            for idx in range(start_model_idx, len(self._candidates)):
                model_name = self._candidates[idx]
                if is_cross_provider_model(model_name):
                    continue

                if k_pos != self._key_idx or idx != self._candidate_idx:
                    try:
                        self._chat = self._create_underlying_chat(client_obj, model_name, history=current_history)
                        self._key_idx = k_pos
                        self._candidate_idx = idx
                        self._current_key_tier = key_tier
                    except Exception as create_err:
                        last_err = create_err
                        continue

                for attempt in range(max(1, self._llm.max_retries_per_model)):
                    t_start = time.time()
                    try:
                        response = self._chat.send_message(message, **kwargs)
                        duration = time.time() - t_start
                        self._llm._emit_gemini_usage(
                            step_name=self._step_name,
                            model=model_name,
                            prompt_text=str(message),
                            output_text=getattr(response, "text", "") or "",
                            response=response,
                            duration_s=duration,
                            key_tier=key_tier,
                        )
                        return response
                    except Exception as exc:
                        last_err = exc
                        if is_thinking_unsupported_error(exc) and "thinking_config" in self._config_kwargs:
                            logger.warning(f"Model {model_name} rejected thinking_config in chat; retrying without it.")
                            self._config_kwargs.pop("thinking_config", None)
                            self._chat = self._create_underlying_chat(client_obj, model_name, history=current_history)
                            continue
                        if is_auth_or_billing_error(exc):
                            break
                        if is_model_not_found_error(exc) or not is_retriable_error(exc):
                            break
                        if attempt + 1 < self._llm.max_retries_per_model:
                            self._llm._sleep_backoff(attempt)

                if last_err is not None and is_auth_or_billing_error(last_err):
                    break

        if last_err is not None:
            raise last_err
        raise RuntimeError("No usable Gemini model candidates available for chat session.")

    def __getattr__(self, item: str) -> Any:
        return getattr(self._chat, item)


class LLMClient:
    """
    Unified resilient LLM client.
    Features:
      - Automatic loading of global ~/.config/llm-client/.env + project .env with strict override precedence.
      - Free vs Paid Gemini key pool routing ('free_first', 'paid_first', 'free_only', 'paid_only').
      - Placeholder API key filtering (ignores 'your-openai-api-key', 'xxxxx', etc.).
      - Logical tiers ('lite', 'flash', 'pro', 'image', etc.) + remote/local ModelRegistry.
      - Strict per-request HTTP timeout (http_options={'timeout': ms}) to prevent indefinite hangs.
      - Automatic retries and model fallback on 503/429/5xx/timeouts/disconnects/404.
      - Cross-provider fallback to OpenAI-compatible endpoints (DeepSeek / OpenAI) for text/JSON.
      - Image generation (generate_image / generate_images), chat sessions, token counting, and file uploads.
      - Optional UsageRecord callback for centralized cost/latency telemetry.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        free_api_keys: Optional[Any] = None,
        paid_api_key: Optional[str] = None,
        paid_api_keys: Optional[Any] = None,
        routing_mode: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        fallback_base_url: Optional[str] = None,
        fallback_model: Optional[str] = None,
        default_model: Optional[str] = None,
        default_tier: str = "flash",
        fallback_models: Optional[Sequence[str]] = None,
        provider: Optional[str] = None,
        timeout_ms: int = 60000,
        image_timeout_ms: int = 90000,
        max_retries_per_model: int = 2,
        retry_backoff_s: float = 1.5,
        registry_url: Optional[str] = None,
        registry_path: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
        usage_callback: Optional[Callable[[UsageRecord], None]] = None,
        mock_handler: Optional[Callable[..., str]] = None,
    ):
        creds = resolve_credentials(
            api_key=api_key,
            free_api_keys=free_api_keys,
            paid_api_key=paid_api_key,
            paid_api_keys=paid_api_keys,
            routing_mode=routing_mode,
            fallback_api_key=fallback_api_key,
            fallback_base_url=fallback_base_url,
            fallback_model=fallback_model,
            provider=provider,
            registry_url=registry_url,
            registry_path=registry_path,
        )

        self.routing_mode = creds.routing_mode
        self._key_pool_specs: List[Tuple[str, str]] = creds.gemini_key_pool
        self.api_key: str = self._key_pool_specs[0][1] if self._key_pool_specs else ""

        self.fallback_api_key = creds.fallback_api_key
        self.fallback_base_url = creds.fallback_base_url
        self.fallback_model = creds.fallback_model
        self.provider = creds.provider
        self.default_model = normalize_model_name(default_model) if default_model else None
        self.default_tier = (default_tier or "flash").lower().strip()
        self.fallback_models = list(fallback_models) if fallback_models else []
        self.timeout_ms = int(timeout_ms)
        self.image_timeout_ms = int(image_timeout_ms)
        self.max_retries_per_model = max(1, int(max_retries_per_model))
        self.retry_backoff_s = float(retry_backoff_s)
        self.usage_callback = usage_callback
        self.mock_handler = mock_handler

        self.registry = registry or ModelRegistry(
            registry_url=creds.registry_url,
            registry_path=creds.registry_path,
        )

        self._gemini_entries: List[Dict[str, Any]] = []
        self.client: Any = None
        self._had_primary_client: bool = False
        if self.provider != "openai_compatible":
            self._init_gemini_client()

    def _build_single_genai_client(self, key: str) -> Any:
        try:
            from google import genai
            from google.genai import types

            return genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=self.timeout_ms),
            )
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}")
            return None

    def _init_gemini_client(self) -> None:
        specs = list(self._key_pool_specs)
        if not specs and self.api_key:
            specs = [("explicit", self.api_key)]

        self._gemini_entries = []
        for tier_label, key in specs:
            built = self._build_single_genai_client(key)
            if built is not None:
                self._gemini_entries.append({"tier": tier_label, "api_key": key, "client": built})

        self.client = self._gemini_entries[0]["client"] if self._gemini_entries else None
        self._had_primary_client = self.client is not None

    def _iter_gemini_clients(self) -> List[Tuple[int, str, Any]]:
        """Yield active (index, key_tier, client_obj) entries, honoring test overrides to `self.client`."""
        if self._had_primary_client and self.client is None:
            return []
        entries: List[Tuple[int, str, Any]] = []
        for idx, entry in enumerate(self._gemini_entries):
            client_obj = self.client if idx == 0 else entry["client"]
            if client_obj is not None:
                entries.append((idx, entry["tier"], client_obj))
        if not entries and self.client is not None:
            entries.append((0, "explicit", self.client))
        return entries

    @property
    def gemini_key_tiers(self) -> List[str]:
        """Return the ordered list of configured Gemini key tier labels (without exposing keys)."""
        return [tier for _, tier, _ in self._iter_gemini_clients()]

    @property
    def is_configured(self) -> bool:
        """Return True if a primary or fallback provider is available."""
        if self.provider == "openai_compatible":
            return bool(self.fallback_api_key)
        return bool(self._iter_gemini_clients() or self.fallback_api_key)

    def _sleep_backoff(self, attempt: int) -> None:
        if self.retry_backoff_s <= 0:
            return
        delay = (1.6**attempt) * self.retry_backoff_s + random.uniform(0.1, 0.5)
        time.sleep(delay)

    def _emit_usage(self, record: UsageRecord) -> None:
        if not self.usage_callback:
            return
        try:
            self.usage_callback(record)
        except Exception as cb_err:
            logger.warning(f"usage_callback raised an exception: {cb_err}")

    def _emit_gemini_usage(
        self,
        *,
        step_name: str,
        model: str,
        prompt_text: str,
        output_text: str,
        response: Any,
        duration_s: float,
        image_count: int = 0,
        key_tier: str = "default",
    ) -> None:
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        thinking_tokens = 0

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
            thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        if input_tokens == 0 and prompt_text:
            input_tokens = max(1, len(prompt_text) // 4)
        if output_tokens == 0:
            if image_count > 0:
                output_tokens = 1120 * image_count
            elif output_text:
                output_tokens = max(1, len(output_text) // 4)

        self._emit_usage(
            UsageRecord(
                step_name=step_name,
                model=model,
                provider="gemini",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                thinking_tokens=thinking_tokens,
                image_count=image_count,
                duration_s=duration_s,
                success=True,
                metadata={"key_tier": key_tier},
            )
        )

    @staticmethod
    def _extract_text_prompt_for_fallback(contents: Any) -> Optional[str]:
        """Convert contents to a single text string for OpenAI-compatible text fallback."""
        if isinstance(contents, str):
            return contents
        if isinstance(contents, (list, tuple)):
            text_parts: List[str] = []
            for item in contents:
                if isinstance(item, str):
                    text_parts.append(item)
                else:
                    return None
            return "\n".join(text_parts)
        return None

    def _call_openai_compatible(
        self,
        *,
        model_name: str,
        prompt_text: str,
        system_instruction: Optional[str],
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        json_mode: bool,
        timeout_ms: int,
        step_name: str,
    ) -> str:
        """Execute chat completion request against an OpenAI-compatible REST endpoint (e.g. DeepSeek)."""
        if not self.fallback_api_key:
            raise ValueError("Fallback API key (DEEPSEEK_API_KEY / OPENAI_API_KEY) is not configured.")

        url = f"{self.fallback_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.fallback_api_key}",
            "Content-Type": "application/json",
        }
        messages: List[Dict[str, str]] = []
        effective_system = system_instruction or ""
        if json_mode and "reasoner" in model_name.lower() and "json" not in effective_system.lower():
            effective_system = (effective_system + "\nRespond with strictly valid JSON.").strip()

        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt_text})

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if temperature is not None and "reasoner" not in model_name.lower():
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        if json_mode and "reasoner" not in model_name.lower():
            payload["response_format"] = {"type": "json_object"}

        timeout_s = max(1.0, timeout_ms / 1000.0)
        t_start = time.time()
        with httpx.Client(timeout=timeout_s) as http_client:
            resp = http_client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI-compatible API error ({resp.status_code}): {resp.text[:500]}")
            data = resp.json()
        duration = time.time() - t_start

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices returned by OpenAI-compatible model '{model_name}': {data}")

        content = (choices[0].get("message") or {}).get("content") or ""
        if not content:
            raise RuntimeError(f"Empty message content returned by '{model_name}'.")

        usage = data.get("usage") or {}
        in_tok = usage.get("prompt_tokens") or max(1, len((effective_system + prompt_text)) // 4)
        out_tok = usage.get("completion_tokens") or max(1, len(content) // 4)
        self._emit_usage(
            UsageRecord(
                step_name=step_name,
                model=model_name,
                provider="openai_compatible",
                input_tokens=int(in_tok),
                output_tokens=int(out_tok),
                duration_s=duration,
                success=True,
                metadata={"routing_mode": self.routing_mode},
            )
        )
        return content.strip()

    def _build_gemini_config(
        self,
        *,
        system_instruction: Optional[str],
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        json_mode: bool,
        response_schema: Optional[Any],
        tools: Optional[List[Any]],
        thinking_budget: Optional[int],
        extra_config: Optional[Any],
        timeout_ms: int,
    ) -> Any:
        from google.genai import types

        if isinstance(extra_config, types.GenerateContentConfig):
            cfg = extra_config.model_copy()
            if system_instruction and not cfg.system_instruction:
                cfg.system_instruction = system_instruction
            if temperature is not None and cfg.temperature is None:
                cfg.temperature = temperature
            if max_output_tokens is not None and cfg.max_output_tokens is None:
                cfg.max_output_tokens = max_output_tokens
            if json_mode and not cfg.response_mime_type:
                cfg.response_mime_type = "application/json"
            if response_schema is not None and cfg.response_schema is None:
                cfg.response_schema = response_schema
            if tools is not None and cfg.tools is None:
                cfg.tools = tools
            if thinking_budget is not None and cfg.thinking_config is None:
                cfg.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
            if cfg.http_options is None:
                cfg.http_options = types.HttpOptions(timeout=timeout_ms)
            elif cfg.http_options.timeout is None:
                cfg.http_options.timeout = timeout_ms
            if not cfg.tools and cfg.automatic_function_calling is None:
                cfg.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
            return cfg

        config_kwargs: Dict[str, Any] = {}
        if isinstance(extra_config, dict):
            config_kwargs.update(extra_config)

        if system_instruction is not None and "system_instruction" not in config_kwargs:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None and "temperature" not in config_kwargs:
            config_kwargs["temperature"] = temperature
        if max_output_tokens is not None and "max_output_tokens" not in config_kwargs:
            config_kwargs["max_output_tokens"] = max_output_tokens
        if json_mode and "response_mime_type" not in config_kwargs:
            config_kwargs["response_mime_type"] = "application/json"
        if response_schema is not None and "response_schema" not in config_kwargs:
            config_kwargs["response_schema"] = response_schema
        if tools is not None and "tools" not in config_kwargs:
            config_kwargs["tools"] = tools
        if thinking_budget is not None and "thinking_config" not in config_kwargs:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

        existing_http = config_kwargs.get("http_options")
        if existing_http is None:
            config_kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms)
        elif isinstance(existing_http, dict):
            merged_http = dict(existing_http)
            merged_http.setdefault("timeout", timeout_ms)
            config_kwargs["http_options"] = types.HttpOptions(**merged_http)

        if not config_kwargs.get("tools") and "automatic_function_calling" not in config_kwargs:
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

        return types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _validate_gemini_text_response(response: Any, model_name: str) -> str:
        if response is None:
            raise RuntimeError(f"Empty response object from Gemini model '{model_name}'.")

        candidates = getattr(response, "candidates", None)
        text = getattr(response, "text", None)

        if candidates:
            candidate = candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            if (
                finish_reason
                and str(finish_reason).upper() not in ("STOP", "FINISH_REASON_UNSPECIFIED", "1", "FINISHREASON.STOP")
                and not text
            ):
                raise RuntimeError(
                    f"Gemini generation stopped with finish_reason='{finish_reason}' on model '{model_name}'."
                )

        if text is None or (isinstance(text, str) and not text.strip() and not candidates):
            raise ValueError(f"Empty text received from Gemini model '{model_name}'.")

        return text.strip() if isinstance(text, str) else str(text)

    def _try_gemini_models_for_client(
        self,
        *,
        client_obj: Any,
        key_tier: str,
        gemini_candidates: Sequence[str],
        actual_contents: Any,
        gemini_cfg: Any,
        retries_per_model: int,
        fallback_text: Optional[str],
        step_name: str,
    ) -> Tuple[bool, Optional[str], Optional[Exception]]:
        """Attempt ordered Gemini candidates against one GenAI client instance."""
        last_err: Optional[Exception] = None
        for model_name in gemini_candidates:
            current_cfg = gemini_cfg
            attempt = 0
            while attempt < retries_per_model:
                attempt += 1
                t_start = time.time()
                try:
                    response = client_obj.models.generate_content(
                        model=model_name,
                        contents=actual_contents,
                        config=current_cfg,
                    )
                    duration = time.time() - t_start
                    result_text = self._validate_gemini_text_response(response, model_name)
                    self._emit_gemini_usage(
                        step_name=step_name,
                        model=model_name,
                        prompt_text=fallback_text or str(actual_contents)[:500],
                        output_text=result_text,
                        response=response,
                        duration_s=duration,
                        key_tier=key_tier,
                    )
                    return True, result_text, None
                except Exception as exc:
                    last_err = exc
                    if is_thinking_unsupported_error(exc) and getattr(current_cfg, "thinking_config", None) is not None:
                        logger.warning(
                            f"[{step_name}] Model '{model_name}' does not support thinking_config; retrying without it."
                        )
                        current_cfg = current_cfg.model_copy(update={"thinking_config": None})
                        attempt -= 1
                        continue

                    if is_auth_or_billing_error(exc):
                        logger.warning(
                            f"[{step_name}] Gemini key tier '{key_tier}' auth/billing error on '{model_name}': {exc}"
                        )
                        return False, None, last_err

                    if is_model_not_found_error(exc):
                        logger.warning(f"[{step_name}] Model '{model_name}' unavailable (404/deprecated), switching next.")
                        break

                    if is_retriable_error(exc):
                        logger.warning(
                            f"[{step_name}] Transient error on '{model_name}' ({key_tier}, attempt {attempt}/{retries_per_model}): {exc}"
                        )
                        if attempt < retries_per_model:
                            self._sleep_backoff(attempt - 1)
                            continue
                        break

                    logger.warning(f"[{step_name}] Model '{model_name}' ({key_tier}) error: {exc}")
                    break

        return False, None, last_err

    def _try_cross_provider_candidates(
        self,
        *,
        cross_candidates: Sequence[str],
        fallback_text: str,
        system_instruction: Optional[str],
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        json_mode: bool,
        timeout_ms: int,
        step_name: str,
    ) -> Tuple[bool, Optional[str], Optional[Exception]]:
        """Attempt OpenAI-compatible cross-provider candidates sequentially."""
        last_err: Optional[Exception] = None
        for model_name in cross_candidates:
            target_fallback_model = model_name
            if (
                "deepseek" in model_name.lower()
                and "deepseek" not in self.fallback_base_url.lower()
                and "deepseek" not in self.fallback_model.lower()
            ):
                target_fallback_model = self.fallback_model
            try:
                logger.warning(f"[{step_name}] Falling back to OpenAI-compatible model: {target_fallback_model}")
                out = self._call_openai_compatible(
                    model_name=target_fallback_model,
                    prompt_text=fallback_text,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    json_mode=json_mode,
                    timeout_ms=timeout_ms,
                    step_name=step_name,
                )
                return True, out, None
            except Exception as fb_err:
                last_err = fb_err
                logger.warning(f"[{step_name}] Cross-provider model '{target_fallback_model}' failed: {fb_err}")
        return False, None, last_err

    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        user_prompt: Optional[str] = None,
        contents: Optional[Any] = None,
        system_instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        json_mode: bool = False,
        response_schema: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        thinking_budget: Optional[int] = None,
        config: Optional[Any] = None,
        max_retries: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        step_name: str = "llm_generate",
    ) -> str:
        """
        Generate text/JSON across ordered Gemini key tiers (free/paid), model fallback queue,
        and optional OpenAI-compatible cross-provider failover.
        """
        actual_contents = contents if contents is not None else (prompt if prompt is not None else user_prompt)
        if actual_contents is None:
            raise ValueError("One of 'prompt', 'user_prompt', or 'contents' must be provided.")

        actual_system = system_instruction if system_instruction is not None else system_prompt
        effective_timeout_ms = int(timeout_ms if timeout_ms is not None else self.timeout_ms)
        retries_per_model = max(1, int(max_retries)) if max_retries is not None else self.max_retries_per_model
        fallback_text = self._extract_text_prompt_for_fallback(actual_contents)
        can_cross_provider = bool(self.fallback_api_key and fallback_text is not None and not tools)

        effective_json_mode = json_mode
        if not effective_json_mode and isinstance(config, dict):
            if config.get("response_mime_type") == "application/json":
                effective_json_mode = True

        if not self.is_configured:
            if self.mock_handler is not None:
                return self.mock_handler(actual_contents, json_mode=effective_json_mode, system_instruction=actual_system)
            raise ValueError(
                "No LLM provider configured. Configure ~/.config/llm-client/.env or set "
                "GEMINI_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY."
            )

        if self.provider == "openai_compatible":
            chosen_model = model or self.fallback_model
            if chosen_model.startswith("gemini-"):
                chosen_model = self.fallback_model
            if fallback_text is None:
                raise ValueError("OpenAI-compatible provider only supports text contents.")
            return self._call_openai_compatible(
                model_name=chosen_model,
                prompt_text=fallback_text,
                system_instruction=actual_system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                json_mode=effective_json_mode,
                timeout_ms=effective_timeout_ms,
                step_name=step_name,
            )

        target_model_or_tier = model if model is not None else self.default_model
        effective_tier = tier or self.default_tier
        candidates = self.registry.resolve_candidates(
            model=target_model_or_tier,
            tier=effective_tier,
            include_cross_provider=can_cross_provider,
            fallback_models=self.fallback_models,
        )
        if can_cross_provider and self.fallback_model not in candidates:
            candidates.append(self.fallback_model)

        gemini_candidates = [c for c in candidates if not is_cross_provider_model(c)]
        cross_candidates = [c for c in candidates if is_cross_provider_model(c)] if can_cross_provider else []
        cross_first = bool(candidates and is_cross_provider_model(candidates[0]))

        last_err: Optional[Exception] = None

        if cross_first and cross_candidates:
            ok, out, err = self._try_cross_provider_candidates(
                cross_candidates=cross_candidates,
                fallback_text=fallback_text or "",
                system_instruction=actual_system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                json_mode=effective_json_mode,
                timeout_ms=effective_timeout_ms,
                step_name=step_name,
            )
            if ok:
                return out or ""
            last_err = err or last_err

        gemini_entries = self._iter_gemini_clients()
        if gemini_entries and gemini_candidates:
            gemini_cfg = self._build_gemini_config(
                system_instruction=actual_system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                json_mode=effective_json_mode,
                response_schema=response_schema,
                tools=tools,
                thinking_budget=thinking_budget,
                extra_config=config,
                timeout_ms=effective_timeout_ms,
            )
            for idx, key_tier, client_obj in gemini_entries:
                ok, out, err = self._try_gemini_models_for_client(
                    client_obj=client_obj,
                    key_tier=key_tier,
                    gemini_candidates=gemini_candidates,
                    actual_contents=actual_contents,
                    gemini_cfg=gemini_cfg,
                    retries_per_model=retries_per_model,
                    fallback_text=fallback_text,
                    step_name=step_name,
                )
                if ok:
                    return out or ""
                if err is not None:
                    last_err = err
                if idx + 1 < len(gemini_entries):
                    logger.warning(
                        f"[{step_name}] Gemini key tier '{key_tier}' failed all candidate models, "
                        f"switching to next key tier '{gemini_entries[idx + 1][1]}'."
                    )

        if not cross_first and cross_candidates:
            ok, out, err = self._try_cross_provider_candidates(
                cross_candidates=cross_candidates,
                fallback_text=fallback_text or "",
                system_instruction=actual_system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                json_mode=effective_json_mode,
                timeout_ms=effective_timeout_ms,
                step_name=step_name,
            )
            if ok:
                return out or ""
            last_err = err or last_err

        if last_err is not None:
            raise last_err
        raise RuntimeError(
            f"All model candidates failed or no candidates resolved for '{target_model_or_tier or effective_tier}'."
        )

    def generate_json(
        self,
        prompt: Optional[str] = None,
        *,
        user_prompt: Optional[str] = None,
        contents: Optional[Any] = None,
        system_instruction: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_schema: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        thinking_budget: Optional[int] = None,
        config: Optional[Any] = None,
        max_retries: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        step_name: str = "llm_generate_json",
    ) -> Any:
        """Call generate(..., json_mode=True) and parse structured JSON safely."""
        raw = self.generate(
            prompt=prompt,
            user_prompt=user_prompt,
            contents=contents,
            system_instruction=system_instruction,
            system_prompt=system_prompt,
            model=model,
            tier=tier,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_mode=True,
            response_schema=response_schema,
            tools=tools,
            thinking_budget=thinking_budget,
            config=config,
            max_retries=max_retries,
            timeout_ms=timeout_ms,
            step_name=step_name,
        )
        cleaned = clean_json_text(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            block = extract_json_block(raw)
            if block is not None:
                return json.loads(block)
            raise

    def generate_images(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        tier: str = "image",
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        response_modalities: Optional[List[str]] = None,
        config: Optional[Any] = None,
        max_retries: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        step_name: str = "llm_generate_image",
    ) -> List[bytes]:
        """Generate images with timeout enforcement, model fallback, and key-pool failover."""
        gemini_entries = self._iter_gemini_clients()
        if not gemini_entries:
            raise RuntimeError("Gemini client is not initialized (missing GEMINI_API_KEY / GOOGLE_API_KEY).")

        from google.genai import types

        effective_timeout_ms = int(timeout_ms if timeout_ms is not None else self.image_timeout_ms)
        retries_per_model = max(1, int(max_retries)) if max_retries is not None else self.max_retries_per_model

        candidates = self.registry.resolve_candidates(
            model=model,
            tier=tier,
            include_cross_provider=False,
        )
        if not candidates:
            candidates = ["gemini-nano-banana-2.1", "gemini-3.1-flash-lite-image"]

        config_kwargs: Dict[str, Any] = {
            "response_modalities": response_modalities or ["IMAGE"],
            "image_config": types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size),
            "http_options": types.HttpOptions(timeout=effective_timeout_ms),
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if isinstance(config, dict):
            config_kwargs.update(config)
            if "http_options" not in config_kwargs or config_kwargs["http_options"] is None:
                config_kwargs["http_options"] = types.HttpOptions(timeout=effective_timeout_ms)

        gen_config = config if isinstance(config, types.GenerateContentConfig) else types.GenerateContentConfig(**config_kwargs)
        if isinstance(gen_config, types.GenerateContentConfig) and gen_config.http_options is None:
            gen_config = gen_config.model_copy(update={"http_options": types.HttpOptions(timeout=effective_timeout_ms)})

        last_err: Optional[Exception] = None
        for idx, key_tier, client_obj in gemini_entries:
            auth_failed_on_key = False
            for model_name in candidates:
                if auth_failed_on_key:
                    break
                for attempt in range(retries_per_model):
                    t_start = time.time()
                    try:
                        response = client_obj.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=gen_config,
                        )
                        duration = time.time() - t_start

                        images: List[bytes] = []
                        for cand in getattr(response, "candidates", None) or []:
                            content = getattr(cand, "content", None)
                            parts = getattr(content, "parts", None) or []
                            for part in parts:
                                inline = getattr(part, "inline_data", None)
                                raw_data = getattr(inline, "data", None) if inline else None
                                if raw_data:
                                    if isinstance(raw_data, bytes):
                                        images.append(raw_data)
                                    elif isinstance(raw_data, str):
                                        images.append(base64.b64decode(raw_data))

                        if not images:
                            raise ValueError(f"No inline image bytes returned from model '{model_name}'.")

                        self._emit_gemini_usage(
                            step_name=step_name,
                            model=model_name,
                            prompt_text=prompt,
                            output_text="",
                            response=response,
                            duration_s=duration,
                            image_count=len(images),
                            key_tier=key_tier,
                        )
                        return images
                    except Exception as exc:
                        last_err = exc
                        if is_auth_or_billing_error(exc):
                            logger.warning(
                                f"[{step_name}] Image auth/billing error on key tier '{key_tier}' ({model_name}): {exc}"
                            )
                            auth_failed_on_key = True
                            break
                        if is_model_not_found_error(exc) or not is_retriable_error(exc):
                            logger.warning(f"[{step_name}] Image model '{model_name}' ({key_tier}) failed: {exc}")
                            break
                        logger.warning(
                            f"[{step_name}] Image error on '{model_name}' ({key_tier}, attempt {attempt + 1}/{retries_per_model}): {exc}"
                        )
                        if attempt + 1 < retries_per_model:
                            self._sleep_backoff(attempt)

            if idx + 1 < len(gemini_entries):
                logger.warning(
                    f"[{step_name}] Image generation failed on key tier '{key_tier}', switching to '{gemini_entries[idx + 1][1]}'."
                )

        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to generate image from all candidate image models and key tiers.")

    def generate_image(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        tier: str = "image",
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        response_modalities: Optional[List[str]] = None,
        config: Optional[Any] = None,
        max_retries: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        step_name: str = "llm_generate_image",
    ) -> bytes:
        """Generate a single image and return raw bytes."""
        images = self.generate_images(
            prompt=prompt,
            model=model,
            tier=tier,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            response_modalities=response_modalities,
            config=config,
            max_retries=max_retries,
            timeout_ms=timeout_ms,
            step_name=step_name,
        )
        return images[0]

    def create_chat(
        self,
        *,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        thinking_budget: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        step_name: str = "llm_chat",
    ) -> ChatSessionWrapper:
        """Create a resilient multi-turn chat session with fallback and timeout protection."""
        if not self._iter_gemini_clients():
            raise RuntimeError("Gemini client is not initialized (missing GEMINI_API_KEY / GOOGLE_API_KEY).")

        from google.genai import types

        effective_timeout_ms = int(timeout_ms if timeout_ms is not None else self.timeout_ms)
        target = model if model is not None else self.default_model
        candidates = self.registry.resolve_candidates(
            model=target,
            tier=tier or self.default_tier,
            include_cross_provider=False,
            fallback_models=self.fallback_models,
        )
        if not candidates:
            candidates = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.5-flash-lite"]

        config_kwargs: Dict[str, Any] = {
            "http_options": types.HttpOptions(timeout=effective_timeout_ms),
        }
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = tools
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

        return ChatSessionWrapper(
            llm_client=self,
            candidates=candidates,
            config_kwargs=config_kwargs,
            step_name=step_name,
        )

    def count_tokens(self, prompt: Any, model: Optional[str] = None) -> int:
        """Count tokens for prompt using Gemini API, returning 0 if client is unavailable."""
        entries = self._iter_gemini_clients()
        if not entries:
            return 0
        target_model = normalize_model_name(model or self.default_model or "")
        if not target_model or target_model in self.registry.tiers:
            candidates = self.registry.resolve_candidates(
                model=target_model or None,
                tier=self.default_tier,
                include_cross_provider=False,
            )
            target_model = candidates[0] if candidates else "gemini-4-hydrogen"
        try:
            resp = entries[0][2].models.count_tokens(model=target_model, contents=prompt)
            return int(getattr(resp, "total_tokens", 0) or 0)
        except Exception as e:
            logger.error(f"Error counting tokens with {target_model}: {e}")
            return 0

    def upload_file(
        self,
        file_path: str,
        *,
        wait_active: bool = True,
        poll_interval_s: float = 2.0,
        max_wait_s: float = 180.0,
    ) -> Any:
        """Upload a multimodal file via Gemini Files API and optionally wait until ACTIVE."""
        entries = self._iter_gemini_clients()
        if not entries:
            raise RuntimeError("Gemini client is not initialized.")

        client_obj = entries[0][2]
        uploaded = client_obj.files.upload(file=file_path)
        if not wait_active:
            return uploaded

        deadline = time.time() + max_wait_s
        while hasattr(uploaded, "state"):
            state_name = getattr(uploaded.state, "name", str(uploaded.state))
            if state_name != "PROCESSING":
                break
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out after {max_wait_s}s waiting for Gemini file '{getattr(uploaded, 'name', '')}' to become ACTIVE."
                )
            time.sleep(poll_interval_s)
            uploaded = client_obj.files.get(name=uploaded.name)

        state_name = getattr(uploaded.state, "name", str(uploaded.state)) if hasattr(uploaded, "state") else "ACTIVE"
        if state_name not in ("ACTIVE", "STATE_UNSPECIFIED", None):
            raise RuntimeError(f"Uploaded file processing failed with state='{state_name}'.")
        return uploaded

    def delete_file(self, file_name: str) -> None:
        """Best-effort cleanup of a remote Gemini Files API resource."""
        entries = self._iter_gemini_clients()
        if not entries or not file_name:
            return
        try:
            entries[0][2].files.delete(name=file_name)
        except Exception as e:
            logger.warning(f"Failed to delete remote file '{file_name}': {e}")
