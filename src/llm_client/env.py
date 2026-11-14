"""Layered environment & credential resolution (~/.config/llm-client/.env + project .env)."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import dotenv_values, find_dotenv, load_dotenv

from llm_client.utils import parse_api_key_list, sanitize_api_key

VALID_ROUTING_MODES = ("free_first", "paid_first", "free_only", "paid_only")


def get_global_env_path() -> Optional[Path]:
    """
    Resolve the global llm-client .env path (~/.config/llm-client/.env by default).
    Setting LLM_CLIENT_ENV_FILE="" or "none" disables loading the global file.
    """
    override = os.getenv("LLM_CLIENT_ENV_FILE")
    if override is not None:
        stripped = override.strip()
        if not stripped or stripped.lower() in ("none", "false", "0", "off"):
            return None
        return Path(stripped).expanduser()

    xdg_config = os.getenv("XDG_CONFIG_HOME", "").strip()
    base_dir = Path(xdg_config).expanduser() if xdg_config else (Path.home() / ".config")
    return base_dir / "llm-client" / ".env"


def load_project_dotenv() -> Optional[Path]:
    """Load the nearest project .env without overriding already-set process variables."""
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(dotenv_path=found, override=False)
        return Path(found)
    return None


def read_global_env_values() -> Dict[str, str]:
    """Read key-value pairs from the global llm-client .env file without mutating os.environ."""
    env_path = get_global_env_path()
    if env_path is None or not env_path.is_file():
        return {}
    raw = dotenv_values(env_path)
    return {str(k): ("" if v is None else str(v)) for k, v in raw.items() if k}


def load_llm_env(override: bool = False) -> Optional[Path]:
    """
    Convenience helper to load project .env and global ~/.config/llm-client/.env
    into os.environ (project/process env always keeps priority when override=False).
    """
    load_project_dotenv()
    global_path = get_global_env_path()
    if global_path is not None and global_path.is_file():
        load_dotenv(dotenv_path=global_path, override=override)
        return global_path
    return None


@dataclass(frozen=True)
class ResolvedCredentials:
    """Normalized key pool and provider settings for an LLMClient instance."""

    gemini_key_pool: List[Tuple[str, str]]  # Ordered list of (tier_label, api_key)
    routing_mode: str
    fallback_api_key: str
    fallback_base_url: str
    fallback_model: str
    provider: str
    registry_url: str
    registry_path: str


def _dedupe_pool(entries: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen_keys: set[str] = set()
    result: List[Tuple[str, str]] = []
    for tier_label, key in entries:
        clean = sanitize_api_key(key)
        if clean and clean not in seen_keys:
            seen_keys.add(clean)
            result.append((tier_label, clean))
    return result


def normalize_routing_mode(mode: Optional[str]) -> str:
    """Normalize LLM routing mode name, defaulting to 'free_first'."""
    norm = (mode or "").strip().lower().replace("-", "_")
    if norm in VALID_ROUTING_MODES:
        return norm
    return "free_first"


def resolve_credentials(
    *,
    api_key: Optional[str] = None,
    free_api_keys: Optional[Any] = None,
    paid_api_key: Optional[str] = None,
    paid_api_keys: Optional[Any] = None,
    routing_mode: Optional[str] = None,
    fallback_api_key: Optional[str] = None,
    fallback_base_url: Optional[str] = None,
    fallback_model: Optional[str] = None,
    provider: Optional[str] = None,
    registry_url: Optional[str] = None,
    registry_path: Optional[str] = None,
) -> ResolvedCredentials:
    """
    Resolve credentials and routing across:
      1. Explicit constructor arguments
      2. Process / project .env (os.environ)
      3. Global ~/.config/llm-client/.env
    """
    load_project_dotenv()
    global_vals = read_global_env_values()

    def _get_var(name: str) -> str:
        env_val = os.getenv(name)
        if env_val is not None and env_val.strip():
            return env_val.strip()
        return (global_vals.get(name) or "").strip()

    effective_mode = normalize_routing_mode(
        routing_mode if routing_mode is not None else _get_var("LLM_ROUTING_MODE")
    )

    # 1. Determine Gemini key pool
    explicit_pool: List[Tuple[str, str]] = []
    has_explicit_gemini = (
        api_key is not None
        or free_api_keys is not None
        or paid_api_key is not None
        or paid_api_keys is not None
    )

    if has_explicit_gemini:
        explicit_free = parse_api_key_list(free_api_keys)
        explicit_paid = parse_api_key_list(paid_api_keys)
        clean_paid_single = sanitize_api_key(paid_api_key)
        if clean_paid_single and clean_paid_single not in explicit_paid:
            explicit_paid.append(clean_paid_single)

        clean_explicit = sanitize_api_key(api_key)
        if clean_explicit:
            explicit_pool.append(("explicit", clean_explicit))

        ordered_tiers = _order_free_paid_pools(explicit_free, explicit_paid, effective_mode)
        gemini_pool = _dedupe_pool([*explicit_pool, *ordered_tiers])
    else:
        # Check if process/project env sets pool keys or a project-specific GEMINI_API_KEY
        env_free = parse_api_key_list(os.getenv("GEMINI_FREE_API_KEYS"))
        env_paid = parse_api_key_list(os.getenv("GEMINI_PAID_API_KEYS")) + parse_api_key_list(
            os.getenv("GEMINI_PAID_API_KEY")
        )
        env_paid = parse_api_key_list(env_paid)

        env_generic = sanitize_api_key(os.getenv("GEMINI_API_KEY")) or sanitize_api_key(
            os.getenv("GOOGLE_API_KEY")
        )

        if env_free or env_paid:
            gemini_pool = _dedupe_pool(_order_free_paid_pools(env_free, env_paid, effective_mode))
            if not gemini_pool and env_generic and effective_mode not in ("free_only", "paid_only"):
                gemini_pool = [("env", env_generic)]
        elif env_generic:
            # Project .env / CI secret explicitly overrides global pool
            gemini_pool = [("env", env_generic)]
        else:
            # Fall back cleanly to global ~/.config/llm-client/.env
            glob_free = parse_api_key_list(global_vals.get("GEMINI_FREE_API_KEYS"))
            glob_paid = parse_api_key_list(global_vals.get("GEMINI_PAID_API_KEYS")) + parse_api_key_list(
                global_vals.get("GEMINI_PAID_API_KEY")
            )
            glob_paid = parse_api_key_list(glob_paid)
            glob_generic = sanitize_api_key(global_vals.get("GEMINI_API_KEY")) or sanitize_api_key(
                global_vals.get("GOOGLE_API_KEY")
            )

            if glob_free or glob_paid:
                gemini_pool = _dedupe_pool(_order_free_paid_pools(glob_free, glob_paid, effective_mode))
            elif glob_generic:
                gemini_pool = [("default", glob_generic)]
            else:
                gemini_pool = []

    # 2. Cross-provider fallback key (DeepSeek / OpenAI-compatible)
    if fallback_api_key is not None:
        resolved_fb_key = sanitize_api_key(fallback_api_key)
        used_deepseek_default = False
    else:
        env_ds = sanitize_api_key(os.getenv("DEEPSEEK_API_KEY"))
        env_oai = sanitize_api_key(os.getenv("OPENAI_API_KEY"))
        glob_ds = sanitize_api_key(global_vals.get("DEEPSEEK_API_KEY"))
        glob_oai = sanitize_api_key(global_vals.get("OPENAI_API_KEY"))

        resolved_fb_key = env_ds or env_oai or glob_ds or glob_oai
        used_deepseek_default = bool(env_ds or (not env_oai and (glob_ds or not glob_oai)))

    resolved_fb_model = (
        fallback_model
        or _get_var("LLM_FALLBACK_MODEL")
        or _get_var("OPENAI_MODEL")
        or "deepseek-chat"
    ).strip()

    env_base_url = _get_var("LLM_FALLBACK_BASE_URL") or _get_var("OPENAI_BASE_URL")
    if fallback_base_url is not None:
        resolved_fb_url = fallback_base_url.strip()
    elif env_base_url:
        resolved_fb_url = env_base_url
    elif "deepseek" in resolved_fb_model.lower() or used_deepseek_default:
        resolved_fb_url = "https://api.deepseek.com/v1"
    else:
        resolved_fb_url = "https://api.openai.com/v1"

    resolved_provider = (provider if provider is not None else (_get_var("LLM_PROVIDER") or "auto")).lower().strip()
    resolved_reg_url = (registry_url if registry_url is not None else _get_var("LLM_REGISTRY_URL")).strip()
    resolved_reg_path = (registry_path if registry_path is not None else _get_var("LLM_REGISTRY_PATH")).strip()

    return ResolvedCredentials(
        gemini_key_pool=gemini_pool,
        routing_mode=effective_mode,
        fallback_api_key=resolved_fb_key,
        fallback_base_url=resolved_fb_url,
        fallback_model=resolved_fb_model,
        provider=resolved_provider,
        registry_url=resolved_reg_url,
        registry_path=resolved_reg_path,
    )


def _order_free_paid_pools(
    free_keys: Sequence[str],
    paid_keys: Sequence[str],
    routing_mode: str,
) -> List[Tuple[str, str]]:
    free_entries = [("free", k) for k in free_keys]
    paid_entries = [("paid", k) for k in paid_keys]

    if routing_mode == "paid_first":
        return paid_entries + free_entries
    if routing_mode == "free_only":
        return free_entries
    if routing_mode == "paid_only":
        return paid_entries
    return free_entries + paid_entries
