"""Unit tests for llm-client library."""

import types as py_types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm_client import (
    LLMClient,
    ModelRegistry,
    UsageRecord,
    clean_json_text,
    extract_json_block,
    is_retriable_error,
)


class _DummyPart:
    def __init__(self, data=None):
        self.inline_data = SimpleNamespace(data=data) if data is not None else None


class _DummyCandidate:
    def __init__(self, parts=None, finish_reason="STOP"):
        self.content = SimpleNamespace(parts=parts or [])
        self.finish_reason = finish_reason


class _DummyResponse:
    def __init__(
        self,
        text="ok",
        prompt_tokens=10,
        output_tokens=20,
        image_bytes=None,
        finish_reason="STOP",
    ):
        self.text = text
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            cached_content_token_count=0,
            thoughts_token_count=0,
        )
        parts = [_DummyPart(data=image_bytes)] if image_bytes is not None else []
        self.candidates = [_DummyCandidate(parts=parts, finish_reason=finish_reason)]


def test_clean_and_extract_json():
    raw = 'Here is the analysis:\n```json\n{"key": "val with {braces} and \\"quotes\\"", "items": [1, 2]}\n```\nHope this helps!'
    cleaned = clean_json_text(raw)
    assert '"key"' in cleaned
    extracted = extract_json_block(raw)
    assert extracted == '{"key": "val with {braces} and \\"quotes\\"", "items": [1, 2]}'

    arr_raw = "Prefix text [1, 2, {\"a\": 3}] suffix text"
    assert extract_json_block(arr_raw) == '[1, 2, {"a": 3}]'


def test_error_classification():
    assert is_retriable_error(RuntimeError("503 UNAVAILABLE: high demand")) is True
    assert is_retriable_error(httpx.ReadTimeout("timed out")) is True
    assert is_retriable_error(httpx.RemoteProtocolError("Server disconnected without sending a response.")) is True
    assert is_retriable_error(RuntimeError("404 NOT_FOUND: model not found")) is True
    assert is_retriable_error(RuntimeError("401 UNAUTHENTICATED: API key not valid")) is False


def test_registry_resolution_and_remote_override():
    registry = ModelRegistry()
    flash_candidates = registry.resolve_candidates(tier="flash")
    assert flash_candidates[0] == "gemini-3.8-flash"
    assert "deepseek-chat" in flash_candidates

    # Concrete deprecated model keeps requested model first, then appends tier backups
    legacy_candidates = registry.resolve_candidates(model="models/gemini-2.5-flash")
    assert legacy_candidates[0] == "gemini-2.5-flash"
    assert "gemini-3.8-flash" in legacy_candidates
    assert "gemini-3.5-flash-lite" in legacy_candidates

    # Image tier excludes text-only cross-provider models
    img_candidates = registry.resolve_candidates(tier="image", include_cross_provider=False)
    assert img_candidates[0] == "gemini-nano-banana-2.1"
    assert all(not c.startswith("deepseek") for c in img_candidates)

    # Remote registry override with mocked httpx response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "version": "2026-12-01",
        "tiers": {"flash": ["gemini-4.0-flash", "deepseek-chat"]},
    }
    mock_resp.raise_for_status.return_value = None
    with patch("llm_client.registry.httpx.Client") as mock_http_cls:
        mock_http_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        remote_reg = ModelRegistry(registry_url="https://example.com/registry.json")
        assert remote_reg.resolve_candidates(tier="flash") == ["gemini-4.0-flash", "deepseek-chat"]


def test_generate_parameter_compatibility_and_timeout_injection():
    recorded: list[UsageRecord] = []
    client = LLMClient(
        api_key="test-gemini-key",
        fallback_api_key="",
        timeout_ms=45000,
        retry_backoff_s=0,
        usage_callback=recorded.append,
    )
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.return_value = _DummyResponse(text="hello world")
    client.client = mock_genai_client

    result = client.generate(
        user_prompt="Say hi",
        system_prompt="Be concise",
        config={"temperature": 0.25},
        step_name="unit_test_step",
    )
    assert result == "hello world"
    call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert call_kwargs["contents"] == "Say hi"
    assert call_kwargs["config"].system_instruction == "Be concise"
    assert call_kwargs["config"].temperature == 0.25
    assert call_kwargs["config"].http_options.timeout == 45000

    assert len(recorded) == 1
    assert recorded[0].step_name == "unit_test_step"
    assert recorded[0].model == "gemini-3.8-flash"
    assert recorded[0].provider == "gemini"
    assert recorded[0].input_tokens == 10
    assert recorded[0].output_tokens == 20


def test_generate_fallback_on_503_and_timeout():
    recorded: list[UsageRecord] = []
    client = LLMClient(
        api_key="test-gemini-key",
        fallback_api_key="",
        max_retries_per_model=2,
        retry_backoff_s=0,
        usage_callback=recorded.append,
    )
    mock_genai_client = MagicMock()

    def _side_effect(model, contents, config):
        if model == "gemini-3.8-flash":
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
        return _DummyResponse(text="recovered on lite", prompt_tokens=15, output_tokens=25)

    mock_genai_client.models.generate_content.side_effect = _side_effect
    client.client = mock_genai_client

    out = client.generate("Ping", tier="flash", step_name="fallback_test")
    assert out == "recovered on lite"
    # First model tried twice, then switched to gemini-3.5-flash-lite
    called_models = [c.kwargs["model"] for c in mock_genai_client.models.generate_content.call_args_list]
    assert called_models == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert recorded[-1].model == "gemini-3.5-flash-lite"


def test_generate_cross_provider_deepseek_fallback():
    recorded: list[UsageRecord] = []
    client = LLMClient(
        api_key="test-gemini-key",
        fallback_api_key="test-deepseek-key",
        max_retries_per_model=1,
        retry_backoff_s=0,
        usage_callback=recorded.append,
    )
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.side_effect = httpx.RemoteProtocolError(
        "Server disconnected without sending a response."
    )
    client.client = mock_genai_client

    mock_http_resp = MagicMock()
    mock_http_resp.status_code = 200
    mock_http_resp.json.return_value = {
        "choices": [{"message": {"content": '{"status": "ok"}'}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }

    with patch("llm_client.client.httpx.Client") as mock_http_cls:
        mock_http_cls.return_value.__enter__.return_value.post.return_value = mock_http_resp
        parsed = client.generate_json("Return status", tier="flash", step_name="cross_provider")

    assert parsed == {"status": "ok"}
    assert len(recorded) == 1
    assert recorded[0].provider == "openai_compatible"
    assert recorded[0].model == "deepseek-chat"
    assert recorded[0].input_tokens == 12
    assert recorded[0].output_tokens == 8


def test_thinking_config_stripping_retry():
    client = LLMClient(api_key="test-key", fallback_api_key="", retry_backoff_s=0)
    mock_genai_client = MagicMock()
    attempts = {"count": 0}

    def _side_effect(model, contents, config):
        attempts["count"] += 1
        if getattr(config, "thinking_config", None) is not None:
            raise RuntimeError("400 INVALID_ARGUMENT: thinking_config is not supported by this model")
        return _DummyResponse(text="stripped thinking works")

    mock_genai_client.models.generate_content.side_effect = _side_effect
    client.client = mock_genai_client

    out = client.generate("Think", model="gemini-3.5-flash-lite", thinking_budget=1024)
    assert out == "stripped thinking works"
    assert attempts["count"] == 2


def test_generate_images_and_image_fallback():
    recorded: list[UsageRecord] = []
    client = LLMClient(
        api_key="test-key",
        fallback_api_key="",
        image_timeout_ms=90000,
        max_retries_per_model=1,
        retry_backoff_s=0,
        usage_callback=recorded.append,
    )
    mock_genai_client = MagicMock()

    def _side_effect(model, contents, config):
        if model == "gemini-nano-banana-2.1":
            raise RuntimeError("503 UNAVAILABLE")
        return _DummyResponse(text=None, image_bytes=b"\x89PNG\r\nfake")

    mock_genai_client.models.generate_content.side_effect = _side_effect
    client.client = mock_genai_client

    img_bytes = client.generate_image("A serene jade bracelet", step_name="hero_img")
    assert img_bytes == b"\x89PNG\r\nfake"
    assert recorded[-1].model == "gemini-3.1-flash-lite-image"
    assert recorded[-1].image_count == 1
    last_cfg = mock_genai_client.models.generate_content.call_args.kwargs["config"]
    assert last_cfg.http_options.timeout == 90000


def test_chat_session_fallback():
    client = LLMClient(
        api_key="test-key",
        fallback_api_key="",
        max_retries_per_model=1,
        retry_backoff_s=0,
    )
    mock_genai_client = MagicMock()

    failing_chat = MagicMock()
    failing_chat.get_history.return_value = []
    failing_chat.send_message.side_effect = RuntimeError("503 UNAVAILABLE")

    healthy_chat = MagicMock()
    healthy_chat.get_history.return_value = []
    healthy_chat.send_message.return_value = _DummyResponse(text="chat reply ok")

    mock_genai_client.chats.create.side_effect = [failing_chat, healthy_chat]
    client.client = mock_genai_client

    chat = client.create_chat(system_instruction="Help", tier="flash")
    resp = chat.send_message("Hello")
    assert resp.text == "chat reply ok"
    assert chat.model == "gemini-3.5-flash-lite"


def test_offline_mock_handler():
    client = LLMClient(
        api_key="",
        fallback_api_key="",
        mock_handler=lambda prompt, json_mode=False, **kw: '{"mock": true}' if json_mode else "mock text",
    )
    assert client.is_configured is False
    assert client.generate("hi") == "mock text"
    assert client.generate_json("hi") == {"mock": True}
