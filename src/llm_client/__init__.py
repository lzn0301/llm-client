"""Public exports for llm-client."""

from llm_client.client import ChatSessionWrapper, LLMClient, UsageRecord
from llm_client.env import get_global_env_path, load_llm_env
from llm_client.registry import ModelRegistry, is_cross_provider_model, normalize_model_name
from llm_client.utils import (
    clean_json_text,
    extract_json_block,
    is_auth_or_billing_error,
    is_model_not_found_error,
    is_retriable_error,
    is_thinking_unsupported_error,
    parse_api_key_list,
    sanitize_api_key,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "LLMClient",
    "ChatSessionWrapper",
    "ModelRegistry",
    "UsageRecord",
    "get_global_env_path",
    "load_llm_env",
    "clean_json_text",
    "extract_json_block",
    "sanitize_api_key",
    "parse_api_key_list",
    "is_retriable_error",
    "is_auth_or_billing_error",
    "is_model_not_found_error",
    "is_thinking_unsupported_error",
    "is_cross_provider_model",
    "normalize_model_name",
]
