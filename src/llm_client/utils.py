"""Utility helpers for JSON cleaning/extraction and error classification."""

import json
import re
from typing import Optional


def clean_json_text(text: str) -> str:
    """Strip Markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    if not text:
        return ""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json|JSON)?\s*\n?(.*?)\n?```", cleaned, flags=re.DOTALL)
    if fence_match and cleaned.startswith("```"):
        return fence_match.group(1).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract the first valid JSON object ({...}) or array ([...]) from text.
    Safely handles surrounding markdown fences, prose, and bracketed prefixes like '[Note]'.
    """
    if not text:
        return None

    stripped = text.strip()
    fence_match = re.search(r"```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```", stripped)
    inner = fence_match.group(1).strip() if fence_match else clean_json_text(stripped)
    if not inner:
        return None

    try:
        parsed = json.loads(inner)
        if isinstance(parsed, (dict, list)):
            return inner
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(inner):
        if ch not in ("{", "["):
            continue
        try:
            obj, end_pos = decoder.raw_decode(inner, idx)
            if isinstance(obj, (dict, list)):
                return inner[idx:end_pos]
        except Exception:
            continue

    return None


def is_auth_or_billing_error(exc: Exception) -> bool:
    """Return True if exception looks like an invalid API key or permission error."""
    err_str = str(exc).lower()
    auth_markers = (
        "401",
        "403",
        "unauthenticated",
        "permission denied",
        "permission_denied",
        "api key not valid",
        "invalid api key",
        "api_key_invalid",
    )
    return any(m in err_str for m in auth_markers)


def is_model_not_found_error(exc: Exception) -> bool:
    """Return True when the target model does not exist or has been deprecated (404)."""
    err_str = str(exc).lower()
    return (
        "404" in err_str
        or "not_found" in err_str
        or "not found" in err_str
        or "is not supported for generatecontent" in err_str
        or "deprecated" in err_str
    )


def is_thinking_unsupported_error(exc: Exception) -> bool:
    """Return True if Gemini rejects thinking_config for the requested model."""
    err_str = str(exc).lower()
    return "thinking" in err_str and (
        "400" in err_str
        or "invalid_argument" in err_str
        or "invalid argument" in err_str
        or "not supported" in err_str
        or "unknown name" in err_str
    )


def is_retriable_error(exc: Exception) -> bool:
    """
    Determine whether an exception is transient or warrants retrying / falling back
    to the next model candidate (timeouts, disconnects, 429, 5xx, 503, 404 model unavailable).
    """
    if is_auth_or_billing_error(exc):
        return False

    if is_model_not_found_error(exc):
        return True

    transient_keywords = (
        "503",
        "502",
        "504",
        "500",
        "429",
        "unavailable",
        "resource_exhausted",
        "resource exhausted",
        "deadline_exceeded",
        "deadline exceeded",
        "rate limit",
        "quota",
        "high demand",
        "overloaded",
        "internal",
        "timeout",
        "timed out",
        "readtimeout",
        "connecttimeout",
        "server disconnected",
        "remoteprotocolerror",
        "connection",
        "reset by peer",
        "broken pipe",
        "eof",
    )
    err_str = str(exc).lower()
    exc_type_name = type(exc).__name__.lower()
    if any(k in err_str or k in exc_type_name for k in transient_keywords):
        return True

    # Inspect exception cause/context chain
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause and cause is not exc:
        return is_retriable_error(cause)

    return False
