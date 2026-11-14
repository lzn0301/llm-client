"""Utility helpers for JSON cleaning/extraction and error classification."""

import re
from typing import Optional


def clean_json_text(text: str) -> str:
    """Strip Markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    if not text:
        return ""
    cleaned = text.strip()
    fence_match = re.match(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", cleaned, flags=re.DOTALL)
    if fence_match:
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
    Extract the first balanced JSON object ({...}) or array ([...]) from text.
    Handles string literals and escaped quotes safely.
    """
    if not text:
        return None

    cleaned = clean_json_text(text)
    if not cleaned:
        return None

    start_idx = -1
    open_char = ""
    close_char = ""
    for idx, ch in enumerate(cleaned):
        if ch == "{":
            start_idx = idx
            open_char, close_char = "{", "}"
            break
        if ch == "[":
            start_idx = idx
            open_char, close_char = "[", "]"
            break

    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start_idx, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return cleaned[start_idx : idx + 1]

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
