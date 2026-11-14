"""Model registry and tier resolution with remote override and built-in fallback."""

import copy
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY: Dict[str, Any] = {
    "version": "2026-11-14",
    "tiers": {
        "lite": [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
            "deepseek-chat",
        ],
        "flash": [
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "deepseek-chat",
        ],
        "pro": [
            "gemini-4-argon",
            "gemini-4-carbon",
            "gemini-3.8-flash",
            "deepseek-reasoner",
            "deepseek-chat",
        ],
        "image": [
            "gemini-nano-banana-2.1",
            "gemini-3.1-flash-lite-image",
            "gemini-3-pro-image",
        ],
        "image_lite": [
            "gemini-3.1-flash-lite-image",
            "gemini-nano-banana-2.1",
        ],
        "image_pro": [
            "gemini-3-pro-image",
            "gemini-nano-banana-2.1",
        ],
    },
    "aliases": {
        "gemini-flash-latest": "flash",
        "gemini-pro-latest": "pro",
        "gemini-2.5-flash": "flash",
        "gemini-2.5-flash-lite": "lite",
        "gemini-2.5-pro": "pro",
        "gemini-2.0-flash": "flash",
        "gemini-2.0-flash-lite": "lite",
        "gemini-1.5-flash": "flash",
        "gemini-1.5-pro": "pro",
        "gemini-3-flash-preview": "flash",
        "gemini-3.1-pro-preview": "pro",
        "gemini-3.1-flash-lite-preview": "lite",
        "gemini-3.7-flash": "flash",
        "gemini-4-hydrogen": "flash",
        "gemini-3.1-flash-image": "image",
        "gemini-3.1-flash-image-preview": "image",
        "gemini-2.5-flash-image": "image",
        "gemini-3-pro-image-preview": "image_pro",
    },
}

_REMOTE_CACHE_LOCK = threading.Lock()
_REMOTE_CACHE: Dict[str, Any] = {}
_DEFAULT_CACHE_TTL_SECONDS = 3600.0


def is_cross_provider_model(model_name: str) -> bool:
    """Return True if the model belongs to a non-Gemini OpenAI-compatible provider."""
    lower = model_name.strip().lower()
    return lower.startswith(("deepseek-", "gpt-", "o1-", "o3-", "o4-", "qwen-", "claude-"))


def normalize_model_name(model_name: str) -> str:
    """Strip optional 'models/' prefix and surrounding whitespace."""
    name = (model_name or "").strip()
    if name.startswith("models/"):
        name = name[len("models/") :]
    return name


class ModelRegistry:
    """
    Resolves logical tiers ('lite', 'flash', 'pro', 'image', etc.) or concrete
    model names into ordered fallback candidate queues.
    """

    def __init__(
        self,
        registry_url: Optional[str] = None,
        registry_path: Optional[str] = None,
        registry_data: Optional[Dict[str, Any]] = None,
        remote_timeout_s: float = 3.0,
        cache_ttl_s: float = _DEFAULT_CACHE_TTL_SECONDS,
    ):
        self.registry_url = (registry_url if registry_url is not None else os.getenv("LLM_REGISTRY_URL", "")).strip()
        self.registry_path = (registry_path if registry_path is not None else os.getenv("LLM_REGISTRY_PATH", "")).strip()
        self.remote_timeout_s = remote_timeout_s
        self.cache_ttl_s = cache_ttl_s
        self._explicit_data = registry_data
        self._data: Dict[str, Any] = self._load_registry()

    def _load_local_base(self) -> Dict[str, Any]:
        """Load built-in or local JSON registry as baseline."""
        if self._explicit_data is not None:
            return copy.deepcopy(self._explicit_data)

        candidate_paths: List[Path] = []
        if self.registry_path:
            candidate_paths.append(Path(self.registry_path))

        # Package-bundled JSON
        package_json = Path(__file__).resolve().parent / "default_registry.json"
        candidate_paths.append(package_json)

        # Repo-root model_registry.json if present alongside editable install
        repo_json = Path(__file__).resolve().parent.parent.parent / "model_registry.json"
        candidate_paths.append(repo_json)

        for path in candidate_paths:
            try:
                if path.is_file():
                    with path.open("r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict) and "tiers" in loaded:
                        return loaded
            except Exception as e:
                logger.debug(f"Failed to read local registry at {path}: {e}")

        return copy.deepcopy(DEFAULT_REGISTRY)

    def _fetch_remote(self) -> Optional[Dict[str, Any]]:
        """Fetch remote registry JSON with process-level TTL caching and fail-safe fallback."""
        if not self.registry_url:
            return None

        now = time.time()
        with _REMOTE_CACHE_LOCK:
            cached = _REMOTE_CACHE.get(self.registry_url)
            if cached and (now - cached["ts"] < self.cache_ttl_s):
                return copy.deepcopy(cached["data"])

        try:
            with httpx.Client(timeout=self.remote_timeout_s) as client:
                resp = client.get(self.registry_url)
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, dict) and isinstance(data.get("tiers"), dict):
                with _REMOTE_CACHE_LOCK:
                    _REMOTE_CACHE[self.registry_url] = {"ts": now, "data": data}
                return copy.deepcopy(data)
        except Exception as e:
            logger.debug(f"Remote registry fetch from {self.registry_url} failed (using local fallback): {e}")

        return None

    def _load_registry(self) -> Dict[str, Any]:
        base = self._load_local_base()
        if self._explicit_data is not None:
            return base
        remote = self._fetch_remote()
        if remote:
            # Merge remote over base so missing keys remain covered
            merged = copy.deepcopy(base)
            if isinstance(remote.get("tiers"), dict):
                merged.setdefault("tiers", {}).update(remote["tiers"])
            if isinstance(remote.get("aliases"), dict):
                merged.setdefault("aliases", {}).update(remote["aliases"])
            if "version" in remote:
                merged["version"] = remote["version"]
            return merged
        return base

    def refresh(self) -> None:
        """Force reload registry (clearing remote cache for this URL if configured)."""
        if self.registry_url:
            with _REMOTE_CACHE_LOCK:
                _REMOTE_CACHE.pop(self.registry_url, None)
        self._data = self._load_registry()

    @property
    def tiers(self) -> Dict[str, List[str]]:
        return self._data.get("tiers", {})

    @property
    def aliases(self) -> Dict[str, str]:
        return self._data.get("aliases", {})

    def infer_tier(self, model_name: Optional[str], default_tier: str = "flash") -> str:
        """Infer the logical tier for a model name."""
        if not model_name:
            return default_tier if default_tier in self.tiers else "flash"

        norm = normalize_model_name(model_name)
        lower = norm.lower()
        if lower in self.tiers:
            return lower

        alias_target = self.aliases.get(norm) or self.aliases.get(lower)
        if alias_target and alias_target in self.tiers:
            return alias_target

        if "image" in lower or "nano-banana" in lower or "imagen" in lower:
            if "pro" in lower and "image_pro" in self.tiers:
                return "image_pro"
            if "lite" in lower and "image_lite" in self.tiers:
                return "image_lite"
            return "image"
        if "lite" in lower:
            return "lite"
        if any(k in lower for k in ("pro", "carbon", "argon", "reasoner")):
            return "pro"
        return default_tier if default_tier in self.tiers else "flash"

    def resolve_candidates(
        self,
        model: Optional[str] = None,
        tier: str = "flash",
        include_cross_provider: bool = True,
        fallback_models: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Resolve a model specification (tier name, model name, or comma-separated list)
        into an ordered, deduplicated candidate queue.
        """
        candidates: List[str] = []

        def _append_unique(item: str) -> None:
            norm = normalize_model_name(item)
            if not norm:
                return
            if not include_cross_provider and is_cross_provider_model(norm):
                return
            if norm not in candidates:
                candidates.append(norm)

        spec = normalize_model_name(model or "")
        effective_tier = tier.lower().strip() if tier else "flash"

        if not spec:
            tier_list = self.tiers.get(effective_tier) or self.tiers.get("flash", [])
            for m in tier_list:
                _append_unique(m)
            if fallback_models:
                for m in fallback_models:
                    _append_unique(m)
            return candidates

        # Support comma-separated list passed via env vars like "gemini-3.8-flash,gemini-3.5-flash-lite"
        parts = [normalize_model_name(p) for p in spec.split(",") if normalize_model_name(p)]
        if len(parts) == 1 and parts[0].lower() in self.tiers:
            tier_list = self.tiers[parts[0].lower()]
            for m in tier_list:
                _append_unique(m)
            if fallback_models:
                for m in fallback_models:
                    _append_unique(m)
            return candidates

        # Concrete model(s) requested: keep them first, then append fallback_models and tier backups
        for p in parts:
            if p.lower() in self.tiers:
                for tm in self.tiers[p.lower()]:
                    _append_unique(tm)
            else:
                _append_unique(p)

        if fallback_models:
            for m in fallback_models:
                _append_unique(m)

        inferred_tier = self.infer_tier(parts[0], default_tier=effective_tier)
        for tm in self.tiers.get(inferred_tier, []):
            _append_unique(tm)

        # Ensure image fallback never falls through to text-only deepseek unless requested
        return candidates
