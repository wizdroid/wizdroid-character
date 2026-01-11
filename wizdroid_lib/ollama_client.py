from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    requests = None


DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Model cache with TTL to prevent UI blocking on repeated INPUT_TYPES calls
_MODELS_CACHE: Dict[str, List[str]] = {}
_MODELS_CACHE_TIME: Dict[str, float] = {}
_MODELS_TTL = 60.0  # Refresh models at most every 60 seconds


def collect_models(ollama_url: str, use_cache: bool = True) -> List[str]:
    """Best-effort discovery of available Ollama models with TTL caching.
    
    Args:
        ollama_url: Ollama server URL
        use_cache: If True, use TTL-based caching to prevent UI blocking
    
    Returns:
        List of model names, or fallback error strings
    """
    global _MODELS_CACHE, _MODELS_CACHE_TIME

    if requests is None:
        return ["install_requests_library"]

    # Check cache
    if use_cache:
        cached = _MODELS_CACHE.get(ollama_url)
        cache_time = _MODELS_CACHE_TIME.get(ollama_url, 0)
        if cached and (time.time() - cache_time) < _MODELS_TTL:
            return cached

    try:
        resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code != 200:
            return _MODELS_CACHE.get(ollama_url) or ["model_not_available"]
        data = resp.json()
        models = [m.get("name", "unknown") for m in data.get("models", [])]
        result = models or ["no_models_found"]
        
        # Update cache
        _MODELS_CACHE[ollama_url] = result
        _MODELS_CACHE_TIME[ollama_url] = time.time()
        return result
        
    except Exception:  # noqa: BLE001
        # Return cached value if available, otherwise error
        return _MODELS_CACHE.get(ollama_url) or ["ollama_not_running"]


def safe_post(url: str, json_body: Dict[str, Any], timeout: int = 120) -> Tuple[bool, str]:
    if requests is None:
        return False, "request_error: requests library not installed"

    try:
        resp = requests.post(url, json=json_body, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return False, f"request_error: {type(e).__name__}: {e}"

    if resp.status_code != 200:
        return False, f"http_error: status {resp.status_code}: {resp.text[:512]}"

    return True, resp.text


def generate_text(
    *,
    ollama_url: str,
    model: str,
    prompt: str,
    system: str,
    options: Optional[Dict[str, Any]] = None,
    images: Optional[List[str]] = None,
    timeout: int = 120,
) -> Tuple[bool, str]:
    """Call Ollama /api/generate and return (ok, response_or_error)."""

    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "prompt": prompt,
        "system": system,
        "options": options or {},
    }
    if images is not None:
        payload["images"] = images

    api_url = f"{ollama_url.rstrip('/')}/api/generate"
    ok, text = safe_post(api_url, payload, timeout=timeout)
    if not ok:
        return False, text

    try:
        data = json.loads(text)
        out = (data.get("response") or "").strip()
    except Exception:  # noqa: BLE001
        out = text.strip()

    if not out:
        return False, "empty_response"

    return True, out
