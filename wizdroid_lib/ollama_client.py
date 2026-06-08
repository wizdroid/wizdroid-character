from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    requests = None

from .constants import DEFAULT_OLLAMA_URL, VISION_KEYWORDS

# Model cache with TTL to prevent UI blocking on repeated INPUT_TYPES calls
_MODELS_CACHE: Dict[str, List[str]] = {}
_MODELS_CACHE_TIME: Dict[str, float] = {}
_MODELS_TTL = 60.0  # Refresh models at most every 60 seconds


def _is_vision_model(ollama_url: str, model_name: str) -> bool:
    """Return True if model is vision-capable.
    
    Uses cheap name-based keyword match first (supports bakllava, llava, etc
    even on older Ollama installs where /api/show may omit "capabilities").
    Falls back to explicit "vision" in capabilities list from /api/show.
    """
    lower = (model_name or "").lower()
    if any(kw in lower for kw in VISION_KEYWORDS):
        return True
    # Fallback to server-reported capabilities (newer Ollama + manifest)
    try:
        resp = requests.post(
            f"{ollama_url.rstrip('/')}/api/show",
            json={"model": model_name},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            caps = data.get("capabilities", [])
            if "vision" in caps:
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def collect_models(ollama_url: str, use_cache: bool = True, vision_only: bool = False) -> List[str]:
    """Best-effort discovery of available Ollama models with TTL caching.
    
    Args:
        ollama_url: Ollama server URL
        use_cache: If True, use TTL-based caching to prevent UI blocking
        vision_only: If True, only return models with vision capabilities (via name keywords or /api/show)
    
    Returns:
        List of model names, or fallback error strings
    """
    global _MODELS_CACHE, _MODELS_CACHE_TIME

    if requests is None:
        return ["install_requests_library"]

    # Use separate cache key for vision-only queries
    cache_key = f"{ollama_url}?vision_only={vision_only}"

    # Check cache
    if use_cache:
        cached = _MODELS_CACHE.get(cache_key)
        cache_time = _MODELS_CACHE_TIME.get(cache_key, 0)
        if cached and (time.time() - cache_time) < _MODELS_TTL:
            return cached

    try:
        resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code != 200:
            fallback = _MODELS_CACHE.get(cache_key)
            return fallback or ["model_not_available"]
        data = resp.json()
        all_models = [m.get("name", "unknown") for m in data.get("models", [])]

        if vision_only:
            # Filter to only vision-capable models (name keywords + capabilities)
            vision_models = [m for m in all_models if _is_vision_model(ollama_url, m)]
            result = vision_models or ["no_vision_models_found"]
        else:
            result = all_models or ["no_models_found"]

        # Update cache
        _MODELS_CACHE[cache_key] = result
        _MODELS_CACHE_TIME[cache_key] = time.time()
        return result

    except Exception:  # noqa: BLE001
        # Return cached value if available, otherwise error
        return _MODELS_CACHE.get(cache_key) or ["ollama_not_running"]


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
    """Call Ollama /api/generate and return (ok, response_or_error).

    Handles thinking-capable models (gemma, qwen, etc.) by:
    - Disabling the internal thinking budget so tokens are reserved for the
      actual response.
    - Falling back to message.content when the response field is empty.
    - Retrying with a higher token budget when done_reason == "length".
    """

    opts = dict(options or {})

    # ---- detect thinking-capable models and disable thinking ---------------
    model_lower = model.lower()
    is_thinking_model = any(
        prefix in model_lower
        for prefix in (
            "gemma",
            "qwen",
            "deepseek-r1",
            "qwq",
            "openthinking",
        )
    )
    if is_thinking_model and "think" not in opts:
        opts["think"] = 0  # disable internal thinking to reserve tokens for response

    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "prompt": prompt,
        "system": system,
        "options": opts,
    }
    if images is not None:
        payload["images"] = images

    api_url = f"{ollama_url.rstrip('/')}/api/generate"

    def _do_request(local_opts: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Return (ok, text, parsed_data_dict_or_None)."""
        p = dict(payload)
        p["options"] = dict(local_opts)
        ok, text = safe_post(api_url, p, timeout=timeout)
        if not ok:
            return False, text, None
        try:
            data = json.loads(text)
        except Exception:  # noqa: BLE001
            # Not valid JSON – treat raw text as the response
            return True, text.strip(), None
        return True, text, data

    ok, raw_text, data = _do_request(opts)

    if not ok:
        return False, raw_text

    # ---- extract response ------------------------------------------------
    if not isinstance(data, dict):
        # JSON parse failure fallback: use raw text
        out = raw_text.strip()
        if out:
            return True, out
        return False, "empty_response"

    # Some models use /api/chat-style embedding inside /api/generate
    msg = data.get("message")
    if isinstance(msg, dict):
        out = (msg.get("content") or "").strip()
        if out:
            return True, out

    out = (data.get("response") or "").strip()
    thinking = (data.get("thinking") or "").strip()
    done_reason = (data.get("done_reason") or "").lower()

    if out:
        return True, out

    # ---- response is empty → diagnose ---------------------------------
    if thinking and done_reason == "length":
        # Thinking consumed all token budget. Retry once with thinking
        # forcefully disabled and double the num_predict budget.
        retry_opts = dict(opts)
        retry_opts["think"] = 0
        if "num_predict" in retry_opts:
            retry_opts["num_predict"] = max(int(retry_opts["num_predict"]) * 2, 512)
        else:
            retry_opts["num_predict"] = 512

        ok2, _, data2 = _do_request(retry_opts)
        if ok2 and isinstance(data2, dict):
            msg2 = data2.get("message")
            if isinstance(msg2, dict):
                out2 = (msg2.get("content") or "").strip()
                if out2:
                    return True, out2
            out2 = (data2.get("response") or "").strip()
            if out2:
                return True, out2

        return False, (
            "empty_response: Model spent all tokens on internal thinking. "
            "Try increasing max_tokens or use a non-thinking model variant."
        )

    if thinking:
        return False, (
            "empty_response: Model returned thinking but no response. "
            "Try increasing max_tokens."
        )

    return False, "empty_response"
