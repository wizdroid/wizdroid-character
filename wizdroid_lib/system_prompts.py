from __future__ import annotations

from typing import Any, Dict

from .content_safety import CONTENT_RATING_CHOICES
from .data_files import load_json, load_text


_DEFAULT_POLICY_FILE = "content_policies.json"


def load_policy_map() -> Dict[str, str]:
    """Load content policies from data/content_policies.json."""

    try:
        payload = load_json(_DEFAULT_POLICY_FILE)
    except Exception:  # noqa: BLE001
        payload = {}

    if not isinstance(payload, dict):
        return {}

    out: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            out[key] = value
    return out


def apply_content_policy(system_prompt: str, content_rating: str) -> str:
    """Append a data-driven content policy block to a system prompt."""

    base = (system_prompt or "").strip()
    policies = load_policy_map()
    policy = policies.get(content_rating) or policies.get("SFW only") or ""
    policy = policy.strip()

    if not policy:
        return base
    if not base:
        return policy

    return f"{base}\n\n{policy}"


def load_system_prompt_text(relative_name: str, content_rating: str) -> str:
    """Load a system prompt from data/ and append the content policy."""

    raw = load_text(relative_name)
    return apply_content_policy(raw, content_rating)


def load_system_prompt_template(relative_name: str, content_rating: str, **kwargs: Any) -> str:
    """Load a templated system prompt and format it with kwargs."""

    raw = load_text(relative_name)
    formatted = raw.format(**kwargs)
    return apply_content_policy(formatted, content_rating)
