from __future__ import annotations

from typing import Any

from .data_files import load_json, load_text


_DEFAULT_POLICY_FILE = "content_policies.json"


def load_content_policy() -> str:
    """Load the default content policy from data/content_policies.json."""

    try:
        payload = load_json(_DEFAULT_POLICY_FILE)
    except Exception:  # noqa: BLE001
        payload = {}

    if isinstance(payload, dict):
        # Return the first (or 'default') policy found
        return str(payload.get("default") or next(iter(payload.values()), ""))
    return ""


def apply_content_policy(system_prompt: str) -> str:
    """Append a data-driven content policy block to a system prompt."""

    base = (system_prompt or "").strip()
    policy = load_content_policy().strip()

    if not policy:
        return base
    if not base:
        return policy

    return f"{base}\n\n{policy}"


def load_system_prompt_text(relative_name: str) -> str:
    """Load a system prompt from data/ and append the content policy."""

    raw = load_text(relative_name)
    return apply_content_policy(raw)


def load_system_prompt_template(relative_name: str, **kwargs: Any) -> str:
    """Load a templated system prompt and format it with kwargs."""

    raw = load_text(relative_name)
    formatted = raw.format(**kwargs)
    return apply_content_policy(formatted)
