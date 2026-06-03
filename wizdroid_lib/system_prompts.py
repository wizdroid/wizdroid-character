from __future__ import annotations

from typing import Any

from .data_files import load_text


def load_system_prompt_text(relative_name: str) -> str:
    """Load a system prompt from data/."""

    return load_text(relative_name)


def load_system_prompt_template(relative_name: str, **kwargs: Any) -> str:
    """Load a templated system prompt and format it with kwargs."""

    raw = load_text(relative_name)
    return raw.format(**kwargs)
