"""Shared helpers for wizdroid-character nodes.

This module provides centralized utilities for:
- Data loading and caching (registry)
- Content safety filtering
- Ollama API integration
- System prompt management
- Common helper functions
"""

from .constants import (
    CONTENT_RATING_CHOICES,
    DEFAULT_OLLAMA_URL,
    NONE_LABEL,
    RANDOM_LABEL,
)
from .registry import DataRegistry, get_character_options, get_prompt_styles
from .ollama_client import collect_models, generate_text
from .content_safety import enforce_sfw, looks_nsfw

__all__ = [
    # Constants
    "CONTENT_RATING_CHOICES",
    "DEFAULT_OLLAMA_URL",
    "NONE_LABEL",
    "RANDOM_LABEL",
    # Registry
    "DataRegistry",
    "get_character_options",
    "get_prompt_styles",
    # Ollama
    "collect_models",
    "generate_text",
    # Safety
    "enforce_sfw",
    "looks_nsfw",
]
