"""Shared constants for wizdroid-character nodes."""

from __future__ import annotations

# UI Labels
RANDOM_LABEL = "Random"
NONE_LABEL = "none"

# Default Ollama URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Vision model name keywords (for legacy models that may not report "capabilities")
VISION_KEYWORDS = {
    "vision",
    "vl",
    "llava",
    "bakllava",
    "moondream",
    "cogvlm",
    "blip",
    "instructblip",
    "minigpt",
    "mplug",
    "qwen-vl",
    "qwen2-vl",
    "florence",
    "idefics",
    "fuyu",
    "llava-phi",
}
