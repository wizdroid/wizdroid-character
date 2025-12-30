from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .paths import DATA_DIR, data_path

# mtime(ns) -> payload cache
_JSON_CACHE: Dict[str, Tuple[int, Any]] = {}
_TEXT_CACHE: Dict[str, Tuple[int, str]] = {}


def load_json(relative_name: str) -> Any:
    """Load a JSON file from data/ with mtime caching."""

    path = data_path(relative_name)
    mtime = int(path.stat().st_mtime_ns)
    cached = _JSON_CACHE.get(relative_name)
    if cached and cached[0] == mtime:
        return cached[1]

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    _JSON_CACHE[relative_name] = (mtime, payload)
    return payload


def load_text(relative_name: str) -> str:
    """Load a UTF-8 text file from data/ with mtime caching."""

    path = data_path(relative_name)
    mtime = int(path.stat().st_mtime_ns)
    cached = _TEXT_CACHE.get(relative_name)
    if cached and cached[0] == mtime:
        return cached[1]

    text = path.read_text(encoding="utf-8")
    _TEXT_CACHE[relative_name] = (mtime, text)
    return text


def exists(relative_name: str) -> bool:
    return data_path(relative_name).exists()


def list_files(glob_pattern: str) -> list[str]:
    return sorted([p.name for p in DATA_DIR.glob(glob_pattern) if p.is_file()])
