from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .paths import DATA_DIR, SHARED_DIR, data_path, shared_path

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


def load_shared(relative_name: str) -> Any:
    """Load a JSON file from data/shared/ with mtime caching."""

    cache_key = f"shared/{relative_name}"
    path = shared_path(relative_name)
    mtime = int(path.stat().st_mtime_ns)
    cached = _JSON_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    _JSON_CACHE[cache_key] = (mtime, payload)
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


def filter_by_gender(items: Any, gender: Optional[str] = None) -> List[Any]:
    """Filter items by gender tag. Returns items matching gender or 'any'.
    
    Supports both list-of-dicts (with 'gender' key) and plain lists (returned as-is).
    If gender is None or empty, returns all items.
    """
    if not isinstance(items, list):
        return []
    
    if not items or not isinstance(items[0], dict):
        return items  # plain string list, no filtering
    
    if not gender or gender.lower() in ("", "none", "non-binary"):
        return items  # return all items
    
    g = gender.lower()
    result = []
    for item in items:
        item_genders = item.get("gender", ["any"])
        if "any" in item_genders or g in item_genders:
            result.append(item)
    return result
