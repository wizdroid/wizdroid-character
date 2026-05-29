"""Shared helper functions for wizdroid-character nodes."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .constants import NONE_LABEL, RANDOM_LABEL


def option_name(item: Any) -> Optional[str]:
    """Extract name from an option item (string or dict)."""
    if isinstance(item, str):
        name = item.strip()
        return name or None
    if isinstance(item, dict):
        name = str(item.get("name") or item.get("value") or item.get("label") or "").strip()
        return name or None
    return None


def option_description(item: Any) -> str:
    """Extract description from an option item."""
    if isinstance(item, dict):
        desc = item.get("description")
        if desc is None:
            desc = item.get("desc")
        return str(desc or "").strip()
    return ""


def normalize_option_list(options: Any) -> List[str]:
    """Normalize a list of option items into a list of option names.

    Supports either:
    - ["a", "b", ...]
    - [{"name": "a", "description": "..."}, ...]
    - {"key": ["a", ...], ...} (grouped dict — flattens all values)
    """
    if isinstance(options, dict):
        # Flatten grouped dict (e.g. regions grouped by continent)
        flat: List[str] = []
        for val in options.values():
            if isinstance(val, list):
                for item in val:
                    name = option_name(item)
                    if name:
                        flat.append(name)
        return flat
    if not isinstance(options, list):
        return []

    out: List[str] = []
    for item in options:
        name = option_name(item)
        if name:
            out.append(name)
    return out


def extract_descriptions(payload: Any) -> Dict[str, str]:
    """Build {option_name: description} mapping from list payload."""
    out: Dict[str, str] = {}
    groups: List[Any] = []
    if isinstance(payload, list):
        groups = [payload]
    elif isinstance(payload, dict):
        # Flatten grouped payloads (e.g. gender-split categories)
        for val in payload.values():
            if isinstance(val, list):
                groups.append(val)
    for group in groups:
        for item in group:
            name = option_name(item)
            if not name:
                continue
            desc = option_description(item)
            if desc:
                out[name] = desc
    return out


def with_random(options: Any) -> Tuple[str, ...]:
    """Prepend Random and none options to a list."""
    normalized = normalize_option_list(options) if not isinstance(options, list) or (options and isinstance(options[0], dict)) else options
    if isinstance(options, list) and options and isinstance(options[0], str):
        normalized = options
    else:
        normalized = normalize_option_list(options)
    
    values: List[str] = [RANDOM_LABEL, NONE_LABEL]
    for option in normalized:
        if option and option != NONE_LABEL:
            values.append(option)
    return tuple(values)


def with_random_tuple(options: Tuple[str, ...]) -> Tuple[str, ...]:
    """Prepend Random and none to a tuple of options."""
    return (RANDOM_LABEL, NONE_LABEL, *options)


def choose(value: Optional[str], options: List[Any], rng: random.Random, seed: Optional[int] = None) -> Optional[str]:
    """Select a value, handling Random and none cases.
    
    If seed is provided, uses sequential iteration (seed % len) instead of random.
    This allows incrementing seed from 0 to iterate through the list in order.
    """
    # Normalize if needed: dict (grouped), list-of-dicts, or plain string list
    needs_norm = isinstance(options, dict) or (isinstance(options, list) and options and isinstance(options[0], dict))
    normalized = normalize_option_list(options) if needs_norm else options
    if value == RANDOM_LABEL:
        pool = [opt for opt in normalized if opt != NONE_LABEL]
        if pool:
            if seed:
                # Sequential iteration: use seed as index
                selection = pool[seed % len(pool)]
            else:
                selection = rng.choice(pool)
        else:
            selection = None
    else:
        selection = value
    return None if selection == NONE_LABEL or selection is None else selection


def choose_tuple(value: Optional[str], options: Tuple[str, ...], rng: random.Random) -> Optional[str]:
    """Select a value from a tuple, handling Random and none cases."""
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt not in (NONE_LABEL, RANDOM_LABEL)]
        if not pool:
            pool = list(options)
        selection = rng.choice(pool)
    else:
        selection = value

    if selection == NONE_LABEL or selection is None:
        return None
    return selection


def split_groups(payload: Any) -> List[str]:
    """Normalize a payload into a flat list of option names."""
    result: List[str] = []
    if isinstance(payload, list):
        for item in payload:
            name = option_name(item)
            if name:
                result.append(name)
    elif isinstance(payload, dict):
        # Flatten grouped payloads
        for val in payload.values():
            if isinstance(val, list):
                for item in val:
                    name = option_name(item)
                    if name:
                        result.append(name)
    return result



