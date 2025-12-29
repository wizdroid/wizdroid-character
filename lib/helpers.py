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
    """
    if not isinstance(options, list):
        return []

    out: List[str] = []
    for item in options:
        name = option_name(item)
        if name:
            out.append(name)
    return out


def extract_descriptions(payload: Any) -> Dict[str, str]:
    """Build {option_name: description} mapping from list/dict(SFW/NSFW) payload."""
    out: Dict[str, str] = {}
    if isinstance(payload, dict):
        groups = [payload.get("sfw") or [], payload.get("nsfw") or []]
    elif isinstance(payload, list):
        groups = [payload]
    else:
        groups = []

    for group in groups:
        if not isinstance(group, list):
            continue
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


def choose(value: Optional[str], options: List[Any], rng: random.Random) -> Optional[str]:
    """Select a value, handling Random and none cases."""
    normalized = normalize_option_list(options) if options and isinstance(options[0], dict) else options
    if value == RANDOM_LABEL:
        pool = [opt for opt in normalized if opt != NONE_LABEL]
        selection = rng.choice(pool) if pool else None
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


def split_groups(payload: Any) -> Tuple[List[str], List[str]]:
    """Split payload into SFW and NSFW lists."""
    if isinstance(payload, dict):
        sfw_raw = payload.get("sfw") or []
        nsfw_raw = payload.get("nsfw") or []
        sfw: List[str] = []
        nsfw: List[str] = []
        if isinstance(sfw_raw, list):
            for item in sfw_raw:
                name = option_name(item)
                if name:
                    sfw.append(name)
        if isinstance(nsfw_raw, list):
            for item in nsfw_raw:
                name = option_name(item)
                if name:
                    nsfw.append(name)
        return sfw, nsfw
    elif isinstance(payload, list):
        sfw = []
        for item in payload:
            name = option_name(item)
            if name:
                sfw.append(name)
        return sfw, []
    return [], []


def pool_for_rating(rating: str, sfw: List[str], nsfw: List[str]) -> List[str]:
    """Get appropriate pool based on content rating."""
    if rating == "SFW":
        return sfw
    if rating == "NSFW":
        return nsfw
    return sfw + nsfw  # Mixed


def choose_for_rating(
    value: Optional[str],
    sfw: List[str],
    nsfw: List[str],
    rating: str,
    rng: random.Random,
) -> Optional[str]:
    """Choose value respecting content rating."""
    if value == RANDOM_LABEL:
        pool = [opt for opt in pool_for_rating(rating, sfw, nsfw) if opt != NONE_LABEL]
        if not pool:
            pool = [opt for opt in (sfw + nsfw) if opt != NONE_LABEL]
        return rng.choice(pool) if pool else None
    return None if value == NONE_LABEL or value is None else value


def get_background_groups(payload: Any) -> Dict[str, List[str]]:
    """Extract background style groups from payload."""
    default = {"studio_controlled": [], "public_exotic_real": [], "imaginative_surreal": []}
    if isinstance(payload, dict):
        return {k: list(payload.get(k) or []) for k in default}
    elif isinstance(payload, list):
        return {**default, "studio_controlled": list(payload)}
    return default
