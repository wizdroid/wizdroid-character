#!/usr/bin/env python3
"""
Simple tests for image category sfw/nsfw selection and rating logic.
"""
import random
from nodes import character_prompt_node as cpn


def test_image_category_rating_sfw_only():
    option_map = cpn._load_json("character_options.json")
    sfw, nsfw = cpn._split_groups(option_map.get("image_category"))
    rng = random.Random(42)
    # ask for random but SFW only
    result = cpn._choose_for_rating(cpn.RANDOM_LABEL, sfw, nsfw, "SFW only", rng)
    assert result in sfw or result is None


def test_image_category_rating_nsfw_only():
    option_map = cpn._load_json("character_options.json")
    sfw, nsfw = cpn._split_groups(option_map.get("image_category"))
    rng = random.Random(42)
    # ask for random but NSFW only
    result = cpn._choose_for_rating(cpn.RANDOM_LABEL, sfw, nsfw, "NSFW only", rng)
    assert result in nsfw or result is None


def test_image_category_rating_mixed():
    option_map = cpn._load_json("character_options.json")
    sfw, nsfw = cpn._split_groups(option_map.get("image_category"))
    rng = random.Random(42)
    result = cpn._choose_for_rating(cpn.RANDOM_LABEL, sfw, nsfw, "Mixed", rng)
    assert result in sfw + nsfw or result is None
