"""Unit tests for wizdroid_lib helper functions."""

import random
import pytest
from wizdroid_lib.helpers import (
    option_name,
    option_description,
    normalize_option_list,
    extract_descriptions,
    with_random,
    choose,
    split_groups,
)


class TestOptionName:
    def test_string_input(self):
        assert option_name("hello") == "hello"

    def test_dict_input(self):
        assert option_name({"name": "test", "description": "desc"}) == "test"

    def test_dict_with_value(self):
        assert option_name({"value": "val"}) == "val"

    def test_empty_string(self):
        assert option_name("") is None

    def test_none(self):
        assert option_name(None) is None


class TestNormalizeOptionList:
    def test_list_of_strings(self):
        assert normalize_option_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_list_of_dicts(self):
        data = [{"name": "a"}, {"name": "b"}]
        assert normalize_option_list(data) == ["a", "b"]

    def test_grouped_dict(self):
        data = {"group1": [{"name": "a"}, {"name": "b"}], "group2": [{"name": "c"}]}
        result = normalize_option_list(data)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert len(result) == 3

    def test_empty_list(self):
        assert normalize_option_list([]) == []

    def test_none(self):
        assert normalize_option_list(None) == []


class TestWithRandom:
    def test_list_of_strings(self):
        result = with_random(["x", "y"])
        assert result[0] == "Random"
        assert result[1] == "none"
        assert "x" in result
        assert "y" in result

    def test_list_of_dicts(self):
        result = with_random([{"name": "a"}, {"name": "b"}])
        assert "a" in result
        assert "b" in result

    def test_grouped_dict(self):
        data = {"g1": [{"name": "x"}], "g2": [{"name": "y"}]}
        result = with_random(data)
        assert "x" in result
        assert "y" in result


class TestChoose:
    def test_specific_value(self):
        rng = random.Random(42)
        assert choose("hello", ["a", "b", "hello"], rng) == "hello"

    def test_random_selection(self):
        rng = random.Random(42)
        result = choose("Random", ["a", "b", "c"], rng)
        assert result in ["a", "b", "c"]

    def test_none_label(self):
        rng = random.Random(42)
        assert choose("none", ["a", "b"], rng) is None

    def test_list_of_dicts(self):
        rng = random.Random(42)
        result = choose("Random", [{"name": "x"}, {"name": "y"}], rng)
        assert result in ["x", "y"]

    def test_grouped_dict(self):
        rng = random.Random(42)
        data = {"g1": [{"name": "a"}, {"name": "b"}], "g2": [{"name": "c"}]}
        result = choose("Random", data, rng)
        assert result in ["a", "b", "c"]

    def test_seed_sequential(self):
        """With seed>0, should iterate through list sequentially."""
        rng = random.Random(42)
        opts = ["a", "b", "c"]
        result1 = choose("Random", opts, rng, seed=1)
        result2 = choose("Random", opts, rng, seed=2)
        result3 = choose("Random", opts, rng, seed=3)
        assert result1 == "b"  # 1 % 3 = 1
        assert result2 == "c"  # 2 % 3 = 2
        assert result3 == "a"  # 3 % 3 = 0


class TestSplitGroups:
    def test_flat_list(self):
        data = [{"name": "a"}, {"name": "b"}]
        assert split_groups(data) == ["a", "b"]

    def test_grouped_dict(self):
        data = {"g1": [{"name": "x"}], "g2": [{"name": "y"}]}
        result = split_groups(data)
        assert "x" in result
        assert "y" in result
        assert len(result) == 2


class TestExtractDescriptions:
    def test_flat_list(self):
        data = [{"name": "a", "description": "desc a"}]
        assert extract_descriptions(data) == {"a": "desc a"}

    def test_grouped_dict(self):
        data = {"g1": [{"name": "a", "description": "desc a"}]}
        assert extract_descriptions(data) == {"a": "desc a"}
