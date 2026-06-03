"""Validate all JSON data files in the project."""

import json
from pathlib import Path
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHARED_DIR = DATA_DIR / "shared"

# (relative_path, expected_top_level_type)
DATA_FILE_CHECKS = [
    ("character_options.json", dict),
    ("prompt_styles.json", dict),
    ("regions.json", dict),
    ("countries.json", dict),
    ("cultures.json", dict),
    ("scene_data.json", dict),
    ("meta_prompt_options.json", dict),
    ("prompt_styles.json", dict),
    ("video_camera_data.json", dict),
    ("video_motion_types.json", dict),
    ("video_scene_types.json", dict),
    ("shared/body_types.json", dict),
    ("shared/emotions.json", dict),
    ("shared/eye_colors.json", (list, dict)),
    ("shared/hair.json", dict),
    ("shared/skin_tones.json", (list, dict)),
    ("shared/makeup.json", dict),
    ("shared/poses.json", dict),
    ("shared/backgrounds.json", dict),
    ("shared/camera_lighting.json", dict),
    ("shared/fashion.json", dict),
    ("shared/background_edit.json", dict),
]


def _load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("rel_path,expected_type", DATA_FILE_CHECKS)
class TestDataFiles:
    def test_valid_json(self, rel_path, expected_type):
        path = DATA_DIR / rel_path
        assert path.exists(), f"{rel_path} not found"
        try:
            data = _load(path)
        except json.JSONDecodeError as e:
            pytest.fail(f"{rel_path} is not valid JSON: {e}")
        if isinstance(expected_type, tuple):
            assert isinstance(data, expected_type), \
                f"{rel_path}: expected {expected_type}, got {type(data)}"
        else:
            assert isinstance(data, expected_type), \
                f"{rel_path}: expected {expected_type}, got {type(data)}"

    def test_no_empty_keys(self, rel_path, expected_type):
        """Ensure no empty string keys exist in data."""
        path = DATA_DIR / rel_path
        data = _load(path)

        def _check(obj, path_so_far=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert k != "", f"{rel_path}: empty key at {path_so_far}"
                    _check(v, f"{path_so_far}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _check(item, f"{path_so_far}[{i}]")

        _check(data)


class TestSystemPrompts:
    """Verify all system prompt text files are readable."""

    SYSTEM_PROMPT_DIR = DATA_DIR / "system_prompts"

    def test_all_system_prompts_readable(self):
        assert self.SYSTEM_PROMPT_DIR.exists()
        txt_files = sorted(self.SYSTEM_PROMPT_DIR.glob("*.txt"))
        assert len(txt_files) > 0, "No system prompt files found"
        for path in txt_files:
            content = path.read_text(encoding="utf-8")
            assert len(content) > 50, f"{path.name} is too short ({len(content)} chars)"
            # Check for unformatted template placeholders
            remaining = content.count("{")
            closed = content.count("}")
            assert remaining == closed, \
                f"{path.name}: mismatched braces ({remaining} open, {closed} closed)"
