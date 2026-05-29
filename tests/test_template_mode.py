"""Test template (non-AI) mode for all Ollama-using nodes.

These tests verify that each node produces valid output when use_ai=False,
without needing Ollama running.
"""

import pytest
from .conftest import import_node_module

# Nodes that support use_ai=False template mode
TEMPLATE_NODES = [
    ("character_prompt_node", "WizdroidCharacterPromptMale"),
    ("character_prompt_node", "WizdroidCharacterPromptFemale"),
    ("meta_prompt_node", "WizdroidMetaPrompt"),
    ("scene_generator_node", "WizdroidSceneGenerator"),
    ("background_edit_node", "WizdroidBackground"),
    ("prompt_combiner_node", "WizdroidPromptCombiner"),
    ("cinematic_shot_designer_node", "WizdroidCinematicShotDesigner"),
    ("motion_choreographer_node", "WizdroidMotionChoreographer"),
    ("video_prompt_builder_node", "WizdroidVideoPromptBuilder"),
    ("video_scene_expander_node", "WizdroidVideoSceneExpander"),
    ("image_to_video_adapter_node", "WizdroidImageToVideoAdapter"),
    ("i2v_animation_describer_node", "WizdroidI2VAnimationDescriber"),
    ("temporal_scene_planner_node", "WizdroidTemporalScenePlanner"),
    ("multi_clip_story_planner_node", "WizdroidMultiClipStoryPlanner"),
    ("prompt_relay_generator_node", "WizdroidPromptRelayGenerator"),
    ("qwen_image_edit_node", "WizdroidImageEdit"),
]


def _build_minimal_kwargs(cls_obj):
    """Build minimal keyword arguments for a node's FUNCTION.

    Uses defaults from INPUT_TYPES so template mode can be exercised.
    """
    from wizdroid_lib.constants import RANDOM_LABEL, NONE_LABEL
    types = cls_obj.INPUT_TYPES()
    kwargs = {}
    for section in ("required",):
        for key, spec in types.get(section, {}).items():
            widget_type, config = spec[0], spec[1]
            default = config.get("default", "")
            kwargs[key] = default
    for key, spec in types.get("optional", {}).items():
        widget_type, config = spec[0], spec[1]
        default = config.get("default", "")
        kwargs[key] = default
    # Ensure use_ai is False for template mode
    if "use_ai" in kwargs:
        kwargs["use_ai"] = False
    return kwargs


@pytest.mark.parametrize("mod_name,cls_name", TEMPLATE_NODES)
class TestTemplateMode:
    """Template (non-AI) mode must produce string output without errors."""

    def test_template_returns_string(self, mod_name, cls_name):
        module = import_node_module(mod_name)
        mappings = _get_mappings(module, cls_name)
        cls_obj = mappings[cls_name]
        func = getattr(cls_obj, cls_obj.FUNCTION)
        instance = cls_obj()
        kwargs = _build_minimal_kwargs(cls_obj)
        try:
            result = func(instance, **kwargs)
        except Exception as e:
            pytest.fail(f"{cls_name}.{cls_obj.FUNCTION}() raised: {e}")
        # Result should be a tuple (ComfyUI convention)
        assert isinstance(result, tuple), f"{cls_name}: expected tuple, got {type(result)}"
        assert len(result) == len(cls_obj.RETURN_TYPES)
        # First output should be a non-empty string
        first = result[0]
        assert isinstance(first, str), f"{cls_name}: first output not a string"
        assert len(first) > 0, f"{cls_name}: template mode returned empty string"


def _get_mappings(module, cls_name):
    """Find NODE_CLASS_MAPPINGS containing cls_name."""
    for attr_name in dir(module):
        if "MAPPINGS" in attr_name.upper():
            mappings = getattr(module, attr_name, {})
            if isinstance(mappings, dict) and cls_name in mappings:
                return mappings
    raise KeyError(f"{cls_name} not found in any MAPPINGS in {module}")
