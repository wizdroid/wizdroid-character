"""Shared fixtures for wizdroid-character tests."""

import sys
from pathlib import Path

# Ensure the package root is on sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest

# List of all node module basenames (used by multiple tests)
ALL_NODE_MODULES = [
    "character_prompt_node",
    "meta_prompt_node",
    "scene_generator_node",
    "background_edit_node",
    "prompt_combiner_node",
    "cinematic_shot_designer_node",
    "motion_choreographer_node",
    "video_prompt_builder_node",
    "video_scene_expander_node",
    "image_to_video_adapter_node",
    "i2v_animation_describer_node",
    "temporal_scene_planner_node",
    "multi_clip_story_planner_node",
    "prompt_relay_generator_node",
    "qwen_image_edit_node",
    "video_negative_prompt_node",
    "photo_aspect_extractor_node",
]

# All JSON data files to validate
ALL_DATA_FILES = [
    "character_options.json",
    "prompt_styles.json",
    "content_policies.json",
    "regions.json",
    "countries.json",
    "cultures.json",
    "scene_data.json",
    "meta_prompt_options.json",
    "video_camera_data.json",
    "video_motion_types.json",
    "video_scene_types.json",
    "shared/body_types.json",
    "shared/emotions.json",
    "shared/eye_colors.json",
    "shared/hair.json",
    "shared/skin_tones.json",
    "shared/makeup.json",
    "shared/poses.json",
    "shared/backgrounds.json",
    "shared/camera_lighting.json",
    "shared/fashion.json",
    "shared/background_edit.json",
    "shared/prompt_styles.json",
]

# Nodes that need package-based import (use relative intra-package imports)
_RELATIVE_IMPORT_NODES = {
    "cinematic_shot_designer_node",
    "motion_choreographer_node",
    "video_prompt_builder_node",
    "video_scene_expander_node",
    "image_to_video_adapter_node",
    "i2v_animation_describer_node",
    "temporal_scene_planner_node",
    "multi_clip_story_planner_node",
    "prompt_relay_generator_node",
    "video_negative_prompt_node",
}


def import_node_module(module_basename: str):
    """Import a node module.

    Nodes with relative intra-package imports (e.g. from .video_scene_expander_node)
    must go through __init__.py's _import_node_module to set up the package namespace.
    """
    import importlib.util

    if module_basename in _RELATIVE_IMPORT_NODES:
        # Use __init__.py's import mechanism for nodes with relative imports
        spec = importlib.util.spec_from_file_location(
            f"wizdroid_character.nodes.{module_basename}",
            _root / "nodes" / f"{module_basename}.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {module_basename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"wizdroid_character.nodes.{module_basename}"] = module
        spec.loader.exec_module(module)
        return module

    # Direct file import for nodes without relative imports
    file_path = _root / "nodes" / f"{module_basename}.py"
    module_name = f"wizdroid_character.tests.{module_basename}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
