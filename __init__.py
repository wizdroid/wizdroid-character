__version__ = "2025.12.01"

import sys
from pathlib import Path

# Add the base directory to sys.path FIRST so that imports like 'from wizdroid_lib.content_safety' work
# This must happen before any other imports that might trigger node module loading
_BASE_DIR = Path(__file__).resolve().parent
_BASE_DIR_STR = str(_BASE_DIR)
if _BASE_DIR_STR not in sys.path:
    sys.path.insert(0, _BASE_DIR_STR)

import importlib
import importlib.util

# Web directory for custom node UI (icons, colors)
WEB_DIRECTORY = "./web"


def _import_node_module(module_basename: str):
    """Import a node module.

    Works both when this folder is imported as a normal package (ComfyUI) and
    when it is imported as a standalone module (pytest collection), where
    relative imports are not available.
    """

    if __package__:
        try:
            return importlib.import_module(f".nodes.{module_basename}", package=__package__)
        except Exception:
            pass

    file_path = (_BASE_DIR / "nodes" / f"{module_basename}.py").resolve()
    module_name = f"wizdroid_character.nodes.{module_basename}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_basename} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# === Prompt Generation Nodes ===

_character_prompt_node = _import_node_module("character_prompt_node")
CHARACTER_NODE_CLASS_MAPPINGS = _character_prompt_node.NODE_CLASS_MAPPINGS
CHARACTER_DISPLAY_NAME_MAPPINGS = _character_prompt_node.NODE_DISPLAY_NAME_MAPPINGS

_character_edit_node = _import_node_module("qwen_multi_angle_node")
CHARACTER_EDIT_NODE_CLASS_MAPPINGS = _character_edit_node.NODE_CLASS_MAPPINGS
CHARACTER_EDIT_DISPLAY_NAME_MAPPINGS = _character_edit_node.NODE_DISPLAY_NAME_MAPPINGS

_background_edit_node = _import_node_module("background_edit_node")
BG_EDIT_NODE_CLASS_MAPPINGS = _background_edit_node.NODE_CLASS_MAPPINGS
BG_EDIT_DISPLAY_NAME_MAPPINGS = _background_edit_node.NODE_DISPLAY_NAME_MAPPINGS

_scene_generator_node = _import_node_module("scene_generator_node")
SCENE_GENERATOR_NODE_CLASS_MAPPINGS = _scene_generator_node.NODE_CLASS_MAPPINGS
SCENE_GENERATOR_DISPLAY_NAME_MAPPINGS = _scene_generator_node.NODE_DISPLAY_NAME_MAPPINGS

_meta_prompt_node = _import_node_module("meta_prompt_node")
META_PROMPT_NODE_CLASS_MAPPINGS = _meta_prompt_node.META_PROMPT_NODE_CLASS_MAPPINGS
META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS = _meta_prompt_node.META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS

_prompt_combiner_node = _import_node_module("prompt_combiner_node")
COMBINER_NODE_CLASS_MAPPINGS = _prompt_combiner_node.NODE_CLASS_MAPPINGS
COMBINER_DISPLAY_NAME_MAPPINGS = _prompt_combiner_node.NODE_DISPLAY_NAME_MAPPINGS

_image_edit_node = _import_node_module("qwen_image_edit_node")
IMAGE_EDIT_NODE_CLASS_MAPPINGS = _image_edit_node.NODE_CLASS_MAPPINGS
IMAGE_EDIT_DISPLAY_NAME_MAPPINGS = _image_edit_node.NODE_DISPLAY_NAME_MAPPINGS

# === Analysis Nodes ===

_photo_aspect_extractor_node = _import_node_module("photo_aspect_extractor_node")
PHOTO_NODE_CLASS_MAPPINGS = _photo_aspect_extractor_node.NODE_CLASS_MAPPINGS
PHOTO_DISPLAY_NAME_MAPPINGS = _photo_aspect_extractor_node.NODE_DISPLAY_NAME_MAPPINGS

# === Training Nodes ===

_lora_dataset_node = _import_node_module("lora_dataset_node")
LORA_DATASET_NODE_CLASS_MAPPINGS = _lora_dataset_node.NODE_CLASS_MAPPINGS
LORA_DATASET_DISPLAY_NAME_MAPPINGS = _lora_dataset_node.NODE_DISPLAY_NAME_MAPPINGS

# === Utility Nodes ===

_utility_nodes = _import_node_module("utility_nodes")
UTILITY_NODE_CLASS_MAPPINGS = _utility_nodes.NODE_CLASS_MAPPINGS
UTILITY_DISPLAY_NAME_MAPPINGS = _utility_nodes.NODE_DISPLAY_NAME_MAPPINGS

# === Video Nodes ===

_video_scene_expander_node = _import_node_module("video_scene_expander_node")
VIDEO_SCENE_EXPANDER_CLASS_MAPPINGS = _video_scene_expander_node.NODE_CLASS_MAPPINGS
VIDEO_SCENE_EXPANDER_DISPLAY_NAME_MAPPINGS = _video_scene_expander_node.NODE_DISPLAY_NAME_MAPPINGS

_video_prompt_builder_node = _import_node_module("video_prompt_builder_node")
VIDEO_PROMPT_BUILDER_CLASS_MAPPINGS = _video_prompt_builder_node.NODE_CLASS_MAPPINGS
VIDEO_PROMPT_BUILDER_DISPLAY_NAME_MAPPINGS = _video_prompt_builder_node.NODE_DISPLAY_NAME_MAPPINGS

_temporal_scene_planner_node = _import_node_module("temporal_scene_planner_node")
TEMPORAL_SCENE_PLANNER_CLASS_MAPPINGS = _temporal_scene_planner_node.NODE_CLASS_MAPPINGS
TEMPORAL_SCENE_PLANNER_DISPLAY_NAME_MAPPINGS = _temporal_scene_planner_node.NODE_DISPLAY_NAME_MAPPINGS

_motion_choreographer_node = _import_node_module("motion_choreographer_node")
MOTION_CHOREOGRAPHER_CLASS_MAPPINGS = _motion_choreographer_node.NODE_CLASS_MAPPINGS
MOTION_CHOREOGRAPHER_DISPLAY_NAME_MAPPINGS = _motion_choreographer_node.NODE_DISPLAY_NAME_MAPPINGS

_cinematic_shot_designer_node = _import_node_module("cinematic_shot_designer_node")
CINEMATIC_SHOT_DESIGNER_CLASS_MAPPINGS = _cinematic_shot_designer_node.NODE_CLASS_MAPPINGS
CINEMATIC_SHOT_DESIGNER_DISPLAY_NAME_MAPPINGS = _cinematic_shot_designer_node.NODE_DISPLAY_NAME_MAPPINGS

_i2v_animation_describer_node = _import_node_module("i2v_animation_describer_node")
I2V_ANIMATION_DESCRIBER_CLASS_MAPPINGS = _i2v_animation_describer_node.NODE_CLASS_MAPPINGS
I2V_ANIMATION_DESCRIBER_DISPLAY_NAME_MAPPINGS = _i2v_animation_describer_node.NODE_DISPLAY_NAME_MAPPINGS

_prompt_relay_generator_node = _import_node_module("prompt_relay_generator_node")
PROMPT_RELAY_GENERATOR_CLASS_MAPPINGS = _prompt_relay_generator_node.NODE_CLASS_MAPPINGS
PROMPT_RELAY_GENERATOR_DISPLAY_NAME_MAPPINGS = _prompt_relay_generator_node.NODE_DISPLAY_NAME_MAPPINGS

_multi_clip_story_planner_node = _import_node_module("multi_clip_story_planner_node")
MULTI_CLIP_STORY_PLANNER_CLASS_MAPPINGS = _multi_clip_story_planner_node.NODE_CLASS_MAPPINGS
MULTI_CLIP_STORY_PLANNER_DISPLAY_NAME_MAPPINGS = _multi_clip_story_planner_node.NODE_DISPLAY_NAME_MAPPINGS

_image_to_video_adapter_node = _import_node_module("image_to_video_adapter_node")
IMAGE_TO_VIDEO_ADAPTER_CLASS_MAPPINGS = _image_to_video_adapter_node.NODE_CLASS_MAPPINGS
IMAGE_TO_VIDEO_ADAPTER_DISPLAY_NAME_MAPPINGS = _image_to_video_adapter_node.NODE_DISPLAY_NAME_MAPPINGS

_video_negative_prompt_node = _import_node_module("video_negative_prompt_node")
VIDEO_NEGATIVE_PROMPT_CLASS_MAPPINGS = _video_negative_prompt_node.NODE_CLASS_MAPPINGS
VIDEO_NEGATIVE_PROMPT_DISPLAY_NAME_MAPPINGS = _video_negative_prompt_node.NODE_DISPLAY_NAME_MAPPINGS

# === Combined Mappings ===

NODE_CLASS_MAPPINGS = {
    # Prompts
    **CHARACTER_NODE_CLASS_MAPPINGS,
    **CHARACTER_EDIT_NODE_CLASS_MAPPINGS,
    **BG_EDIT_NODE_CLASS_MAPPINGS,
    **SCENE_GENERATOR_NODE_CLASS_MAPPINGS,
    **META_PROMPT_NODE_CLASS_MAPPINGS,
    **COMBINER_NODE_CLASS_MAPPINGS,
    **IMAGE_EDIT_NODE_CLASS_MAPPINGS,
    # Analysis
    **PHOTO_NODE_CLASS_MAPPINGS,
    # Training
    **LORA_DATASET_NODE_CLASS_MAPPINGS,
    # Utilities
    **UTILITY_NODE_CLASS_MAPPINGS,
    # Video
    **VIDEO_SCENE_EXPANDER_CLASS_MAPPINGS,
    **VIDEO_PROMPT_BUILDER_CLASS_MAPPINGS,
    **TEMPORAL_SCENE_PLANNER_CLASS_MAPPINGS,
    **MOTION_CHOREOGRAPHER_CLASS_MAPPINGS,
    **CINEMATIC_SHOT_DESIGNER_CLASS_MAPPINGS,
    **I2V_ANIMATION_DESCRIBER_CLASS_MAPPINGS,
    **PROMPT_RELAY_GENERATOR_CLASS_MAPPINGS,
    **MULTI_CLIP_STORY_PLANNER_CLASS_MAPPINGS,
    **IMAGE_TO_VIDEO_ADAPTER_CLASS_MAPPINGS,
    **VIDEO_NEGATIVE_PROMPT_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Prompts
    **CHARACTER_DISPLAY_NAME_MAPPINGS,
    **CHARACTER_EDIT_DISPLAY_NAME_MAPPINGS,
    **BG_EDIT_DISPLAY_NAME_MAPPINGS,
    **SCENE_GENERATOR_DISPLAY_NAME_MAPPINGS,
    **META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS,
    **COMBINER_DISPLAY_NAME_MAPPINGS,
    **IMAGE_EDIT_DISPLAY_NAME_MAPPINGS,
    # Analysis
    **PHOTO_DISPLAY_NAME_MAPPINGS,
    # Utilities
    **UTILITY_DISPLAY_NAME_MAPPINGS,
    # Training
    **LORA_DATASET_DISPLAY_NAME_MAPPINGS,
    # Video
    **VIDEO_SCENE_EXPANDER_DISPLAY_NAME_MAPPINGS,
    **VIDEO_PROMPT_BUILDER_DISPLAY_NAME_MAPPINGS,
    **TEMPORAL_SCENE_PLANNER_DISPLAY_NAME_MAPPINGS,
    **MOTION_CHOREOGRAPHER_DISPLAY_NAME_MAPPINGS,
    **CINEMATIC_SHOT_DESIGNER_DISPLAY_NAME_MAPPINGS,
    **I2V_ANIMATION_DESCRIBER_DISPLAY_NAME_MAPPINGS,
    **PROMPT_RELAY_GENERATOR_DISPLAY_NAME_MAPPINGS,
    **MULTI_CLIP_STORY_PLANNER_DISPLAY_NAME_MAPPINGS,
    **IMAGE_TO_VIDEO_ADAPTER_DISPLAY_NAME_MAPPINGS,
    **VIDEO_NEGATIVE_PROMPT_DISPLAY_NAME_MAPPINGS,
}
