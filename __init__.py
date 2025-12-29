__version__ = "2025.11.01"

import importlib
import importlib.util
import sys
from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parent

# Add the base directory to sys.path so that imports like 'from lib.content_safety' work
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))


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


_character_prompt_node = _import_node_module("character_prompt_node")
CHARACTER_NODE_CLASS_MAPPINGS = _character_prompt_node.NODE_CLASS_MAPPINGS
CHARACTER_DISPLAY_NAME_MAPPINGS = _character_prompt_node.NODE_DISPLAY_NAME_MAPPINGS

_photo_aspect_extractor_node = _import_node_module("photo_aspect_extractor_node")
PHOTO_NODE_CLASS_MAPPINGS = _photo_aspect_extractor_node.NODE_CLASS_MAPPINGS
PHOTO_DISPLAY_NAME_MAPPINGS = _photo_aspect_extractor_node.NODE_DISPLAY_NAME_MAPPINGS

_character_edit_node = _import_node_module("character_edit_node")
EDIT_NODE_CLASS_MAPPINGS = _character_edit_node.NODE_CLASS_MAPPINGS
EDIT_DISPLAY_NAME_MAPPINGS = _character_edit_node.NODE_DISPLAY_NAME_MAPPINGS

_background_edit_node = _import_node_module("background_edit_node")
BG_EDIT_NODE_CLASS_MAPPINGS = _background_edit_node.NODE_CLASS_MAPPINGS
BG_EDIT_DISPLAY_NAME_MAPPINGS = _background_edit_node.NODE_DISPLAY_NAME_MAPPINGS

_scene_generator_node = _import_node_module("scene_generator_node")
SCENE_GENERATOR_NODE_CLASS_MAPPINGS = _scene_generator_node.NODE_CLASS_MAPPINGS
SCENE_GENERATOR_DISPLAY_NAME_MAPPINGS = _scene_generator_node.NODE_DISPLAY_NAME_MAPPINGS

_prompt_combiner_node = _import_node_module("prompt_combiner_node")
COMBINER_NODE_CLASS_MAPPINGS = _prompt_combiner_node.NODE_CLASS_MAPPINGS
COMBINER_DISPLAY_NAME_MAPPINGS = _prompt_combiner_node.NODE_DISPLAY_NAME_MAPPINGS

_lora_dataset_node = _import_node_module("lora_dataset_node")
LORA_DATASET_NODE_CLASS_MAPPINGS = _lora_dataset_node.NODE_CLASS_MAPPINGS
LORA_DATASET_DISPLAY_NAME_MAPPINGS = _lora_dataset_node.NODE_DISPLAY_NAME_MAPPINGS

_lora_train_node = _import_node_module("lora_train_node")
LORA_TRAIN_NODE_CLASS_MAPPINGS = _lora_train_node.NODE_CLASS_MAPPINGS
LORA_TRAIN_DISPLAY_NAME_MAPPINGS = _lora_train_node.NODE_DISPLAY_NAME_MAPPINGS

_lora_validate_node = _import_node_module("lora_validate_node")
LORA_VALIDATE_NODE_CLASS_MAPPINGS = _lora_validate_node.NODE_CLASS_MAPPINGS
LORA_VALIDATE_DISPLAY_NAME_MAPPINGS = _lora_validate_node.NODE_DISPLAY_NAME_MAPPINGS

_lora_dataset_validator_node = _import_node_module("lora_dataset_validator_node")
LORA_VALIDATOR_NODE_CLASS_MAPPINGS = _lora_dataset_validator_node.NODE_CLASS_MAPPINGS
LORA_VALIDATOR_DISPLAY_NAME_MAPPINGS = _lora_dataset_validator_node.NODE_DISPLAY_NAME_MAPPINGS

_meta_prompt_node = _import_node_module("meta_prompt_node")
META_PROMPT_NODE_CLASS_MAPPINGS = _meta_prompt_node.META_PROMPT_NODE_CLASS_MAPPINGS
META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS = _meta_prompt_node.META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS

_contest_prompt_node = _import_node_module("contest_prompt_node")
CONTEST_NODE_CLASS_MAPPINGS = _contest_prompt_node.NODE_CLASS_MAPPINGS
CONTEST_DISPLAY_NAME_MAPPINGS = _contest_prompt_node.NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    **CHARACTER_NODE_CLASS_MAPPINGS,
    **PHOTO_NODE_CLASS_MAPPINGS,
    **EDIT_NODE_CLASS_MAPPINGS,
    **BG_EDIT_NODE_CLASS_MAPPINGS,
    **SCENE_GENERATOR_NODE_CLASS_MAPPINGS,
    **COMBINER_NODE_CLASS_MAPPINGS,
    **LORA_DATASET_NODE_CLASS_MAPPINGS,
    **LORA_TRAIN_NODE_CLASS_MAPPINGS,
    **LORA_VALIDATE_NODE_CLASS_MAPPINGS,
    **LORA_VALIDATOR_NODE_CLASS_MAPPINGS,
    **META_PROMPT_NODE_CLASS_MAPPINGS,
    **CONTEST_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CHARACTER_DISPLAY_NAME_MAPPINGS,
    **PHOTO_DISPLAY_NAME_MAPPINGS,
    **EDIT_DISPLAY_NAME_MAPPINGS,
    **BG_EDIT_DISPLAY_NAME_MAPPINGS,
    **SCENE_GENERATOR_DISPLAY_NAME_MAPPINGS,
    **COMBINER_DISPLAY_NAME_MAPPINGS,
    **LORA_DATASET_DISPLAY_NAME_MAPPINGS,
    **LORA_TRAIN_DISPLAY_NAME_MAPPINGS,
    **LORA_VALIDATE_DISPLAY_NAME_MAPPINGS,
    **LORA_VALIDATOR_DISPLAY_NAME_MAPPINGS,
    **META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS,
    **CONTEST_DISPLAY_NAME_MAPPINGS,
}
