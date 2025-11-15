import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
POSE_RATING_CHOICES = ("SFW only", "NSFW only", "Mixed")
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@lru_cache(maxsize=None)
def _load_json(name: str) -> Dict:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _with_random(options: List[str]) -> Tuple[str, ...]:
    values: List[str] = [RANDOM_LABEL, NONE_LABEL]
    for option in options:
        if option == NONE_LABEL:
            continue
        values.append(option)
    return tuple(values)


def _choose(value: Optional[str], options: List[str], rng: random.Random) -> Optional[str]:
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt != NONE_LABEL]
        if not pool:
            pool = options[:]
        selection = rng.choice(pool)
    else:
        selection = value

    if selection == NONE_LABEL or selection is None:
        return None
    return selection


def _split_pose_groups(payload: Any) -> Tuple[List[str], List[str]]:
    if isinstance(payload, dict):
        sfw = list(payload.get("sfw", []) or [])
        nsfw = list(payload.get("nsfw", []) or [])
    elif isinstance(payload, list):
        sfw = list(payload)
        nsfw = []
    else:
        sfw = []
        nsfw = []
    return sfw, nsfw


def _pose_pool_for_rating(rating: str, sfw: List[str], nsfw: List[str]) -> List[str]:
    if rating == "SFW only":
        return sfw
    if rating == "NSFW only":
        return nsfw
    return sfw + nsfw


def _choose_pose(value: Optional[str], sfw: List[str], nsfw: List[str], rating: str, rng: random.Random) -> Optional[str]:
    combined = sfw + nsfw
    if value == RANDOM_LABEL:
        pool = [opt for opt in _pose_pool_for_rating(rating, sfw, nsfw) if opt != NONE_LABEL]
        if not pool:
            pool = [opt for opt in combined if opt != NONE_LABEL]
        if not pool:
            return None
        selection = rng.choice(pool)
    else:
        selection = value

    if selection == NONE_LABEL or selection is None:
        return None
    return selection





class CharacterEditNode:
    """
    ComfyUI node to generate prompts for editing character images, focusing on
    face angles, camera angles, and poses using Flux Kontext or Qwen Image Edit models.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("edit_prompt", "preview")
    FUNCTION = "generate_edit_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        character_options = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        ollama_models = cls._collect_ollama_models()

        style_options = [style_key for style_key in prompt_styles.keys()]
        pose_sfw, pose_nsfw = _split_pose_groups(character_options.get("pose_style"))
        pose_choices = pose_sfw + pose_nsfw

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (style_options, {"default": "SDXL"}),
                "retain_face": ("BOOLEAN", {"default": True}),
                "target_face_angle": (_with_random(character_options["face_angle"]), {"default": RANDOM_LABEL}),
                "target_camera_angle": (_with_random(character_options["camera_angle"]), {"default": RANDOM_LABEL}),
                "pose_content_rating": (POSE_RATING_CHOICES, {"default": "SFW only"}),
                "target_pose": (_with_random(pose_choices), {"default": RANDOM_LABEL}),
                "gender": (_with_random(character_options["gender"]), {"default": RANDOM_LABEL}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            }
        }

    def generate_edit_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        retain_face: bool,
        target_face_angle: str,
        target_camera_angle: str,
        pose_content_rating: str,
        target_pose: str,
        gender: str,
        custom_text: str,
        seed: int = 0,
    ) -> Tuple[str]:
        character_options = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")

        rng = random.Random(seed)
        pose_sfw, pose_nsfw = _split_pose_groups(character_options.get("pose_style"))

        resolved_face_angle = _choose(target_face_angle, character_options["face_angle"], rng)
        resolved_camera_angle = _choose(target_camera_angle, character_options["camera_angle"], rng)
        resolved_pose = _choose_pose(target_pose, pose_sfw, pose_nsfw, pose_content_rating, rng)
        resolved_gender = _choose(gender, character_options["gender"], rng)

        # Get style configuration
        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        # use style presets; token budget is managed externally by model/service

        # Build the prompt instruction for the LLM
        if retain_face:
            system_prompt = (
                "You are a text-to-image prompt engineer for face-preserving image editing models (Flux Kontext, Qwen Image Edit). "
                "Create concise prompts that preserve the original face while modifying other aspects. "
                "ALWAYS start prompts with 'Retain the facial features from the original image.' Then describe angle, pose, and positioning changes. "
                "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', the model/style name (Flux, SDXL, Qwen, HiDream, etc.), or any meta preface. "
                "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s). "
                "You must honor every provided attribute literally—do not substitute synonyms or reinterpret selections. "
                "Never include reasoning traces, deliberation markers, or text enclosed in '<think>' or similar tags."
            )
        else:
            system_prompt = (
                "You are a text-to-image prompt engineer for image editing models (Flux Kontext, Qwen Image Edit). "
                "Create concise prompts for editing character images. "
                "Focus on face angles, camera angles, and poses. Be specific and descriptive but avoid excessive verbosity. "
                "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', the model/style name (Flux, SDXL, Qwen, HiDream, etc.), or any meta preface. "
                "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s). "
                "You must honor every provided attribute literally—do not substitute synonyms or reinterpret selections. "
                "Never include reasoning traces, deliberation markers, or text enclosed in '<think>' or similar tags."
            )

        # Build attribute list, filtering out None values
        attr_parts = []
        if resolved_face_angle:
            attr_parts.append(f"face angle: {resolved_face_angle}")
        if resolved_camera_angle:
            attr_parts.append(f"camera angle: {resolved_camera_angle}")
        if resolved_pose:
            attr_parts.append(f"pose: {resolved_pose}")
        if resolved_gender:
            attr_parts.append(f"gender: {resolved_gender}")

        if retain_face:
            lines = [
                f"Create a {prompt_style} face-preserving image edit prompt for changing angles and pose:",
                ", ".join(attr_parts),
            ]
        else:
            lines = [
                f"Create a {prompt_style} image edit prompt for changing character angles and pose:",
                ", ".join(attr_parts),
            ]

        if custom_text.strip():
            lines.append(f"\nAdditional notes: {custom_text.strip()}")

        if retain_face:
            lines.extend([
                f"\nFormat: {style_guidance}",
                "CRITICAL - Start with: 'Retain the facial features from the original image.'",
                "Then describe:",
                "  * New face angle and head positioning",
                "  * New camera angle and perspective",
                "  * New pose and body positioning",
                "Do NOT describe facial features (eyes, nose, mouth, face shape) - these are preserved from original",
                "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', or similar",
                "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
                "Remove any reasoning or planning text; do not include '<think>' or similar tags",
                "MANDATORY: Use every specified attribute verbatim (face angle, camera angle, pose, gender); do not invent or alter values",
                "Output only the final prompt text:"
            ])
        else:
            lines.extend([
                f"\nFormat: {style_guidance}",
                "CRITICAL - Describe:",
                "  * Face angle and head positioning",
                "  * Camera angle and perspective",
                "  * Pose and body positioning",
                "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', or similar",
                "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
                "Remove any reasoning or planning text; do not include '<think>' or similar tags",
                "MANDATORY: Use every specified attribute verbatim (face angle, camera angle, pose, gender); do not invent or alter values",
                "Output only the final prompt text:"
            ])

        user_prompt = "\n".join(lines)

        # Ensure URL has the /api/generate endpoint
        generate_url = ollama_url
        if not generate_url.endswith("/api/generate"):
            generate_url = generate_url.rstrip("/") + "/api/generate"

        payload = {
            "model": ollama_model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
            }
        }

        logger = logging.getLogger(__name__)
        logger.debug(f"[CharacterEdit] Generating edit prompt")
        logger.debug(f"[CharacterEdit] Retain face: {retain_face}")
        logger.debug(f"[CharacterEdit] Target face angle: {resolved_face_angle}")
        logger.debug(f"[CharacterEdit] Target camera angle: {resolved_camera_angle}")
        logger.debug(f"[CharacterEdit] Pose rating: {pose_content_rating}")
        logger.debug(f"[CharacterEdit] Target pose: {resolved_pose}")
        logger.debug(f"[CharacterEdit] Gender: {resolved_gender}")
        logger.debug(f"[CharacterEdit] Prompt style: {style_label}")
        logger.debug(f"[CharacterEdit] Using model: {ollama_model}")

        response = self._invoke_ollama(generate_url, payload)

        if not response or response.startswith("[ERROR"):
            error_msg = f"Failed to generate prompt: {response}"
            return (error_msg, error_msg)

        return (response, response)

    @staticmethod
    def _invoke_ollama(ollama_url: str, payload: Dict) -> Optional[str]:
        if requests is None:
            raise RuntimeError("'requests' is required for Ollama integration. Install optional dependencies.")
        try:
            logging.getLogger(__name__).debug(f"[CharacterEdit] Sending request to {ollama_url}")
            logging.getLogger(__name__).debug(f"[CharacterEdit] Model: {payload.get('model')}")

            response = requests.post(ollama_url, json=payload, timeout=120)

            logging.getLogger(__name__).debug(f"[CharacterEdit] Response status: {response.status_code}")

            response.raise_for_status()
            data = response.json()

            result = (data.get("response") or "").strip()
            logging.getLogger(__name__).debug(f"[CharacterEdit] Received response ({len(result)} chars): {result[:100]}...")

            if not result:
                logging.getLogger(__name__).warning("[CharacterEdit] WARNING: Empty response from Ollama")
                return "[Empty response from LLM]"

            return result
        except requests.exceptions.HTTPError as exc:
            error_msg = f"[CharacterEdit] HTTP error: {exc}"
            logging.getLogger(__name__).error(error_msg)
            if hasattr(exc.response, 'text'):
                logging.getLogger(__name__).error(f"[CharacterEdit] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            error_msg = f"[CharacterEdit] Connection error: {exc}"
            logging.getLogger(__name__).warning(error_msg)
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            error_msg = f"[CharacterEdit] Timeout error: {exc}"
            logging.getLogger(__name__).warning(error_msg)
            return "[ERROR: Request timed out]"
        except Exception as exc:
            error_msg = f"[CharacterEdit] Error invoking Ollama: {exc}"
            logging.getLogger(__name__).exception(error_msg)
            return f"[ERROR: {str(exc)}]"

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        if requests is None:
            return ["install_requests_library"]
        try:
            tags_url = f"{ollama_url}/api/tags"
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            all_models = [model["name"] for model in models_data.get("models", [])]

            if not all_models:
                return ["no_models_found"]

            return all_models
        except requests.exceptions.ConnectionError:
            return ["ollama_not_running"]
        except requests.exceptions.Timeout:
            return ["ollama_timeout"]
        except Exception as exc:
            logging.getLogger(__name__).exception(f"[CharacterEdit] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "WizdroidCharacterEdit": CharacterEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidCharacterEdit": "Character Edit",
}