import random
from typing import Dict, List, Tuple
import logging

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json
from wizdroid_lib.helpers import choose, choose_for_rating, split_groups, with_random
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_text


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
        character_options = load_json("character_options.json")
        prompt_styles = load_json("prompt_styles.json")
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        style_options = [style_key for style_key in prompt_styles.keys()]
        pose_sfw, pose_nsfw = split_groups(character_options.get("pose_style"))
        pose_choices = pose_sfw + pose_nsfw

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "prompt_style": (style_options, {"default": "SDXL"}),
                "retain_face": ("BOOLEAN", {"default": True}),
                "target_face_angle": (with_random(character_options["face_angle"]), {"default": RANDOM_LABEL}),
                "target_camera_angle": (with_random(character_options["camera_angle"]), {"default": RANDOM_LABEL}),
                "target_pose": (with_random(pose_choices), {"default": RANDOM_LABEL}),
                "gender": (with_random(character_options["gender"]), {"default": RANDOM_LABEL}),
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
        content_rating: str,
        prompt_style: str,
        retain_face: bool,
        target_face_angle: str,
        target_camera_angle: str,
        target_pose: str,
        gender: str,
        custom_text: str,
        seed: int = 0,
    ) -> Tuple[str, str]:
        character_options = load_json("character_options.json")
        prompt_styles = load_json("prompt_styles.json")

        rng = random.Random(seed)
        pose_sfw, pose_nsfw = split_groups(character_options.get("pose_style"))

        resolved_face_angle = choose(target_face_angle, character_options["face_angle"], rng)
        resolved_camera_angle = choose(target_camera_angle, character_options["camera_angle"], rng)
        resolved_pose = choose_for_rating(target_pose, pose_sfw, pose_nsfw, content_rating, rng)
        resolved_gender = choose(gender, character_options["gender"], rng)

        # Validate SFW mode doesn't use NSFW pose
        if content_rating == "SFW" and resolved_pose and resolved_pose in set(pose_nsfw):
            blocked = "[ERROR: Selected pose is NSFW but content_rating is 'SFW'. Choose an SFW pose or switch content_rating to 'Mixed' or 'NSFW'.]"
            return (blocked, blocked)

        # Get style configuration
        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]

        system_prompt = load_system_prompt_text(
            "system_prompts/character_edit_system_retain_face.txt" if retain_face else "system_prompts/character_edit_system.txt",
            content_rating,
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

        logger = logging.getLogger(__name__)
        logger.debug(f"[CharacterEdit] Generating edit prompt")
        logger.debug(f"[CharacterEdit] Retain face: {retain_face}")
        logger.debug(f"[CharacterEdit] Target face angle: {resolved_face_angle}")
        logger.debug(f"[CharacterEdit] Target camera angle: {resolved_camera_angle}")
        logger.debug(f"[CharacterEdit] Content rating: {content_rating}")
        logger.debug(f"[CharacterEdit] Target pose: {resolved_pose}")
        logger.debug(f"[CharacterEdit] Gender: {resolved_gender}")
        logger.debug(f"[CharacterEdit] Prompt style: {style_label}")
        logger.debug(f"[CharacterEdit] Using model: {ollama_model}")

        ok, response = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={
                "temperature": 0.8,
                "num_predict": 512,
                "seed": int(seed),
            },
            timeout=120,
        )

        if not ok:
            error_msg = f"Failed to generate prompt: {response}"
            return (error_msg, error_msg)

        if content_rating == "SFW":
            err = enforce_sfw(response)
            if err:
                blocked = "[Blocked: potential NSFW content detected. Switch content_rating to 'Mixed' or 'NSFW'.]"
                return (blocked, blocked)

        return (response, response)


NODE_CLASS_MAPPINGS = {
    "WizdroidCharacterEdit": CharacterEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidCharacterEdit": "Character Edit",
}
