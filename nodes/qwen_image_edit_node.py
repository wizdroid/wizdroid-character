import random
from typing import Tuple
import logging

from wizdroid_lib.constants import DEFAULT_OLLAMA_URL
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_text


# Preset edit modes for common use cases
EDIT_MODE_CHOICES = (
    "Face Swap (Image 1 → Image 2)",
    "Face Swap (Image 1 → Image 3)",
    "Style Transfer (Image 1 style → Image 2)",
    "Triple Blend",
    "Custom",
)


class QwenImageEditNode:
    """
    ComfyUI node to generate prompts for Qwen Image Edit or similar multi-image
    editing models. Supports up to 3 image descriptions with LLM-generated
    editing instructions.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("edit_prompt", "preview")
    FUNCTION = "generate_edit_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "edit_mode": (EDIT_MODE_CHOICES, {"default": "Face Swap (Image 1 → Image 2)"}),
                "image_1_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe Image 1 (e.g., 'Source face - young woman with blue eyes')"
                }),
                "image_2_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe Image 2 (e.g., 'Target scene - person in red dress at beach')"
                }),
                "image_3_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe Image 3 (optional - for triple blend modes)"
                }),
                "additional_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Any additional editing instructions or constraints"
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            }
        }

    def generate_edit_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        edit_mode: str,
        image_1_description: str,
        image_2_description: str,
        image_3_description: str,
        additional_instructions: str,
        seed: int = 0,
    ) -> Tuple[str, str]:
        logger = logging.getLogger(__name__)

        # Validate inputs
        img1 = image_1_description.strip()
        img2 = image_2_description.strip()
        img3 = image_3_description.strip()

        if not img1:
            error = "[ERROR: Image 1 description is required]"
            return (error, error)

        if not img2:
            error = "[ERROR: Image 2 description is required]"
            return (error, error)

        # Build the mode-specific guidance
        mode_guidance = self._get_mode_guidance(edit_mode, img1, img2, img3)

        # Load system prompt (using SFW as default since this is for editing)
        system_prompt = load_system_prompt_text(
            "system_prompts/qwen_image_edit_system.txt",
            "SFW",
        )

        # Build user prompt
        lines = [
            "Generate a detailed image editing prompt for a multi-image AI model.",
            "",
            f"Edit Mode: {edit_mode}",
            "",
            "Image Descriptions:",
            f"- Image 1: {img1}",
            f"- Image 2: {img2}",
        ]

        if img3:
            lines.append(f"- Image 3: {img3}")

        lines.extend([
            "",
            "Mode-Specific Guidance:",
            mode_guidance,
        ])

        if additional_instructions.strip():
            lines.extend([
                "",
                f"Additional Instructions: {additional_instructions.strip()}",
            ])

        lines.extend([
            "",
            "Output Requirements:",
            "- Generate a single, clear editing instruction prompt",
            "- Reference images as 'Image 1', 'Image 2', 'Image 3' as needed",
            "- Be specific about what to preserve and what to transfer",
            "- Describe the expected final result",
            "- Do NOT include any explanations, markdown, or preamble",
            "- Start directly with the editing instruction",
            "- Remove any reasoning or planning text; do not include '<think>' or similar tags",
            "",
            "Output only the final editing prompt:",
        ])

        user_prompt = "\n".join(lines)

        logger.debug(f"[QwenImageEdit] Edit mode: {edit_mode}")
        logger.debug(f"[QwenImageEdit] Image 1: {img1[:50]}...")
        logger.debug(f"[QwenImageEdit] Image 2: {img2[:50]}...")
        if img3:
            logger.debug(f"[QwenImageEdit] Image 3: {img3[:50]}...")
        logger.debug(f"[QwenImageEdit] Using model: {ollama_model}")

        ok, response = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={
                "temperature": 0.7,
                "num_predict": 1024,
                "seed": int(seed),
            },
            timeout=120,
        )

        if not ok:
            error_msg = f"Failed to generate prompt: {response}"
            return (error_msg, error_msg)

        return (response, response)

    def _get_mode_guidance(self, edit_mode: str, img1: str, img2: str, img3: str) -> str:
        """Get mode-specific guidance for prompt generation."""

        if edit_mode == "Face Swap (Image 1 → Image 2)":
            return (
                "Transfer the facial features (including face shape, eyes, nose, mouth, skin tone, "
                "and distinctive facial characteristics) from Image 1 onto the subject in Image 2, "
                "while preserving all other elements from Image 2 exactly as they are—including "
                "clothing, body pose, facial expression/emotion, hairstyle (unless it obscures facial "
                "features), lighting, background, and overall composition. Do not alter the pose, "
                "outfit, emotional expression, or scene from Image 2 in any way. The final image "
                "must look like the person from Image 1 naturally inhabiting the exact moment, "
                "attire, and setting shown in Image 2."
            )

        elif edit_mode == "Face Swap (Image 1 → Image 3)":
            if not img3:
                return (
                    "ERROR: Image 3 description is required for this mode. "
                    "Please provide a description for Image 3."
                )
            return (
                "Transfer the facial features (including face shape, eyes, nose, mouth, skin tone, "
                "and distinctive facial characteristics) from Image 1 onto the subject in Image 3, "
                "while preserving all other elements from Image 3 exactly as they are—including "
                "clothing, body pose, facial expression/emotion, hairstyle (unless it obscures facial "
                "features), lighting, background, and overall composition. Image 2 may provide "
                "additional context or reference. Do not alter the pose, outfit, emotional expression, "
                "or scene from Image 3 in any way. The final image must look like the person from "
                "Image 1 naturally inhabiting the exact moment, attire, and setting shown in Image 3."
            )

        elif edit_mode == "Style Transfer (Image 1 style → Image 2)":
            return (
                "Apply the visual style, artistic treatment, color palette, and aesthetic qualities "
                "from Image 1 to the content and subjects in Image 2. Preserve the core subjects, "
                "composition, and identifiable elements from Image 2, but render them in the style "
                "captured in Image 1. This includes lighting mood, texture treatment, color grading, "
                "and any distinctive artistic effects."
            )

        elif edit_mode == "Triple Blend":
            if not img3:
                return (
                    "ERROR: Image 3 description is required for Triple Blend mode. "
                    "Please provide a description for Image 3."
                )
            return (
                "Combine elements from all three images into a cohesive final result. "
                "Use Image 1 as the primary source for facial features or main subject identity, "
                "Image 2 for pose, scene, and composition, and Image 3 for style, lighting, or "
                "additional contextual elements. Blend these elements naturally to create a unified image."
            )

        else:  # Custom mode
            return (
                "Generate a custom editing prompt based on the image descriptions provided. "
                "Analyze what the user likely wants to achieve based on the descriptions and "
                "additional instructions, then create an appropriate editing prompt."
            )


NODE_CLASS_MAPPINGS = {
    "WizdroidQwenImageEdit": QwenImageEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidQwenImageEdit": "Qwen Image Edit",
}
