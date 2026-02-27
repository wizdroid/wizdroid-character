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


class WizdroidImageEditNode:
    """🧙 Generate prompts for multi-image AI editing models using Ollama LLM."""

    CATEGORY = "🧙 Wizdroid/Prompts"
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
            "Generate an ultra-detailed, high-fashion-quality image editing prompt for a multi-image AI model.",
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
            "- Generate a RICHLY DETAILED, multi-paragraph prompt (150–300 words)",
            "- Reference images as 'Image 1', 'Image 2', 'Image 3' as needed",
            "- ALWAYS name concrete elements (e.g. 'the woman with auburn hair from Image 1', "
            "  'the red silk dress from Image 2') — NEVER say 'the subject of Image N'",
            "- For style transfers, DESCRIBE the style explicitly (palette, texture, genre)",
            "- Cover ALL layers: identity, clothing, pose, beauty/makeup, photography quality",
            "- ALWAYS include professional makeup/beauty enhancements for human subjects",
            "- ALWAYS close with photography quality descriptors (studio lighting, 8K, "
            "  cinematic color grading, masterpiece, best quality, photorealistic, no artifacts)",
            "- Be lavishly specific about what to preserve and what to transfer",
            "- Do NOT include any explanations, markdown, headers, or preamble",
            "- Start directly with the descriptive editing instruction",
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
                "temperature": 0.75,
                "num_predict": 2048,
                "seed": int(seed),
            },
            timeout=180,
        )

        if not ok:
            error_msg = f"Failed to generate prompt: {response}"
            return (error_msg, error_msg)

        return (response, response)

    def _get_mode_guidance(self, edit_mode: str, img1: str, img2: str, img3: str) -> str:
        """Get mode-specific guidance for prompt generation."""

        if edit_mode == "Face Swap (Image 1 → Image 2)":
            return (
                f"Generate an ultra-detailed, high-fashion-grade face swap prompt (150–300 words, "
                f"multiple paragraphs).\n\n"
                f"SOURCE IDENTITY (Image 1): {img1}\n"
                f"TARGET SCENE  (Image 2): {img2}\n\n"
                f"PARAGRAPH 1 — IDENTITY & OUTFIT:\n"
                f"Describe a stunning portrait of the EXACT same person from Image 1, using their "
                f"precise facial features, skin tone, eye color/shape, hair color/style/texture with "
                f"absolutely zero changes to identity. Then describe the exact outfit, fabric details, "
                f"fit, accessories, and jewelry from Image 2 that the person must wear.\n\n"
                f"PARAGRAPH 2 — POSE & COMPOSITION:\n"
                f"Preserve the exact body pose, camera angle, framing, and spatial arrangement from "
                f"Image 2. Describe the background/setting from Image 2 in detail.\n\n"
                f"PARAGRAPH 3 — BEAUTY & MAKEUP:\n"
                f"Add lavish professional studio makeup: flawless skin finish with subtle glow, "
                f"contouring, highlighting, elegant eyeshadow, voluminous lashes, defined brows, "
                f"natural blush, bold yet elegant lips. Describe expression and eye catchlights.\n\n"
                f"PARAGRAPH 4 — PHOTOGRAPHY QUALITY:\n"
                f"Close with: professional beauty photography, soft diffused studio lighting, gentle "
                f"rim light, high-end fashion magazine style, ultra-sharp details, 8K resolution, "
                f"cinematic color grading, masterpiece, best quality, photorealistic, no deformations, "
                f"no artifacts."
            )

        elif edit_mode == "Face Swap (Image 1 → Image 3)":
            if not img3:
                return (
                    "ERROR: Image 3 description is required for this mode. "
                    "Please provide a description for Image 3."
                )
            return (
                f"Generate an ultra-detailed, high-fashion-grade face swap prompt (150–300 words, "
                f"multiple paragraphs).\n\n"
                f"SOURCE IDENTITY (Image 1): {img1}\n"
                f"OUTFIT / REF   (Image 2): {img2}\n"
                f"TARGET SCENE   (Image 3): {img3}\n\n"
                f"PARAGRAPH 1 — IDENTITY & OUTFIT:\n"
                f"Describe a stunning portrait of the EXACT same person from Image 1, using their "
                f"precise facial features, skin tone, eye color/shape, hair with zero identity changes. "
                f"If Image 2 provides clothing or accessory reference, incorporate those exact garment "
                f"details (fabric, fit, color, accessories). Otherwise use Image 3's outfit.\n\n"
                f"PARAGRAPH 2 — POSE & COMPOSITION:\n"
                f"Preserve the exact body pose, camera angle, framing, and spatial arrangement from "
                f"Image 3. Describe Image 3's background/setting in detail.\n\n"
                f"PARAGRAPH 3 — BEAUTY & MAKEUP:\n"
                f"Add lavish professional studio makeup: flawless skin with subtle glow, contouring, "
                f"highlighting, elegant eyeshadow, voluminous lashes, defined brows, natural blush, "
                f"bold yet elegant lips. Describe expression and eye catchlights.\n\n"
                f"PARAGRAPH 4 — PHOTOGRAPHY QUALITY:\n"
                f"Close with: professional beauty photography, soft diffused studio lighting, gentle "
                f"rim light, high-end fashion magazine style, ultra-sharp details, 8K resolution, "
                f"cinematic color grading, masterpiece, best quality, photorealistic, no deformations, "
                f"no artifacts."
            )

        elif edit_mode == "Style Transfer (Image 1 style → Image 2)":
            return (
                f"Generate an ultra-detailed style transfer prompt (150–300 words, multiple "
                f"paragraphs).\n\n"
                f"STYLE SOURCE  (Image 1): {img1}\n"
                f"CONTENT SOURCE(Image 2): {img2}\n\n"
                f"PARAGRAPH 1 — STYLE DESCRIPTION:\n"
                f"Explicitly describe Image 1's visual style in rich detail: color palette, texture "
                f"quality, brush stroke character, lighting mood, tonal range, artistic genre, and "
                f"any distinctive aesthetic signatures. NEVER just say 'the style of Image 1'.\n\n"
                f"PARAGRAPH 2 — CONTENT PRESERVATION:\n"
                f"Describe the visual content from Image 2 that must be preserved intact: core "
                f"subjects, their identity/features, composition, spatial arrangement, and key "
                f"details. State that these must remain fully recognizable.\n\n"
                f"PARAGRAPH 3 — INTEGRATION & BEAUTY:\n"
                f"Describe how the style characteristics wrap around the preserved content — how "
                f"lighting, color grading, and texture treatment transform the scene. If human "
                f"subjects are present, add flattering beauty enhancements appropriate to the "
                f"style.\n\n"
                f"PARAGRAPH 4 — PHOTOGRAPHY QUALITY:\n"
                f"Close with quality descriptors appropriate to the style: ultra-sharp details, "
                f"high resolution, masterpiece, best quality, no deformations, no artifacts."
            )

        elif edit_mode == "Triple Blend":
            if not img3:
                return (
                    "ERROR: Image 3 description is required for Triple Blend mode. "
                    "Please provide a description for Image 3."
                )
            return (
                f"Generate an ultra-detailed triple-blend prompt (150–300 words, multiple "
                f"paragraphs).\n\n"
                f"IDENTITY SOURCE (Image 1): {img1}\n"
                f"OUTFIT / POSE  (Image 2): {img2}\n"
                f"STYLE / SCENE  (Image 3): {img3}\n\n"
                f"PARAGRAPH 1 — IDENTITY:\n"
                f"Describe a stunning portrait of the EXACT same person from Image 1, using their "
                f"precise facial features, skin tone, eye color/shape, hair color/style/texture "
                f"with absolutely zero changes to identity.\n\n"
                f"PARAGRAPH 2 — OUTFIT & POSE:\n"
                f"Describe the exact outfit from Image 2 (fabric details, fit, accessories, jewelry) "
                f"and the exact pose, camera angle, framing from Image 2.\n\n"
                f"PARAGRAPH 3 — STYLE & SCENE:\n"
                f"If Image 3 provides a style reference, describe those style characteristics in "
                f"detail (palette, texture, lighting, mood). If Image 3 provides a scene/background, "
                f"describe it vividly.\n\n"
                f"PARAGRAPH 4 — BEAUTY & MAKEUP:\n"
                f"Add lavish professional studio makeup: flawless skin with subtle glow, contouring, "
                f"highlighting, elegant eyeshadow, voluminous lashes, defined brows, natural blush, "
                f"bold yet elegant lips. Describe expression and eye catchlights.\n\n"
                f"PARAGRAPH 5 — PHOTOGRAPHY QUALITY:\n"
                f"Close with: professional beauty photography, soft diffused studio lighting, gentle "
                f"rim light, high-end fashion magazine style, ultra-sharp details, 8K resolution, "
                f"cinematic color grading, masterpiece, best quality, photorealistic, no deformations, "
                f"no artifacts."
            )

        else:  # Custom mode
            return (
                f"Generate an ultra-detailed, high-fashion-grade custom editing prompt (150–300 "
                f"words, multiple paragraphs).\n\n"
                f"Image 1: {img1}\n"
                f"Image 2: {img2}\n"
                f"{'Image 3: ' + img3 + chr(10) if img3 else ''}"
                f"\nAnalyze the image descriptions and additional instructions to determine the "
                f"user's intent, then craft a richly detailed prompt that covers:\n"
                f"- IDENTITY: precise facial features, skin tone, distinguishing characteristics\n"
                f"- CLOTHING: exact garment details, fabric, fit, accessories\n"
                f"- POSE: body position, camera angle, framing, composition\n"
                f"- BEAUTY: professional makeup, expression, catchlights\n"
                f"- QUALITY: professional photography, studio lighting, 8K resolution, cinematic "
                f"color grading, masterpiece, best quality, photorealistic, no artifacts\n\n"
                f"ALWAYS name concrete elements from each image rather than generic 'the subject'. "
                f"For style transfers, describe style characteristics explicitly."
            )


NODE_CLASS_MAPPINGS = {
    "WizdroidImageEdit": WizdroidImageEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidImageEdit": "🧙 Wizdroid: Image Edit",
}
