"""🎨 Style Transfer Node - Select comic/anime art styles and output their prompt descriptions."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from wizdroid_lib.constants import DEFAULT_OLLAMA_URL
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.registry import DataRegistry
from wizdroid_lib.system_prompts import load_system_prompt_text

logger = logging.getLogger(__name__)


class WizdroidStyleTransferNode:
    """🎨 Select an art style and output its full prompt description.

    Provides a curated collection of 15+ comic, manga, and anime art styles
    (American Comic Book, Manga, Studio Ghibli, Cel-Shaded Anime, etc.).
    Each style includes a detailed prompt description suitable for image generation.

    When ai_enhanced is enabled, uses Ollama to transform the style description
    into a richer, more detailed generation prompt.
    """

    CATEGORY = "🧙 Wizdroid/Prompts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "style_name")
    FUNCTION = "get_style"

    @classmethod
    def INPUT_TYPES(cls):
        styles = DataRegistry.get_style_transfer_styles()
        style_names = [s["name"] for s in styles]
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                "style": (["Random", *style_names], {"default": "Random"}),
                "ai_enhanced": ("BOOLEAN", {"default": False}),
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 512, "min": 50, "max": 2048, "step": 10}),
                "user_subject": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe your subject/scene here... e.g. a samurai at sunset, a cyberpunk city street"}),
            }
        }

    def get_style(
        self,
        style: str,
        ai_enhanced: bool = False,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        user_subject: str = "",
    ) -> Tuple[str, str]:
        """Return the prompt description and name for the selected style.

        When ai_enhanced is True, uses Ollama to enrich the style description
        into a more detailed generation prompt.
        """
        styles = DataRegistry.get_style_transfer_styles()

        if style == "Random":
            chosen = random.choice(styles)
        else:
            chosen = None
            for s in styles:
                if s["name"] == style:
                    chosen = s
                    break

        if chosen is None:
            logger.warning("Style '%s' not found, falling back to first available style", style)
            chosen = styles[0] if styles else {"style_description": "", "name": "None"}

        style_name = chosen["name"]
        style_description = chosen["style_description"]
        subject = user_subject.strip()

        if not ai_enhanced:
            # Static mode: pure style transfer prompt
            # Always start with "Transform the image" to signal style transfer only
            if subject:
                final_prompt = f"Transform the image to {style_name} style, {subject}, {style_description}"
            else:
                final_prompt = f"Transform the image to {style_name} style, {style_description}"
            return (final_prompt, style_name)

        # --- AI Enhanced mode: invoke Ollama ---
        system_prompt = load_system_prompt_text("system_prompts/style_transfer_system.txt")

        user_prompt = (
            f"Target Art Style: {style_name}\n"
            f"Style Description: {style_description}\n"
        )
        if subject:
            user_prompt += f"Subject to preserve: {subject}\n"

        user_prompt += (
            "\nGenerate a style-transfer prompt that starts with 'Transform the image'.\n"
            "Focus exclusively on applying the visual style (linework, coloring, shading, texture, palette).\n"
            "Do NOT add new characters, speech bubbles, props, or scene elements.\n"
            "Only change how the existing image is rendered artistically."
        )

        ok, response_text = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            timeout=120,
        )

        if ok and response_text.strip():
            enhanced_prompt = response_text.strip()
            # Clean up any markdown code blocks that might leak through
            enhanced_prompt = enhanced_prompt.replace("```", "").strip()
            return (enhanced_prompt, style_name)

        # Fallback: if Ollama fails, return a static style-transfer prompt
        logger.warning(
            "Ollama enhancement failed for style '%s', falling back to static description",
            style_name,
        )
        if subject:
            fallback = f"Transform the image to {style_name} style, {subject}, {style_description}"
        else:
            fallback = f"Transform the image to {style_name} style, {style_description}"
        return (fallback, style_name)


NODE_CLASS_MAPPINGS = {
    "WizdroidStyleTransferNode": WizdroidStyleTransferNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidStyleTransferNode": "🧙 Wizdroid: Style Transfer",
}
