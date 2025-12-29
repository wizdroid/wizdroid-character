import random
from typing import Any, Dict, List, Optional, Tuple
import logging

from lib.content_safety import CONTENT_RATING_CHOICES, enforce_sfw
from lib.data_files import load_json
from lib.ollama_client import DEFAULT_OLLAMA_URL, collect_models, generate_text
from lib.system_prompts import load_system_prompt_text


def _normalize_token_limit(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


class PromptCombinerNode:
    """
    ComfyUI node to combine multiple text prompts into one coherent prompt
    using different model styles from prompt_styles.json.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("combined_prompt", "preview")
    FUNCTION = "combine_prompts"

    @classmethod
    def INPUT_TYPES(cls):
        prompt_styles = load_json("prompt_styles.json")
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        style_options = [style_key for style_key in prompt_styles.keys()]

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW only"}),
                "prompt_style": (style_options, {"default": "SDXL"}),
                "input_prompt_1": ("STRING", {"multiline": True, "default": ""}),
                "input_prompt_2": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "input_prompt_3": ("STRING", {"multiline": True, "default": ""}),
                "input_prompt_4": ("STRING", {"multiline": True, "default": ""}),
                "input_prompt_5": ("STRING", {"multiline": True, "default": ""}),
                "custom_instructions": ("STRING", {"multiline": True, "default": ""}),
                "token_limit_override": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    def combine_prompts(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        prompt_style: str,
        input_prompt_1: str,
        input_prompt_2: str,
        input_prompt_3: str = "",
        input_prompt_4: str = "",
        input_prompt_5: str = "",
        custom_instructions: str = "",
        token_limit_override: str = "",
    ) -> Tuple[str]:
        prompt_styles = load_json("prompt_styles.json")

        # Collect all non-empty input prompts
        input_prompts = [
            prompt.strip() for prompt in [input_prompt_1, input_prompt_2, input_prompt_3, input_prompt_4, input_prompt_5]
            if prompt.strip()
        ]

        if len(input_prompts) < 2:
            return ("[ERROR: Need at least 2 input prompts to combine]",)

        # Get style configuration
        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        token_limit = style_config["token_limit"]
        override_limit = _normalize_token_limit(token_limit_override)
        if override_limit:
            token_limit = override_limit

        # Build the prompt instruction for the LLM
        system_prompt = load_system_prompt_text("system_prompts/prompt_combiner_system.txt", content_rating)

        # Build the user prompt
        lines = [
            f"Combine these {len(input_prompts)} prompts into one coherent {prompt_style} prompt:",
        ]

        for i, prompt in enumerate(input_prompts, 1):
            lines.append(f"Prompt {i}: {prompt}")

        if custom_instructions.strip():
            lines.append(f"\nAdditional combination instructions: {custom_instructions.strip()}")

        lines.extend([
            f"\nFormat: {style_guidance}",
            "CRITICAL - Create one unified prompt that:",
            "  * Merges all key elements from input prompts",
            "  * Eliminates redundancy and contradictions",
            "  * Maintains artistic coherence",
            "  * Preserves important details from each prompt",
            "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', the model/style name (Flux, SDXL, Qwen, HiDream, etc.), or similar",
            "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
            "Remove any reasoning or planning text; do not include '<think>' or similar tags",
            "Output only the final combined prompt text:"
        ])

        user_prompt = "\n".join(lines)

        logger = logging.getLogger(__name__)
        logger.debug(f"[PromptCombiner] Combining {len(input_prompts)} prompts")
        logger.debug(f"[PromptCombiner] Prompt style: {style_label}")
        logger.debug(f"[PromptCombiner] Using model: {ollama_model}")
        for i, prompt in enumerate(input_prompts, 1):
            logger.debug(f"[PromptCombiner] Input {i}: {prompt[:50]}...")

        ok, response = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={
                "temperature": 0.7,
                "num_predict": int(token_limit) if isinstance(token_limit, int) else 512,
            },
            timeout=120,
        )

        if not ok:
            error_msg = f"Failed to combine prompts: {response}"
            return (error_msg, error_msg)

        if content_rating != "NSFW allowed":
            err = enforce_sfw(response)
            if err:
                blocked = (
                    "PromptCombiner blocked: potential NSFW content detected. "
                    "Switch content_rating to 'NSFW allowed' or revise inputs."
                )
                return (blocked, blocked)

        return (response, response)


NODE_CLASS_MAPPINGS = {
    "WizdroidPromptCombiner": PromptCombinerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidPromptCombiner": "Prompt Combiner",
}