import random
from typing import Any, Dict, List, Optional, Tuple

from lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from lib.content_safety import enforce_sfw
from lib.data_files import load_json
from lib.helpers import extract_descriptions, normalize_option_list, with_random, choose
from lib.ollama_client import collect_models, generate_text
from lib.system_prompts import load_system_prompt_text


def _flatten_style_payload(payload: Any) -> List[Any]:
    """Allow visual_styles to be grouped by category while keeping downstream logic list-based."""

    if isinstance(payload, dict):
        merged: List[Any] = []
        for group in payload.values():
            if isinstance(group, list):
                merged.extend(group)
        return merged
    if isinstance(payload, list):
        return payload
    return []


def _should_echo_style_label(style_name: str) -> bool:
    lowered = style_name.lower()
    banned_tokens = (
        "studio",
        "disney",
        "pixar",
        "ghibli",
        "cartoon network",
    )
    return not any(token in lowered for token in banned_tokens)


def _get_meta_options() -> Dict[str, Any]:
    region_payload = load_json("regions.json")
    meta_payload = load_json("meta_prompt_options.json")

    visual_payload = _flatten_style_payload(meta_payload.get("visual_styles", []))
    futuristic_payload = meta_payload.get("futuristic_settings", [])
    ancient_payload = meta_payload.get("ancient_eras", [])
    mythology_payload = meta_payload.get("mythological_elements", [])

    return {
        "regions": list(region_payload.get("regions", [])),
        "futuristic_settings": list(futuristic_payload),
        "ancient_eras": list(ancient_payload),
        "mythological_elements": list(mythology_payload),
        "visual_styles": normalize_option_list(visual_payload),
        "visual_style_desc_map": extract_descriptions(visual_payload),
        "futuristic_desc_map": extract_descriptions(futuristic_payload),
        "ancient_desc_map": extract_descriptions(ancient_payload),
        "mythology_desc_map": extract_descriptions(mythology_payload),
    }


class MetaPromptGeneratorNode:
    """Generate a detailed image prompt from loose keywords using Ollama.

    This node takes a short text input (keywords or fragments) and asks an
    Ollama-hosted LLM to expand it into a 150–200 token imaginative prompt.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:  # noqa: N802
        models = collect_models(DEFAULT_OLLAMA_URL)
        options = _get_meta_options()

        return {
            "required": {
                "ollama_url": (
                    "STRING",
                    {
                        "default": DEFAULT_OLLAMA_URL,
                        "multiline": False,
                    },
                ),
                "ollama_model": (
                    models,
                    {
                        "default": models[0] if models else "model_not_available",
                    },
                ),
                "keywords": (
                    "STRING",
                    {
                        "default": "female knight, rainy neon city, blue coat",
                        "multiline": True,
                    },
                ),
                "content_rating": (
                    CONTENT_RATING_CHOICES,
                    {
                        "default": "SFW",
                    },
                ),
                "regional_style": (
                    with_random(options["regions"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "futuristic_setting": (
                    with_random(options["futuristic_settings"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "ancient_setting": (
                    with_random(options["ancient_eras"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "mythological_element": (
                    with_random(options["mythological_elements"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "visual_style": (
                    with_random(options["visual_styles"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
            "optional": {
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.05,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Wizdroid/character"

    def generate_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        keywords: str,
        content_rating: str,
        regional_style: str,
        futuristic_setting: str,
        ancient_setting: str,
        mythological_element: str,
        visual_style: str,
        seed: int,
        max_tokens: int = 1024,
        temperature: float = 0.8,
    ) -> Tuple[str]:
        """Call Ollama to expand user keywords into a detailed prompt."""

        # Use seed to introduce slight variation for batch mode consistency
        rng = random.Random(seed)
        meta_options = _get_meta_options()

        selected_directives = {
            "region": choose(regional_style, meta_options["regions"], rng),
            "futuristic": choose(futuristic_setting, meta_options["futuristic_settings"], rng),
            "ancient": choose(ancient_setting, meta_options["ancient_eras"], rng),
            "mythology": choose(mythological_element, meta_options["mythological_elements"], rng),
            "style": choose(visual_style, meta_options["visual_styles"], rng),
        }

        directive_parts: List[str] = []
        if selected_directives["region"]:
            directive_parts.append(f"Regional inspiration: {selected_directives['region']}")
        if selected_directives["futuristic"]:
            directive_parts.append(f"Futuristic setting: {selected_directives['futuristic']}")
            futur_desc = (meta_options.get("futuristic_desc_map") or {}).get(selected_directives["futuristic"], "")
            if futur_desc:
                directive_parts.append(f"Futuristic detail: {futur_desc}")
        if selected_directives["ancient"]:
            directive_parts.append(f"Historical inspiration: {selected_directives['ancient']}")
            ancient_desc = (meta_options.get("ancient_desc_map") or {}).get(selected_directives["ancient"], "")
            if ancient_desc:
                directive_parts.append(f"Historical detail: {ancient_desc}")
        if selected_directives["mythology"]:
            directive_parts.append(f"Mythological element: {selected_directives['mythology']}")
            myth_desc = (meta_options.get("mythology_desc_map") or {}).get(selected_directives["mythology"], "")
            if myth_desc:
                directive_parts.append(f"Mythology detail: {myth_desc}")
        if selected_directives["style"]:
            style_name = selected_directives["style"]
            style_desc = (meta_options.get("visual_style_desc_map") or {}).get(style_name, "")
            if _should_echo_style_label(style_name):
                directive_parts.append(
                    "Visual style (HARD REQUIREMENT): "
                    f"{style_name} (include this exact phrase verbatim in the final prompt)"
                )
            else:
                directive_parts.append(f"Visual style: {style_name}")
                directive_parts.append(
                    "Brand safety: Do NOT mention brand/franchise names; express the look using generic descriptors."
                )
            if style_desc:
                directive_parts.append(f"Visual style meaning: {style_desc}")
            directive_parts.append(
                "Visual style enforcement: Make the style unmistakable via explicit medium + rendering cues "
                "(linework/inking, shading, palette, texture, and composition)."
            )

        temp_variation = rng.uniform(-0.1, 0.1)  # Small temperature variation
        adjusted_temperature = max(0.0, min(1.5, temperature + temp_variation))

        keyword_payload = keywords.strip() or "dreamlike character vignette"
        if directive_parts:
            context_block = "Context directives:\n" + "\n".join(f"- {part}" for part in directive_parts)
            prompt_input = f"{keyword_payload}\n{context_block}"
        else:
            prompt_input = keyword_payload

        system_prompt = load_system_prompt_text("system_prompts/meta_prompt_system.txt", content_rating)
        ok, output = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=prompt_input,
            system=system_prompt,
            options={
                "temperature": float(adjusted_temperature),
                "num_predict": int(max_tokens),
                "seed": int(seed),
            },
            timeout=120,
        )
        if not ok:
            return (f"MetaPrompt error: {output}",)

        if not output:
            output = "MetaPrompt error: empty response from Ollama"

        # Guardrail: in strict SFW mode, block outputs that look NSFW.
        if content_rating == "SFW":
            err = enforce_sfw(output)
            if err:
                return (
                    "MetaPrompt blocked: potential NSFW content detected. "
                    "Switch content_rating to 'Mixed' or 'NSFW' or revise keywords.",
                )

        return (output,)


META_PROMPT_NODE_CLASS_MAPPINGS = {
    "MetaPromptGenerator": MetaPromptGeneratorNode,
}

META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaPromptGenerator": "Meta Prompt Generator",
}
