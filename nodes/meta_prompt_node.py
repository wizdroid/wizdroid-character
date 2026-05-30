import random
from typing import Any, Dict, List, Optional, Tuple

from wizdroid_lib.constants import DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json
from wizdroid_lib.helpers import extract_descriptions, normalize_option_list, with_random, choose
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_text


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
    prompt_styles_payload = load_json("prompt_styles.json")

    visual_payload = _flatten_style_payload(meta_payload.get("visual_styles", []))
    futuristic_payload = meta_payload.get("futuristic_settings", [])
    ancient_payload = meta_payload.get("ancient_eras", [])
    mythology_payload = meta_payload.get("mythological_elements", [])

    # Build prompt style list (exclude video-only models)
    prompt_styles = {k: v for k, v in prompt_styles_payload.items()
                     if not k.startswith(("WAN-", "LTX-", "CogVideo", "Mochi", "Kling-", "PyramidFlow"))}
    style_keys = list(prompt_styles.keys())
    style_labels = [prompt_styles[k]["label"] for k in style_keys]

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
        "prompt_styles": prompt_styles,
        "prompt_style_keys": style_keys,
    }


class WizdroidMetaPromptNode:
    """🧙 Expand loose keywords into detailed image prompts using Ollama LLM."""

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
                "prompt_style": (tuple(options["prompt_style_keys"]), {"default": "SDXL"}),
                "use_ai": ("BOOLEAN", {"default": True}),
                "spiciness": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "detail_level": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "fantasy": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "output_language": (
                    ["English", "Deutsch", "中文", "日本語"],
                    {
                        "default": "English",
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
    CATEGORY = "🧙 Wizdroid/Prompts"

    def generate_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        keywords: str,
        regional_style: str,
        futuristic_setting: str,
        ancient_setting: str,
        mythological_element: str,
        visual_style: str,
        prompt_style: str,
        use_ai: bool,
        spiciness: int = 0,
        detail_level: int = 5,
        fantasy: int = 0,
        noise_seed: int = 0,
        output_language: str = "English",
        max_tokens: int = 1024,
        temperature: float = 0.8,
    ) -> Tuple[str]:
        """Call Ollama to expand user keywords into a detailed prompt."""

        # Use seed to introduce slight variation for batch mode consistency
        rng = random.Random(noise_seed)
        meta_options = _get_meta_options()

        selected_directives = {
            "region": choose(regional_style, meta_options["regions"], rng, noise_seed),
            "futuristic": choose(futuristic_setting, meta_options["futuristic_settings"], rng, noise_seed),
            "ancient": choose(ancient_setting, meta_options["ancient_eras"], rng, noise_seed),
            "mythology": choose(mythological_element, meta_options["mythological_elements"], rng, noise_seed),
            "style": choose(visual_style, meta_options["visual_styles"], rng, noise_seed),
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

        target_language = output_language.strip() or "English"

        # Add language instruction for LLM-native translation (if not English)
        lang_instruction = ""
        if target_language.lower() not in ("english", "en"):
            lang_instruction = f"\n\nCRITICAL: Write the expanded prompt entirely in {target_language}. Do NOT output in English."

        if not use_ai:
            # Template mode: format keywords + directives into a verbose comma-separated prompt
            tmpl_parts = [keyword_payload]
            for part in directive_parts:
                # Extract the actual content after the colon
                if ":" in part:
                    header, content = part.split(":", 1)
                    if "exact phrase verbatim" not in content:
                        tmpl_parts.append(content.strip())
            output = ", ".join(tmpl_parts)
            if output:
                output = output[0].upper() + output[1:]
        else:
            # Spiciness directive
            spice_levels = {
                1: "\nINSTRUCTION - Tone: mildly suggestive, artistic flirtation.",
                2: "\nINSTRUCTION - Tone: playful sensuality, tasteful allure.",
                3: "\nINSTRUCTION - Tone: romantic intimacy, implied sensuality.",
                4: "\nINSTRUCTION - Tone: sensual and warm. Artistic nudity permitted.",
                5: "\nINSTRUCTION - Tone: openly sensual. Erotic themes allowed.",
                6: "\nINSTRUCTION - Tone: boldly erotic. Explicit sensuality.",
                7: "\nINSTRUCTION - Tone: unabashedly erotic. Full adult content.",
                8: "\nINSTRUCTION - Tone: raw and explicit. No restrictions.",
                9: "\nINSTRUCTION - Tone: extremely explicit. Maximum intensity.",
                10: "\nINSTRUCTION - Tone: MAXIMUM SPICINESS. Fully uncensored. No boundaries whatsoever.",
            }
            spice_directive = spice_levels.get(spiciness, "")

            # Detail level directive
            if detail_level <= 2:
                detail_d = f"\nINSTRUCTION - Write at detail level {detail_level}/10: Be extremely concise with minimal descriptors."
            elif detail_level <= 5:
                detail_d = f"\nINSTRUCTION - Write at detail level {detail_level}/10: Use balanced, focused descriptions."
            elif detail_level <= 7:
                detail_d = f"\nINSTRUCTION - Write at detail level {detail_level}/10: Use rich, concrete visual details and sensory language."
            else:
                detail_d = f"\nINSTRUCTION - Write at detail level {detail_level}/10: Use lavishly detailed, multi-layered descriptions."
            # Fantasy level directive
            if fantasy == 0:
                fantasy_d = "\nINSTRUCTION - Realism: Strictly realistic. No magical/sci-fi/fantasy elements."
            elif fantasy <= 3:
                fantasy_d = f"\nINSTRUCTION - Fantasy level {fantasy}/10: Subtle fantastical touches allowed but keep grounded."
            elif fantasy <= 7:
                fantasy_d = f"\nINSTRUCTION - Fantasy level {fantasy}/10: Include significant fantasy/sci-fi elements."
            else:
                fantasy_d = f"\nINSTRUCTION - Fantasy level {fantasy}/10: Fully fantastical. No reality constraints."

            # Style-specific format guidance
            meta_style_options = meta_options.get("prompt_styles", {})
            style_meta = meta_style_options.get(prompt_style, meta_style_options.get("SDXL", {}))
            style_guidance = style_meta.get("guidance", "Natural flowing sentence or comma-separated phrases")
            style_token_limit = min(int(max_tokens), style_meta.get("token_limit", 512))
            style_directive = f"\nFormat: {style_guidance}. Keep under {style_token_limit} tokens."

            system_prompt = load_system_prompt_text("system_prompts/meta_prompt_system.txt")
            ok, output = generate_text(
                ollama_url=ollama_url,
                model=ollama_model,
                prompt=prompt_input + lang_instruction + spice_directive + detail_d + fantasy_d + style_directive,
                system=system_prompt,
                options={
                    "temperature": float(adjusted_temperature),
                    "num_predict": int(max_tokens),
                    "seed": int(noise_seed),
                },
                timeout=120,
            )
            if not ok:
                return (f"MetaPrompt error: {output}",)

            if not output:
                output = "MetaPrompt error: empty response from Ollama"

        # Content safety guardrail (skip when spiciness > 0)
        if spiciness == 0:
            err = enforce_sfw(output)
            if err:
                return (
                    "MetaPrompt blocked: potential NSFW content detected. "
                    "Revise keywords.",
                )

        return (output,)


META_PROMPT_NODE_CLASS_MAPPINGS = {
    "WizdroidMetaPrompt": WizdroidMetaPromptNode,
}

META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidMetaPrompt": "🧙 Wizdroid: Meta Prompt",
}
