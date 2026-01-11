"""🧙 Wizdroid Character Prompt Node - Generate detailed character prompts using Ollama LLM."""

import hashlib
import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.registry import DataRegistry
from wizdroid_lib.helpers import (
    choose,
    choose_for_rating,
    extract_descriptions,
    get_background_groups,
    split_groups,
    with_random,
)
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

logger = logging.getLogger(__name__)

# Prompt cache with LRU-style eviction
_PROMPT_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 100


def _cache_key(selections: Dict, prompt_style: str, retain_face: bool, custom_text: str, temp: float) -> str:
    """Generate cache key for prompt caching."""
    data = json.dumps(selections, sort_keys=True) + prompt_style + str(retain_face) + custom_text + str(temp)
    return hashlib.md5(data.encode()).hexdigest()


class WizdroidCharacterPromptNode:
    """🧙 Generate detailed character prompts for AI image generation using Ollama LLM."""

    CATEGORY = "🧙 Wizdroid/Prompts"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "preview")
    FUNCTION = "build_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        # Use centralized registry for cached data access
        opt = DataRegistry.get_character_options()
        styles = DataRegistry.get_prompt_styles()
        regions = DataRegistry.get_regions()
        countries = DataRegistry.get_countries()
        cultures = DataRegistry.get_cultures()

        # Process SFW/NSFW splits
        pose_sfw, pose_nsfw = split_groups(opt.get("pose_style"))
        image_sfw, image_nsfw = split_groups(opt.get("image_category"))
        bg = get_background_groups(opt.get("background_style"))

        # Get Ollama models (with TTL caching)
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                # LLM Settings
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "prompt_style": (tuple(styles.keys()), {"default": "SDXL"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 50, "max": 2048, "step": 10}),
                # Character Identity
                "character_name": ("STRING", {"default": ""}),
                "retain_face": ("BOOLEAN", {"default": False}),
                "gender": (with_random(opt.get("gender", [])), {"default": RANDOM_LABEL}),
                "race": (with_random(opt.get("race", [])), {"default": NONE_LABEL}),
                "age_group": (with_random(opt.get("age_group", [])), {"default": RANDOM_LABEL}),
                "body_type": (with_random(opt.get("body_type", [])), {"default": NONE_LABEL}),
                # Appearance
                "hair_color": (with_random(opt.get("hair_color", [])), {"default": NONE_LABEL}),
                "hair_style": (with_random(opt.get("hair_style", [])), {"default": NONE_LABEL}),
                "eye_color": (with_random(opt.get("eye_color", [])), {"default": NONE_LABEL}),
                "makeup_style": (with_random(opt.get("makeup_style", [])), {"default": NONE_LABEL}),
                # Expression & Pose
                "facial_expression": (with_random(opt.get("facial_expression", [])), {"default": RANDOM_LABEL}),
                "face_angle": (with_random(opt.get("face_angle", [])), {"default": NONE_LABEL}),
                "pose_style": (with_random(pose_sfw + pose_nsfw), {"default": RANDOM_LABEL}),
                # Fashion
                "fashion_outfit": (with_random(opt.get("fashion_outfit", [])), {"default": RANDOM_LABEL}),
                "fashion_style": (with_random(opt.get("fashion_style", [])), {"default": RANDOM_LABEL}),
                "footwear_style": (with_random(opt.get("footwear_style", [])), {"default": NONE_LABEL}),
                "upcycled_fashion": (with_random(opt.get("upcycled_materials", [])), {"default": NONE_LABEL}),
                # Scene & Camera
                "image_category": (with_random(image_sfw + image_nsfw), {"default": RANDOM_LABEL}),
                "background_stage": (with_random(bg.get("studio_controlled", [])), {"default": NONE_LABEL}),
                "background_location": (with_random(bg.get("public_exotic_real", [])), {"default": NONE_LABEL}),
                "background_imaginative": (with_random(bg.get("imaginative_surreal", [])), {"default": NONE_LABEL}),
                "lighting_style": (with_random(opt.get("lighting_style", [])), {"default": RANDOM_LABEL}),
                "camera_angle": (with_random(opt.get("camera_angle", [])), {"default": NONE_LABEL}),
                "camera_lens": (with_random(opt.get("camera_lens", [])), {"default": NONE_LABEL}),
                "color_palette": (with_random(opt.get("color_palette", [])), {"default": NONE_LABEL}),
                # Cultural Context
                "region": (with_random(regions.get("regions", [])), {"default": NONE_LABEL}),
                "country": (with_random(countries.get("countries", [])), {"default": NONE_LABEL}),
                "culture": (with_random(cultures.get("cultures", [])), {"default": NONE_LABEL}),
                # Custom Text
                "custom_text_llm": ("STRING", {"multiline": True, "default": ""}),
                "custom_text_append": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    def build_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        prompt_style: str,
        temperature: float,
        max_tokens: int,
        character_name: str,
        retain_face: bool,
        gender: str,
        race: str,
        age_group: str,
        body_type: str,
        hair_color: str,
        hair_style: str,
        eye_color: str,
        makeup_style: str,
        facial_expression: str,
        face_angle: str,
        pose_style: str,
        fashion_outfit: str,
        fashion_style: str,
        footwear_style: str,
        upcycled_fashion: str,
        image_category: str,
        background_stage: str,
        background_location: str,
        background_imaginative: str,
        lighting_style: str,
        camera_angle: str,
        camera_lens: str,
        color_palette: str,
        region: str,
        country: str,
        culture: str,
        custom_text_llm: str,
        custom_text_append: str,
        seed: int = 0,
    ) -> Tuple[str, str, str]:
        # Load data from registry
        opt = DataRegistry.get_character_options()
        styles = DataRegistry.get_prompt_styles()
        regions_data = DataRegistry.get_regions()
        countries_data = DataRegistry.get_countries()
        cultures_data = DataRegistry.get_cultures()

        pose_sfw, pose_nsfw = split_groups(opt.get("pose_style"))
        image_sfw, image_nsfw = split_groups(opt.get("image_category"))
        bg = get_background_groups(opt.get("background_style"))

        rng = random.Random(seed)

        # Resolve all selections with helper
        def resolve(val: str, opts: List[Any]) -> Optional[str]:
            return choose(val, opts, rng, seed)

        def resolve_rated(val: str, sfw: List[str], nsfw: List[str]) -> Optional[str]:
            return choose_for_rating(val, sfw, nsfw, content_rating, rng, seed)

        resolved = {
            "character_name": character_name.strip() or None,
            "image_category": resolve_rated(image_category, image_sfw, image_nsfw),
            "gender": resolve(gender, opt.get("gender", [])),
            "race": resolve(race, opt.get("race", [])),
            "age_group": resolve(age_group, opt.get("age_group", [])),
            "body_type": resolve(body_type, opt.get("body_type", [])),
            "hair_color": resolve(hair_color, opt.get("hair_color", [])),
            "hair_style": resolve(hair_style, opt.get("hair_style", [])),
            "eye_color": resolve(eye_color, opt.get("eye_color", [])),
            "facial_expression": resolve(facial_expression, opt.get("facial_expression", [])),
            "face_angle": resolve(face_angle, opt.get("face_angle", [])),
            "camera_angle": resolve(camera_angle, opt.get("camera_angle", [])),
            "pose_style": resolve_rated(pose_style, pose_sfw, pose_nsfw),
            "makeup_style": resolve(makeup_style, opt.get("makeup_style", [])),
            "fashion_outfit": resolve(fashion_outfit, opt.get("fashion_outfit", [])),
            "fashion_style": resolve(fashion_style, opt.get("fashion_style", [])),
            "footwear_style": resolve(footwear_style, opt.get("footwear_style", [])),
            "upcycled_fashion": resolve(upcycled_fashion, opt.get("upcycled_materials", [])),
            "background_style": (
                resolve(background_stage, bg.get("studio_controlled", []))
                or resolve(background_location, bg.get("public_exotic_real", []))
                or resolve(background_imaginative, bg.get("imaginative_surreal", []))
            ),
            "lighting_style": resolve(lighting_style, opt.get("lighting_style", [])),
            "camera_lens": resolve(camera_lens, opt.get("camera_lens", [])),
            "color_palette": resolve(color_palette, opt.get("color_palette", [])),
            "region": resolve(region, regions_data.get("regions", [])),
            "country": resolve(country, countries_data.get("countries", [])),
            "culture": resolve(culture, cultures_data.get("cultures", [])),
        }

        # Validate content rating vs selections
        if content_rating == "SFW":
            if resolved.get("image_category") in set(image_nsfw):
                return ("[ERROR: SFW rating but NSFW image_category selected]", "", "")
            if resolved.get("pose_style") in set(pose_nsfw):
                return ("[ERROR: SFW rating but NSFW pose_style selected]", "", "")

        # Smart country selection based on region
        if resolved.get("region") and country == RANDOM_LABEL:
            resolved["country"] = f"authentic to {resolved['region']}"

        # Get style metadata
        style_meta = styles.get(prompt_style, {})
        custom_text = custom_text_llm.strip()

        # Check cache
        cache_key = _cache_key(resolved, prompt_style, retain_face, custom_text, temperature)
        if cache_key in _PROMPT_CACHE:
            llm_response = _PROMPT_CACHE[cache_key]
        else:
            # Extract descriptions for enhanced prompting
            desc_maps = {
                "image_category": extract_descriptions(opt.get("image_category")),
                "fashion_style": extract_descriptions(opt.get("fashion_style")),
                "fashion_outfit": extract_descriptions(opt.get("fashion_outfit")),
                "makeup_style": extract_descriptions(opt.get("makeup_style")),
                "lighting_style": extract_descriptions(opt.get("lighting_style")),
                "camera_lens": extract_descriptions(opt.get("camera_lens")),
                "color_palette": extract_descriptions(opt.get("color_palette")),
            }

            llm_response = self._invoke_llm(
                ollama_url, ollama_model, content_rating, prompt_style, retain_face,
                style_meta, resolved, desc_maps, custom_text, temperature, max_tokens, seed
            )

            # Cache with LRU eviction
            if len(_PROMPT_CACHE) >= _MAX_CACHE_SIZE:
                _PROMPT_CACHE.pop(next(iter(_PROMPT_CACHE)))
            _PROMPT_CACHE[cache_key] = llm_response

        negative_prompt = style_meta.get("negative_prompt", "")
        final_prompt = (llm_response or "").strip()

        # Prepend artistic category if needed
        img_cat = resolved.get("image_category")
        if img_cat and final_prompt:
            artistic_kw = ["anime", "illustration", "painting", "art", "render", "manga", "chibi", "webtoon", "vtuber"]
            if any(kw in img_cat.lower() for kw in artistic_kw):
                if not final_prompt.lower().startswith(img_cat.lower().split()[0]):
                    final_prompt = f"{img_cat}, {final_prompt}"

        # Append custom text
        append_text = custom_text_append.strip()
        if append_text:
            sep = " " if final_prompt.endswith(",") else ", "
            final_prompt = f"{final_prompt.rstrip()}{sep}{append_text}" if final_prompt else append_text

        # SFW safety check
        if content_rating == "SFW":
            err = enforce_sfw(final_prompt)
            if err:
                blocked = "[Blocked: potential NSFW content detected]"
                return (blocked, negative_prompt, blocked)

        return (final_prompt, negative_prompt, final_prompt)

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        prompt_style: str,
        retain_face: bool,
        style_meta: Dict,
        selections: Dict,
        desc_maps: Dict[str, Dict[str, str]],
        custom_text: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        """Invoke Ollama LLM to generate prompt."""
        img_cat = selections.get("image_category")

        # Determine if artistic style
        artistic_styles = {
            "anime", "semi-real anime", "vtuber", "waifu", "kemonomimi", "concept art",
            "fantasy illustration", "sci-fi illustration", "digital painting", "3d character",
            "game character", "oil painting", "watercolor", "art nouveau", "pop art",
            "cyberpunk artwork", "manga", "chibi", "webtoon"
        }
        is_artistic = img_cat and any(s in img_cat.lower() for s in artistic_styles)

        # Build style directive
        if is_artistic:
            style_directive = (
                f"CRITICAL: Output MUST be {img_cat} style, NOT a photograph. "
                f"Start with '{img_cat},' and use artistic terms (illustrated, stylized). "
                "NEVER use: photo, camera, DSLR, realistic skin."
            )
        elif img_cat:
            style_directive = f"Style: {img_cat}. Match visual conventions exactly."
        else:
            style_directive = ""

        # Load system prompt
        template = "system_prompts/character_prompt_system_retain_face.txt" if retain_face else "system_prompts/character_prompt_system.txt"
        system_prompt = load_system_prompt_template(template, content_rating, style_directive=style_directive)

        # Build attribute string (excluding image_category which is handled separately)
        attrs = ", ".join(f"{k}: {v}" for k, v in selections.items() if v and k != "image_category")

        # Build context from descriptions
        context_lines = []
        for key, val in selections.items():
            if val and key in desc_maps and val in desc_maps[key]:
                context_lines.append(f"{key.upper()}: {desc_maps[key][val]}")

        guidance = style_meta.get("guidance", "Comma-separated descriptors")
        token_limit = min(max_tokens, style_meta.get("token_limit", 1024))

        user_prompt = (
            f"Generate a {prompt_style} {'face-preserving edit' if retain_face else 'image'} prompt.\n"
            f"{'PRIMARY STYLE: ' + img_cat + chr(10) if img_cat else ''}"
            f"{chr(10).join(context_lines) + chr(10) if context_lines else ''}"
            f"Attributes: {attrs}\n"
            f"{f'Notes: {custom_text}' + chr(10) if custom_text else ''}"
            f"Format: {guidance}. Under {token_limit} tokens. Output only the prompt:"
        )

        ok, result = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={"temperature": float(temperature), "num_predict": int(token_limit) * 4, "seed": int(seed)},
            timeout=120,
        )

        if not ok:
            return f"[Error: {result}]"

        # Clean common LLM prefixes
        for prefix in ["Here is", "Here's", "This prompt", "Prompt:", f"{prompt_style}:"]:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].lstrip(": ")

        return result or "[Empty response]"


NODE_CLASS_MAPPINGS = {"WizdroidCharacterPrompt": WizdroidCharacterPromptNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidCharacterPrompt": "🧙 Wizdroid: Character Prompt"}
