"""🧙 Wizdroid Character Prompt Nodes - Gender-specific character prompts using Ollama LLM."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from wizdroid_lib.constants import DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import filter_by_gender
from wizdroid_lib.registry import DataRegistry
from wizdroid_lib.helpers import (
    choose,
    extract_descriptions,
    normalize_option_list,
    split_groups,
    with_random,
)
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

logger = logging.getLogger(__name__)


class _BaseWizdroidCharacterPromptNode:
    """🧙 Base character prompt node - shared logic for male/female subclasses."""

    CATEGORY = "🧙 Wizdroid/Prompts"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"

    GENDER = None  # Override in subclasses to "male" or "female"

    @classmethod
    def _build_inputs(cls, gender: str):
        """Build INPUT_TYPES with options pre-filtered for the given gender."""
        opt = DataRegistry.get_character_options()
        styles = DataRegistry.get_prompt_styles()
        regions = DataRegistry.get_regions()
        countries = DataRegistry.get_countries()
        cultures = DataRegistry.get_cultures()

        body_data = DataRegistry.get_body_types() or {}
        emotions_data = DataRegistry.get_emotions() or {}
        eye_data = DataRegistry.get_eye_colors() or {}
        hair_data = DataRegistry.get_hair() or {}
        skin_data = DataRegistry.get_skin_tones() or {}
        makeup_data = DataRegistry.get_makeup() or {}
        poses_data = DataRegistry.get_poses() or {}
        bg_data = DataRegistry.get_backgrounds() or {}
        cam_data = DataRegistry.get_camera_lighting() or {}
        fashion_data = DataRegistry.get_fashion() or {}

        # Gender-filtered option lists
        body_types = normalize_option_list(
            filter_by_gender(body_data.get("body_types", []), gender)
        )
        makeup_styles = normalize_option_list(
            filter_by_gender(makeup_data.get("makeup_styles", []), gender)
        )
        fashion_outfits = normalize_option_list(
            filter_by_gender(fashion_data.get("fashion_outfits", []), gender)
        )
        fashion_styles_list = normalize_option_list(
            filter_by_gender(fashion_data.get("fashion_styles", []), gender)
        )

        # Gender-specific poses
        pose_list = (
            poses_data.get("pose_styles", {}).get("any", [])
            + poses_data.get("pose_styles", {}).get(gender, [])
        )

        # Non-gender-filtered shared data
        emotions_nested = emotions_data.get("emotions", {})
        emotions = []
        if isinstance(emotions_nested, dict):
            for cat in emotions_nested.values():
                emotions.extend(normalize_option_list(cat))
        else:
            emotions = normalize_option_list(emotions_nested)
        eye_colors = eye_data.get("eye_colors", [])
        hair_colors = hair_data.get("hair_colors", [])
        hair_styles = (
            hair_data.get("hair_styles", {}).get("any", [])
            + hair_data.get("hair_styles", {}).get(gender, [])
        )
        skin_tones = skin_data.get("skin_tones", [])
        footwear = fashion_data.get("footwear_styles", [])

        backgrounds = bg_data.get("backgrounds", {})
        bg_studio = backgrounds.get("studio_controlled", [])
        bg_real = backgrounds.get("public_exotic_real", [])
        bg_imaginative = backgrounds.get("imaginative_surreal", [])

        lighting_styles = normalize_option_list(cam_data.get("lighting_styles", []))
        camera_angles = cam_data.get("camera_angles", [])
        camera_lenses = normalize_option_list(cam_data.get("camera_lenses", []))
        color_palettes = normalize_option_list(cam_data.get("color_palettes", []))
        face_angles = cam_data.get("face_angles", [])

        image_categories = split_groups(opt.get("image_category"))
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (tuple(styles.keys()), {"default": "SDXL"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 50, "max": 2048, "step": 10}),
                "character_name": ("STRING", {"default": ""}),
                "retain_face": ("BOOLEAN", {"default": False}),
                "use_ai": ("BOOLEAN", {"default": True}),
                "race": (with_random(opt.get("race", [])), {"default": NONE_LABEL}),
                "age_group": (with_random(opt.get("age_group", [])), {"default": RANDOM_LABEL}),
                "body_type": (with_random(body_types), {"default": NONE_LABEL}),
                "skin_tone": (with_random(skin_tones), {"default": NONE_LABEL}),
                "hair_color": (with_random(hair_colors), {"default": NONE_LABEL}),
                "hair_style": (with_random(hair_styles), {"default": NONE_LABEL}),
                "eye_color": (with_random(eye_colors), {"default": NONE_LABEL}),
                "makeup_type": (with_random(makeup_styles), {"default": NONE_LABEL}),
                "facial_expression": (with_random(emotions), {"default": RANDOM_LABEL}),
                "face_angle": (with_random(face_angles), {"default": NONE_LABEL}),
                "pose_style": (with_random(pose_list), {"default": RANDOM_LABEL}),
                "fashion_outfit": (with_random(fashion_outfits), {"default": RANDOM_LABEL}),
                "fashion_style": (with_random(fashion_styles_list), {"default": RANDOM_LABEL}),
                "footwear_style": (with_random(footwear), {"default": NONE_LABEL}),
                "image_category": (with_random(image_categories), {"default": RANDOM_LABEL}),
                "background_stage": (with_random(bg_studio), {"default": NONE_LABEL}),
                "background_location": (with_random(bg_real), {"default": NONE_LABEL}),
                "background_imaginative": (with_random(bg_imaginative), {"default": NONE_LABEL}),
                "lighting_style": (with_random(lighting_styles), {"default": RANDOM_LABEL}),
                "camera_angle": (with_random(camera_angles), {"default": NONE_LABEL}),
                "camera_lens": (with_random(camera_lenses), {"default": NONE_LABEL}),
                "color_palette": (with_random(color_palettes), {"default": NONE_LABEL}),
                "region": (with_random(regions.get("regions", [])), {"default": NONE_LABEL}),
                "country": (with_random(countries.get("countries", [])), {"default": NONE_LABEL}),
                "culture": (with_random(cultures.get("cultures", [])), {"default": NONE_LABEL}),
                "custom_text_llm": ("STRING", {"multiline": True, "default": ""}),
                "custom_text_append": ("STRING", {"multiline": True, "default": ""}),
                "output_language": (
                    ["Disable", "English", "Deutsch", "中文", "日本語", "Español", "Français", "Português", "Italiano", "Русский", "한국어", "العربية", "हिन्दी"],
                    {"default": "Disable"},
                ),
                "spiciness": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "detail_level": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "fantasy": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    # Subclasses override INPUT_TYPES via _build_inputs
    @classmethod
    def INPUT_TYPES(cls):
        return cls._build_inputs(cls.GENDER or "male")

    def build_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        temperature: float,
        max_tokens: int,
        character_name: str,
        retain_face: bool,
        use_ai: bool,
        race: str,
        age_group: str,
        body_type: str,
        skin_tone: str,
        hair_color: str,
        hair_style: str,
        eye_color: str,
        makeup_type: str,
        facial_expression: str,
        face_angle: str,
        pose_style: str,
        fashion_outfit: str,
        fashion_style: str,
        footwear_style: str,
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
        output_language: str = "Disable",
        spiciness: int = 0,
        detail_level: int = 5,
        fantasy: int = 0,
        seed: int = 0,
    ) -> str:
        gender = self.GENDER
        opt = DataRegistry.get_character_options()
        styles = DataRegistry.get_prompt_styles()
        regions_data = DataRegistry.get_regions()
        countries_data = DataRegistry.get_countries()
        cultures_data = DataRegistry.get_cultures()
        body_data = DataRegistry.get_body_types() or {}
        emotions_data = DataRegistry.get_emotions() or {}
        eye_data = DataRegistry.get_eye_colors() or {}
        hair_data = DataRegistry.get_hair() or {}
        skin_data = DataRegistry.get_skin_tones() or {}
        makeup_data = DataRegistry.get_makeup() or {}
        poses_data = DataRegistry.get_poses() or {}
        bg_data = DataRegistry.get_backgrounds() or {}
        cam_data = DataRegistry.get_camera_lighting() or {}
        fashion_data = DataRegistry.get_fashion() or {}

        rng = random.Random(seed)

        # Gender-filtered option lists
        body_types = normalize_option_list(
            filter_by_gender(body_data.get("body_types", []), gender)
        )
        makeup_styles_list = normalize_option_list(
            filter_by_gender(makeup_data.get("makeup_styles", []), gender)
        )
        fashion_outfits_list = normalize_option_list(
            filter_by_gender(fashion_data.get("fashion_outfits", []), gender)
        )
        fashion_styles_list = normalize_option_list(
            filter_by_gender(fashion_data.get("fashion_styles", []), gender)
        )

        # Gender-specific poses
        pose_list = (
            poses_data.get("pose_styles", {}).get("any", [])
            + poses_data.get("pose_styles", {}).get(gender, [])
        )

        # Non-gender-filtered shared data
        emotions_nested = emotions_data.get("emotions", {})
        emotions = []
        if isinstance(emotions_nested, dict):
            for category in emotions_nested.values():
                emotions.extend(normalize_option_list(category))
        else:
            emotions = normalize_option_list(emotions_nested)
        eye_colors = eye_data.get("eye_colors", [])
        hair_colors = hair_data.get("hair_colors", [])
        hair_styles = (
            hair_data.get("hair_styles", {}).get("any", [])
            + hair_data.get("hair_styles", {}).get(gender, [])
        )
        skin_tones = skin_data.get("skin_tones", [])
        footwear = fashion_data.get("footwear_styles", [])

        backgrounds = bg_data.get("backgrounds", {})
        bg_studio = backgrounds.get("studio_controlled", [])
        bg_real = backgrounds.get("public_exotic_real", [])
        bg_imaginative = backgrounds.get("imaginative_surreal", [])

        lighting_styles_list = normalize_option_list(cam_data.get("lighting_styles", []))
        camera_angles_list = cam_data.get("camera_angles", [])
        camera_lenses_list = normalize_option_list(cam_data.get("camera_lenses", []))
        color_palettes_list = normalize_option_list(cam_data.get("color_palettes", []))
        face_angles = cam_data.get("face_angles", [])

        image_categories = split_groups(opt.get("image_category"))

        # Resolve all selections
        def resolve(val: str, opts: List[Any]) -> Optional[str]:
            return choose(val, opts, rng, seed)

        resolved = {
            "character_name": character_name.strip() or None,
            "image_category": resolve(image_category, image_categories),
            "gender": gender,
            "race": resolve(race, opt.get("race", [])),
            "age_group": resolve(age_group, opt.get("age_group", [])),
            "body_type": resolve(body_type, body_types),
            "skin_tone": resolve(skin_tone, skin_tones),
            "hair_color": resolve(hair_color, hair_colors),
            "hair_style": resolve(hair_style, hair_styles),
            "eye_color": resolve(eye_color, eye_colors),
            "facial_expression": resolve(facial_expression, emotions),
            "face_angle": resolve(face_angle, face_angles),
            "camera_angle": resolve(camera_angle, camera_angles_list),
            "pose_style": resolve(pose_style, pose_list),
            "makeup_type": resolve(makeup_type, makeup_styles_list),
            "fashion_outfit": resolve(fashion_outfit, fashion_outfits_list),
            "fashion_style": resolve(fashion_style, fashion_styles_list),
            "footwear_style": resolve(footwear_style, footwear),
            "background_style": (
                resolve(background_stage, bg_studio)
                or resolve(background_location, bg_real)
                or resolve(background_imaginative, bg_imaginative)
            ),
            "lighting_style": resolve(lighting_style, lighting_styles_list),
            "camera_lens": resolve(camera_lens, camera_lenses_list),
            "color_palette": resolve(color_palette, color_palettes_list),
            "region": resolve(region, regions_data.get("regions", [])),
            "country": resolve(country, countries_data.get("countries", [])),
            "culture": resolve(culture, cultures_data.get("cultures", [])),
        }

        # Smart country selection based on region
        if resolved.get("region") and country == RANDOM_LABEL:
            resolved["country"] = f"authentic to {resolved['region']}"

        # Get style metadata
        style_meta = styles.get(prompt_style, {})
        custom_text = custom_text_llm.strip()

        if not use_ai:
            # Template mode: format selections directly, no Ollama call
            final_prompt = self._template_prompt(resolved, custom_text)
        else:
            # AI mode: invoke LLM
            desc_maps = {
                "image_category": extract_descriptions(opt.get("image_category")),
                "fashion_style": extract_descriptions(fashion_data.get("fashion_styles")),
                "fashion_outfit": extract_descriptions(fashion_data.get("fashion_outfits")),
                "makeup_type": extract_descriptions(makeup_data.get("makeup_styles")),
                "lighting_style": extract_descriptions(cam_data.get("lighting_styles")),
                "camera_lens": extract_descriptions(cam_data.get("camera_lenses")),
                "color_palette": extract_descriptions(cam_data.get("color_palettes")),
            }

            llm_response, raw_prompt = self._invoke_llm(
                ollama_url, ollama_model, prompt_style, retain_face,
                style_meta, resolved, desc_maps, custom_text, temperature, max_tokens, seed,
                output_language, spiciness, detail_level, fantasy,
            )

            if llm_response and llm_response.strip():
                final_prompt = llm_response.strip()
            else:
                final_prompt = raw_prompt if raw_prompt else "[Error: No prompt generated]"

        # Prepend artistic category if needed
        img_cat = resolved.get("image_category")
        if img_cat and final_prompt:
            artistic_kw = ["anime", "illustration", "painting", "art", "render", "manga", "chibi", "webtoon", "vtuber"]
            if any(kw in img_cat.lower() for kw in artistic_kw):
                if not final_prompt.lower().startswith(img_cat.lower().split()[0]):
                    final_prompt = f"{img_cat}, {final_prompt}"

        append_text = custom_text_append.strip()
        if append_text:
            sep = " " if final_prompt.endswith(",") else ", "
            final_prompt = f"{final_prompt.rstrip()}{sep}{append_text}" if final_prompt else append_text

        # Content safety check (skip when spiciness > 0)
        if spiciness == 0:
            err = enforce_sfw(final_prompt)
            if err:
                return ("[Blocked: potential NSFW content detected]",)

        return (final_prompt,)

    @staticmethod
    def _template_prompt(resolved: Dict[str, Optional[str]], custom_text: str) -> str:
        """Build a verbose template-based prompt from resolved selections (no AI required)."""

        def v(key: str, prefix: str = "", suffix: str = "") -> str:
            val = resolved.get(key)
            if not val:
                return ""
            # prettify underscores
            label = val.replace("_", " ")
            return f"{prefix}{label}{suffix}"

        parts = []

        # Character class line (image category)
        img_cat = v("image_category")
        if img_cat:
            parts.append(img_cat)

        # Core identity
        name = resolved.get("character_name") or ""
        gender = resolved.get("gender", "")
        age = v("age_group")
        race = v("race")
        body = v("body_type")

        identity = []
        if name:
            identity.append(f'named "{name}"')
        if age:
            identity.append(age)
        if race:
            identity.append(race)
        if body:
            identity.append(f"{body} build")
        if gender:
            identity.insert(0, gender)
        if identity:
            parts.append(" ".join(identity))

        # Physical features
        skin = v("skin_tone")
        hair_color = v("hair_color")
        hair_style = v("hair_style")
        eyes = v("eye_color")
        makeup = v("makeup_type")

        physical = []
        if skin:
            physical.append(f"{skin} skin")
        if hair_color or hair_style:
            h = f"{hair_color} {hair_style}".strip()
            physical.append(f"{h} hair")
        if eyes:
            physical.append(f"{eyes} eyes")
        if makeup:
            physical.append(f"{makeup} makeup")
        if physical:
            parts.append(", ".join(physical))

        # Expression and pose
        expr = v("facial_expression")
        face_ang = v("face_angle")
        pose = v("pose_style")
        action = []
        if expr:
            action.append(expr)
        if pose:
            action.append(pose)
        if face_ang:
            action.append(f"facing {face_ang}")
        if action:
            parts.append(", ".join(action))

        # Fashion
        outfit = v("fashion_outfit")
        style = v("fashion_style")
        shoes = v("footwear_style")
        wear = []
        if outfit or style:
            w = f"{outfit}{' in ' + style + ' style' if style else ''}"
            wear.append(w)
        if shoes:
            wear.append(f"{shoes}")
        if wear:
            parts.append("wearing " + ", ".join(wear))

        # Scene and camera
        bg = v("background_style")
        lighting = v("lighting_style")
        palette = v("color_palette")
        cam_angle = v("camera_angle")
        lens = v("camera_lens")

        scene = []
        if bg:
            scene.append(bg + " background")
        if lighting:
            scene.append(f"{lighting} lighting")
        if palette:
            scene.append(f"{palette} color palette")
        if cam_angle:
            scene.append(f"{cam_angle} shot")
        if lens:
            scene.append(f"{lens} lens")
        if scene:
            parts.append(", ".join(scene))

        # Culture context
        region = v("region")
        country = v("country")
        culture = v("culture")
        ctx = []
        if region:
            ctx.append(region)
        if country:
            ctx.append(country)
        if culture:
            ctx.append(f"{culture} culture")
        if ctx:
            parts.append("inspired by " + ", ".join(ctx))

        # Custom text
        if custom_text:
            parts.append(custom_text)

        prompt = ", ".join(parts)
        # Capitalize first letter
        if prompt:
            prompt = prompt[0].upper() + prompt[1:]
        return prompt or "[No selections made]"

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        retain_face: bool,
        style_meta: Dict,
        selections: Dict,
        desc_maps: Dict[str, Dict[str, str]],
        custom_text: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        output_language: str = "Disable",
        spiciness: int = 0,
        detail_level: int = 5,
        fantasy: int = 0,
    ) -> Tuple[str, str]:
        """Invoke Ollama LLM to generate prompt, returning (llm_response, raw_prompt)."""
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

        guidance = style_meta.get("guidance", "Comma-separated descriptors")
        token_limit = min(max_tokens, style_meta.get("token_limit", 1024))

        # Spiciness directive
        spice_levels = {
            0: "",
            1: "\nINSTRUCTION - Tone: mildly suggestive, artistic flirtation. Keep it classy and implied.",
            2: "\nINSTRUCTION - Tone: playful sensuality, tasteful allure. Suggestive but not explicit.",
            3: "\nINSTRUCTION - Tone: romantic intimacy, soft sensuality. Implied nudity is acceptable artistically.",
            4: "\nINSTRUCTION - Tone: sensual and warm. Artistic nudity and passionate undertones are permitted.",
            5: "\nINSTRUCTION - Tone: openly sensual. Erotic themes are allowed with artistic framing.",
            6: "\nINSTRUCTION - Tone: boldly erotic. Explicit sensuality and suggestive poses are permitted.",
            7: "\nINSTRUCTION - Tone: unabashedly erotic. Explicit adult content with vivid sensory detail is allowed.",
            8: "\nINSTRUCTION - Tone: raw and explicit. Full adult content without restrictions.",
            9: "\nINSTRUCTION - Tone: extremely explicit. Maximum intensity adult content. No boundaries.",
            10: "\nINSTRUCTION - Tone: MAXIMUM SPICINESS. Fully uncensored. Any and all explicit adult content, fetish, kink, and extreme themes are permitted without any restriction whatsoever. No holds barred.",
        }
        spice_directive = spice_levels.get(spiciness, "")

        # Detail level directive
        if detail_level <= 2:
            detail_d = f"\nINSTRUCTION - Write at detail level {detail_level}/10: Be extremely concise. Use only the most essential keywords and minimal descriptors. Omit any non-critical details."
        elif detail_level <= 5:
            detail_d = f"\nINSTRUCTION - Write at detail level {detail_level}/10: Use balanced, moderate detail. Include key visual elements but keep descriptions reasonably focused."
        elif detail_level <= 7:
            detail_d = f"\nINSTRUCTION - Write at detail level {detail_level}/10: Use rich descriptive language with concrete visual details, textures, and sensory cues."
        else:
            detail_d = f"\nINSTRUCTION - Write at detail level {detail_level}/10: Use lavishly detailed, multi-layered descriptions. Include evocative sensory language, precise textures, complex composition, and atmospheric nuances."

        # Fantasy level directive
        if fantasy == 0:
            fantasy_d = "\nINSTRUCTION - Realism level: Strictly realistic. Do NOT include any magical, mythical, sci-fi, or speculative elements. Keep everything grounded in reality."
        elif fantasy <= 3:
            fantasy_d = f"\nINSTRUCTION - Fantasy level {fantasy}/10: Subtle surreal or fantastical touches are acceptable, but keep the overall description feeling grounded and mostly realistic."
        elif fantasy <= 7:
            fantasy_d = f"\nINSTRUCTION - Fantasy level {fantasy}/10: Include significant fantasy, sci-fi, or magical elements. Blend the extraordinary with realistic details."
        else:
            fantasy_d = f"\nINSTRUCTION - Fantasy level {fantasy}/10: Fully fantastical. No reality constraints. Emphasize imaginative, mythical, surreal, and otherworldly elements without restriction."

        # Load system prompt
        template = "system_prompts/character_prompt_system_retain_face.txt" if retain_face else "system_prompts/character_prompt_system.txt"
        system_prompt = load_system_prompt_template(template, style_directive=style_directive, format_guidance=guidance)

        # Build context from descriptions
        context_lines = []
        for key, val in selections.items():
            if val and key in desc_maps and val in desc_maps[key]:
                context_lines.append(f"{key.upper()}: {desc_maps[key][val]}")

        # Build attribute string using only descriptions when available
        attrs_list = []
        for k, v in selections.items():
            if v and k != "image_category":
                # If we have a description for this value, use it; otherwise use the value name
                if k in desc_maps and v in desc_maps[k]:
                    attrs_list.append(f"{k}: {desc_maps[k][v]}")
                else:
                    attrs_list.append(f"{k}: {v}")
        attrs_with_desc = ", ".join(attrs_list)

        user_prompt = (
            f"Generate a {prompt_style} {'face-preserving edit' if retain_face else 'image'} prompt.\n"
            f"{'PRIMARY STYLE: ' + img_cat + chr(10) if img_cat else ''}"
            f"{chr(10).join(context_lines) + chr(10) if context_lines else ''}"
            f"Attributes: {attrs_with_desc}\n"
            f"{f'Notes: {custom_text}' + chr(10) if custom_text else ''}"
            f"Format: {guidance}. Under {token_limit} tokens. Output only the prompt."
            f"{f' Write the prompt entirely in {output_language}.' if output_language and output_language.lower() not in ('disable', 'english', 'en') else ''}"
            f"{spice_directive if spiciness > 0 else ''}"
            f"{detail_d}{fantasy_d}"
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
            return f"[Error: {result}]", user_prompt

        # Clean common LLM prefixes
        for prefix in ["Here is", "Here's", "This prompt", "Prompt:", f"{prompt_style}:"]:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].lstrip(": ")

        return (result or "[Empty response]", user_prompt)


class WizdroidCharacterPromptMaleNode(_BaseWizdroidCharacterPromptNode):
    """🧙 Generate male character prompts with male-specific options pre-filtered."""
    GENDER = "male"


class WizdroidCharacterPromptFemaleNode(_BaseWizdroidCharacterPromptNode):
    """🧙 Generate female character prompts with female-specific options pre-filtered."""
    GENDER = "female"


NODE_CLASS_MAPPINGS = {
    "WizdroidCharacterPromptMale": WizdroidCharacterPromptMaleNode,
    "WizdroidCharacterPromptFemale": WizdroidCharacterPromptFemaleNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidCharacterPromptMale": "🧙 Wizdroid: Character Prompt (Male)",
    "WizdroidCharacterPromptFemale": "🧙 Wizdroid: Character Prompt (Female)",
}
