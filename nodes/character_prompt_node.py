import json
import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import logging

from lib.content_safety import CONTENT_RATING_CHOICES, enforce_sfw
from lib.data_files import load_json
from lib.ollama_client import DEFAULT_OLLAMA_URL, collect_models, generate_text
from lib.system_prompts import load_system_prompt_template
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
POSE_RATING_CHOICES = ("SFW only", "NSFW only", "Mixed")
DEFAULT_OLLAMA_URL = "http://localhost:11434"

_PROMPT_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 100

logger = logging.getLogger(__name__)


def _option_name(item: Any) -> Optional[str]:
    if isinstance(item, str):
        name = item.strip()
        return name or None
    if isinstance(item, dict):
        name = str(item.get("name") or item.get("value") or item.get("label") or "").strip()
        return name or None
    return None


def _option_description(item: Any) -> str:
    if isinstance(item, dict):
        desc = item.get("description")
        if desc is None:
            desc = item.get("desc")
        return str(desc or "").strip()
    return ""


def _normalize_option_list(options: Any) -> List[str]:
    """Normalize a list of option items into a list of option names.

    Supports either:
    - ["a", "b", ...]
    - [{"name": "a", "description": "..."}, ...]
    """

    if not isinstance(options, list):
        return []

    out: List[str] = []
    for item in options:
        name = _option_name(item)
        if name:
            out.append(name)
    return out


def _extract_descriptions(payload: Any) -> Dict[str, str]:
    """Build {option_name: description} mapping from list/dict(SFW/NSFW) payload."""

    out: Dict[str, str] = {}
    if isinstance(payload, dict):
        groups = [payload.get("sfw") or [], payload.get("nsfw") or []]
    elif isinstance(payload, list):
        groups = [payload]
    else:
        groups = []

    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            name = _option_name(item)
            if not name:
                continue
            desc = _option_description(item)
            if desc:
                out[name] = desc
    return out


def _with_random(options: List[str]) -> Tuple[str, ...]:
    """Prepend Random and none options to a list."""
    normalized = _normalize_option_list(options)
    values: List[str] = [RANDOM_LABEL, NONE_LABEL]
    for option in normalized:
        if option and option != NONE_LABEL:
            values.append(option)
    return tuple(values)


def _choose(value: Optional[str], options: List[Any], rng: random.Random) -> Optional[str]:
    """Select a value, handling Random and none cases."""
    normalized = _normalize_option_list(options)
    if value == RANDOM_LABEL:
        pool = [opt for opt in normalized if opt != NONE_LABEL]
        selection = rng.choice(pool) if pool else None
    else:
        selection = value
    return None if selection == NONE_LABEL or selection is None else selection


def _split_groups(payload: Any) -> Tuple[List[str], List[str]]:
    """Split payload into SFW and NSFW lists."""
    if isinstance(payload, dict):
        sfw_raw = payload.get("sfw") or []
        nsfw_raw = payload.get("nsfw") or []
        sfw: List[str] = []
        nsfw: List[str] = []
        if isinstance(sfw_raw, list):
            for item in sfw_raw:
                name = _option_name(item)
                if name:
                    sfw.append(name)
        if isinstance(nsfw_raw, list):
            for item in nsfw_raw:
                name = _option_name(item)
                if name:
                    nsfw.append(name)
        return sfw, nsfw
    elif isinstance(payload, list):
        sfw: List[str] = []
        for item in payload:
            name = _option_name(item)
            if name:
                sfw.append(name)
        return sfw, []
    return [], []


def _pool_for_rating(rating: str, sfw: List[str], nsfw: List[str]) -> List[str]:
    """Get appropriate pool based on content rating."""
    if rating == "SFW only":
        return sfw
    if rating == "NSFW only":
        return nsfw
    return sfw + nsfw


def _choose_for_rating(value: Optional[str], sfw: List[str], nsfw: List[str], rating: str, rng: random.Random) -> Optional[str]:
    """Choose value respecting content rating."""
    if value == RANDOM_LABEL:
        pool = [opt for opt in _pool_for_rating(rating, sfw, nsfw) if opt != NONE_LABEL]
        if not pool:
            pool = [opt for opt in (sfw + nsfw) if opt != NONE_LABEL]
        return rng.choice(pool) if pool else None
    return None if value == NONE_LABEL or value is None else value


def _get_background_groups(payload: Any) -> Dict[str, List[str]]:
    """Extract background style groups from payload."""
    default = {"studio_controlled": [], "public_exotic_real": [], "imaginative_surreal": []}
    if isinstance(payload, dict):
        return {k: list(payload.get(k) or []) for k in default}
    elif isinstance(payload, list):
        return {**default, "studio_controlled": list(payload)}
    return default


def _get_option_data() -> Dict[str, Any]:
    """Load and process all option data. Single source of truth."""
    option_map = load_json("character_options.json")
    prompt_styles = load_json("prompt_styles.json")
    region_options = load_json("regions.json")
    country_options = load_json("countries.json")
    culture_options = load_json("cultures.json")
    
    pose_sfw, pose_nsfw = _split_groups(option_map.get("pose_style"))
    image_sfw, image_nsfw = _split_groups(option_map.get("image_category"))
    image_desc_map = _extract_descriptions(option_map.get("image_category"))
    fashion_style_desc_map = _extract_descriptions(option_map.get("fashion_style"))
    fashion_outfit_desc_map = _extract_descriptions(option_map.get("fashion_outfit"))
    makeup_style_desc_map = _extract_descriptions(option_map.get("makeup_style"))
    lighting_style_desc_map = _extract_descriptions(option_map.get("lighting_style"))
    camera_lens_desc_map = _extract_descriptions(option_map.get("camera_lens"))
    color_palette_desc_map = _extract_descriptions(option_map.get("color_palette"))
    background_groups = _get_background_groups(option_map.get("background_style"))
    
    return {
        "option_map": option_map,
        "prompt_styles": prompt_styles,
        "region_options": region_options,
        "country_options": country_options,
        "culture_options": culture_options,
        "pose_sfw": pose_sfw,
        "pose_nsfw": pose_nsfw,
        "image_sfw": image_sfw,
        "image_nsfw": image_nsfw,
        "image_desc_map": image_desc_map,
        "fashion_style_desc_map": fashion_style_desc_map,
        "fashion_outfit_desc_map": fashion_outfit_desc_map,
        "makeup_style_desc_map": makeup_style_desc_map,
        "lighting_style_desc_map": lighting_style_desc_map,
        "camera_lens_desc_map": camera_lens_desc_map,
        "color_palette_desc_map": color_palette_desc_map,
        "background_groups": background_groups,
    }


def _cache_key(selections: Dict, prompt_style: str, retain_face: bool, custom_text: str, temperature: float) -> str:
    """Generate cache key for prompt caching."""
    data = json.dumps(selections, sort_keys=True) + prompt_style + str(retain_face) + custom_text + str(temperature)
    return hashlib.md5(data.encode()).hexdigest()


class CharacterPromptBuilder:
    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "preview")
    FUNCTION = "build_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        data = _get_option_data()
        opt = data["option_map"]
        bg = data["background_groups"]
        
        # Fetch models dynamically
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)
        
        return {
            "required": {
                # === LLM Settings ===
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW only"}),
                "prompt_style": (tuple(data["prompt_styles"].keys()), {"default": "SDXL"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 50, "max": 2048, "step": 10}),
                
                # === Character Identity ===
                "character_name": ("STRING", {"default": ""}),
                "retain_face": ("BOOLEAN", {"default": False}),
                "gender": (_with_random(opt["gender"]), {"default": RANDOM_LABEL}),
                "race": (_with_random(opt.get("race") or [NONE_LABEL]), {"default": NONE_LABEL}),
                "age_group": (_with_random(opt["age_group"]), {"default": RANDOM_LABEL}),
                "body_type": (_with_random(opt["body_type"]), {"default": NONE_LABEL}),
                
                # === Appearance (optional - default none) ===
                "hair_color": (_with_random(opt["hair_color"]), {"default": NONE_LABEL}),
                "hair_style": (_with_random(opt["hair_style"]), {"default": NONE_LABEL}),
                "eye_color": (_with_random(opt["eye_color"]), {"default": NONE_LABEL}),
                "makeup_style": (_with_random(opt["makeup_style"]), {"default": NONE_LABEL}),
                
                # === Expression & Pose ===
                "facial_expression": (_with_random(opt["facial_expression"]), {"default": RANDOM_LABEL}),
                "face_angle": (_with_random(opt["face_angle"]), {"default": NONE_LABEL}),
                "pose_content_rating": (POSE_RATING_CHOICES, {"default": "SFW only"}),
                "pose_style": (_with_random(data["pose_sfw"] + data["pose_nsfw"]), {"default": RANDOM_LABEL}),
                
                # === Fashion ===
                "fashion_outfit": (_with_random(opt.get("fashion_outfit") or [NONE_LABEL]), {"default": RANDOM_LABEL}),
                "fashion_style": (_with_random(opt.get("fashion_style") or [NONE_LABEL]), {"default": RANDOM_LABEL}),
                "footwear_style": (_with_random(opt.get("footwear_style") or [NONE_LABEL]), {"default": NONE_LABEL}),
                "upcycled_fashion": (_with_random(opt.get("upcycled_materials") or [NONE_LABEL]), {"default": NONE_LABEL}),
                
                # === Scene & Camera ===
                "image_category": (_with_random(data["image_sfw"] + data["image_nsfw"]), {"default": RANDOM_LABEL}),
                "image_content_rating": (POSE_RATING_CHOICES, {"default": "SFW only"}),
                "background_stage_style": (_with_random(bg["studio_controlled"] or [NONE_LABEL]), {"default": NONE_LABEL}),
                "background_location_style": (_with_random(bg["public_exotic_real"] or [NONE_LABEL]), {"default": NONE_LABEL}),
                "background_imaginative_style": (_with_random(bg["imaginative_surreal"] or [NONE_LABEL]), {"default": NONE_LABEL}),
                "lighting_style": (_with_random(opt.get("lighting_style") or [NONE_LABEL]), {"default": RANDOM_LABEL}),
                "camera_angle": (_with_random(opt["camera_angle"]), {"default": NONE_LABEL}),
                "camera_lens": (_with_random(opt.get("camera_lens") or [NONE_LABEL]), {"default": NONE_LABEL}),
                "color_palette": (_with_random(opt.get("color_palette") or [NONE_LABEL]), {"default": NONE_LABEL}),
                
                # === Cultural Context ===
                "region": (_with_random(data["region_options"]["regions"]), {"default": NONE_LABEL}),
                "country": (_with_random(data["country_options"]["countries"]), {"default": NONE_LABEL}),
                "culture": (_with_random(data["culture_options"]["cultures"]), {"default": NONE_LABEL}),
                
                # === Custom Text ===
                "custom_text_llm": ("STRING", {"multiline": True, "default": ""}),
                "custom_text_append": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
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
        pose_content_rating: str,
        pose_style: str,
        fashion_outfit: str,
        fashion_style: str,
        footwear_style: str,
        upcycled_fashion: str,
        image_category: str,
        image_content_rating: str,
        background_stage_style: str,
        background_location_style: str,
        background_imaginative_style: str,
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
    ):
        data = _get_option_data()
        opt = data["option_map"]
        bg = data["background_groups"]
        rng = random.Random(seed)

        if content_rating != "NSFW allowed":
            if pose_content_rating != "SFW only" or image_content_rating != "SFW only":
                msg = (
                    "[ERROR: content_rating is 'SFW only' but pose_content_rating/image_content_rating are not both 'SFW only'. "
                    "Set both to 'SFW only' or switch content_rating to 'NSFW allowed'.]"
                )
                return msg, "", msg

        # Resolve all selections
        resolved = {
            "character_name": character_name.strip() or None,
            "image_category": _choose_for_rating(image_category, data["image_sfw"], data["image_nsfw"], image_content_rating, rng),
            "gender": _choose(gender, opt["gender"], rng),
            "race": _choose(race, opt.get("race") or [], rng),
            "age_group": _choose(age_group, opt["age_group"], rng),
            "body_type": _choose(body_type, opt["body_type"], rng),
            "hair_color": _choose(hair_color, opt["hair_color"], rng),
            "hair_style": _choose(hair_style, opt["hair_style"], rng),
            "eye_color": _choose(eye_color, opt["eye_color"], rng),
            "facial_expression": _choose(facial_expression, opt["facial_expression"], rng),
            "face_angle": _choose(face_angle, opt["face_angle"], rng),
            "camera_angle": _choose(camera_angle, opt["camera_angle"], rng),
            "pose_style": _choose_for_rating(pose_style, data["pose_sfw"], data["pose_nsfw"], pose_content_rating, rng),
            "makeup_style": _choose(makeup_style, opt["makeup_style"], rng),
            "fashion_outfit": _choose(fashion_outfit, opt.get("fashion_outfit") or [], rng),
            "fashion_style": _choose(fashion_style, opt.get("fashion_style") or [], rng),
            "footwear_style": _choose(footwear_style, opt.get("footwear_style") or [], rng),
            "upcycled_fashion": _choose(upcycled_fashion, opt.get("upcycled_materials") or [], rng),
            "background_stage_style": _choose(background_stage_style, bg["studio_controlled"], rng),
            "background_location_style": _choose(background_location_style, bg["public_exotic_real"], rng),
            "background_imaginative_style": _choose(background_imaginative_style, bg["imaginative_surreal"], rng),
            "lighting_style": _choose(lighting_style, opt.get("lighting_style") or [], rng),
            "camera_lens": _choose(camera_lens, opt.get("camera_lens") or [], rng),
            "color_palette": _choose(color_palette, opt.get("color_palette") or [], rng),
            "region": _choose(region, data["region_options"]["regions"], rng),
            "country": _choose(country, data["country_options"]["countries"], rng),
            "culture": _choose(culture, data["culture_options"]["cultures"], rng),
        }

        # Enforce SFW-only pool restrictions even for manual selections.
        # (The dropdown includes both SFW and NSFW strings, so we must validate.)
        resolved_image_category = resolved.get("image_category")
        if image_content_rating == "SFW only" and resolved_image_category in set(data["image_nsfw"]):
            msg = (
                "[ERROR: image_content_rating is 'SFW only' but selected image_category is NSFW. "
                "Choose an SFW image_category or set image_content_rating to 'Mixed'/'NSFW only' (and content_rating to 'NSFW allowed' if needed).]"
            )
            return msg, "", msg

        resolved_pose_style = resolved.get("pose_style")
        if pose_content_rating == "SFW only" and resolved_pose_style in set(data["pose_nsfw"]):
            msg = (
                "[ERROR: pose_content_rating is 'SFW only' but selected pose_style is NSFW. "
                "Choose an SFW pose_style or set pose_content_rating to 'Mixed'/'NSFW only' (and content_rating to 'NSFW allowed' if needed).]"
            )
            return msg, "", msg

        # Smart country selection based on region
        if resolved.get("region") and country == RANDOM_LABEL:
            resolved["country"] = f"culturally authentic country within {resolved['region']}"

        # Consolidate background
        resolved["background_style"] = (
            resolved.get("background_stage_style")
            or resolved.get("background_location_style")
            or resolved.get("background_imaginative_style")
        )

        style_meta = data["prompt_styles"][prompt_style]
        custom_text = custom_text_llm.strip()
        
        # Check cache
        cache_key = _cache_key(resolved, prompt_style, retain_face, custom_text, temperature)
        if cache_key in _PROMPT_CACHE:
            llm_response = _PROMPT_CACHE[cache_key]
        else:
            image_category_desc = data.get("image_desc_map", {}).get(resolved.get("image_category") or "", "")
            fashion_style_desc = data.get("fashion_style_desc_map", {}).get(resolved.get("fashion_style") or "", "")
            fashion_outfit_desc = data.get("fashion_outfit_desc_map", {}).get(resolved.get("fashion_outfit") or "", "")
            makeup_style_desc = data.get("makeup_style_desc_map", {}).get(resolved.get("makeup_style") or "", "")
            lighting_style_desc = data.get("lighting_style_desc_map", {}).get(resolved.get("lighting_style") or "", "")
            camera_lens_desc = data.get("camera_lens_desc_map", {}).get(resolved.get("camera_lens") or "", "")
            color_palette_desc = data.get("color_palette_desc_map", {}).get(resolved.get("color_palette") or "", "")
            llm_response = self._invoke_llm(
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                content_rating=content_rating,
                prompt_style=prompt_style,
                retain_face=retain_face,
                style_meta=style_meta,
                selections=resolved,
                image_category_desc=image_category_desc,
                fashion_style_desc=fashion_style_desc,
                fashion_outfit_desc=fashion_outfit_desc,
                makeup_style_desc=makeup_style_desc,
                lighting_style_desc=lighting_style_desc,
                camera_lens_desc=camera_lens_desc,
                color_palette_desc=color_palette_desc,
                custom_text=custom_text,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            # Cache result (with size limit)
            if len(_PROMPT_CACHE) >= _MAX_CACHE_SIZE:
                _PROMPT_CACHE.pop(next(iter(_PROMPT_CACHE)))
            _PROMPT_CACHE[cache_key] = llm_response

        negative_prompt = style_meta.get("negative_prompt", "")
        final_prompt = (llm_response or "").strip()
        
        # Ensure image_category is at the start for artistic styles
        image_category = resolved.get("image_category")
        if image_category and final_prompt:
            artistic_keywords = ["anime", "illustration", "painting", "art", "render", "concept", "manga", "chibi", "webtoon", "vtuber", "waifu"]
            is_artistic = any(kw in image_category.lower() for kw in artistic_keywords)
            
            # Check if the category is already at the start
            if is_artistic and not final_prompt.lower().startswith(image_category.lower().split()[0]):
                # Prepend the category to ensure it's emphasized
                final_prompt = f"{image_category}, {final_prompt}"
        
        append_text = custom_text_append.strip()

        if append_text and final_prompt:
            separator = " " if final_prompt.endswith(",") else ", "
            final_prompt = f"{final_prompt.rstrip()}{separator}{append_text}"
        elif append_text:
            final_prompt = append_text

        if content_rating != "NSFW allowed":
            err = enforce_sfw(final_prompt)
            if err:
                blocked = "[Blocked: potential NSFW content detected. Switch content_rating to 'NSFW allowed'.]"
                return blocked, negative_prompt, blocked

        return final_prompt, negative_prompt, final_prompt

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        prompt_style: str,
        retain_face: bool,
        style_meta: Dict,
        selections: Dict,
        image_category_desc: str,
        fashion_style_desc: str,
        fashion_outfit_desc: str,
        makeup_style_desc: str,
        lighting_style_desc: str,
        camera_lens_desc: str,
        color_palette_desc: str,
        custom_text: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        """Invoke Ollama LLM with token limit enforcement."""
        
        # Extract image_category as PRIMARY style directive
        image_category = selections.get("image_category")
        image_category_desc = (image_category_desc or "").strip()

        fashion_style = selections.get("fashion_style")
        fashion_style_desc = (fashion_style_desc or "").strip()

        fashion_outfit = selections.get("fashion_outfit")
        fashion_outfit_desc = (fashion_outfit_desc or "").strip()

        makeup_style = selections.get("makeup_style")
        makeup_style_desc = (makeup_style_desc or "").strip()

        lighting_style = selections.get("lighting_style")
        lighting_style_desc = (lighting_style_desc or "").strip()

        camera_lens = selections.get("camera_lens")
        camera_lens_desc = (camera_lens_desc or "").strip()

        color_palette = selections.get("color_palette")
        color_palette_desc = (color_palette_desc or "").strip()
        
        # Determine if this is an artistic/illustrated style vs photorealistic
        artistic_styles = {
            "anime style", "semi-real anime", "vtuber model", "waifu", "kemonomimi",
            "concept art", "fantasy illustration", "sci-fi illustration", "digital painting",
            "3D character render", "game character art", "oil painting portrait", 
            "watercolor portrait", "art nouveau poster", "pop art style", "cyberpunk artwork",
            "pin-up art", "manga style", "chibi art", "webtoon style"
        }
        
        is_artistic = image_category and any(
            style in image_category.lower() 
            for style in [s.lower() for s in artistic_styles]
        )
        
        # Build style-specific system prompt
        if is_artistic:
            style_directive = (
                f"CRITICAL: This MUST be {image_category} style - NOT a photograph. "
                f"Start the prompt with '{image_category},' or '{image_category} of'. "
                "Use artistic descriptors (illustrated, drawn, rendered, stylized). "
                "NEVER use photographic terms (photo, photograph, camera, DSLR, realistic skin)."
            )
        elif image_category:
            style_directive = (
                f"This is a {image_category}. Match the visual style exactly. "
                "Use appropriate terminology for this category."
            )
        else:
            style_directive = ""
        
        system_prompt = load_system_prompt_template(
            "system_prompts/character_prompt_system_retain_face.txt" if retain_face else "system_prompts/character_prompt_system.txt",
            content_rating,
            style_directive=style_directive,
        )

        # Build compact attribute list (image_category handled separately)
        attrs = ", ".join(
            f"{k.replace('_', ' ')}: {v}" 
            for k, v in selections.items() 
            if v is not None and k != "image_category"
        )
        
        guidance = style_meta.get('guidance', 'Single paragraph with comma-separated descriptors')
        token_limit = min(max_tokens, style_meta.get('token_limit', 1024))
        
        # Emphasize image category in user prompt
        category_instruction = ""
        if image_category:
            if is_artistic:
                category_instruction = f"PRIMARY STYLE: {image_category} (artistic/illustrated - NOT a photo)\n"
            else:
                category_instruction = f"PRIMARY STYLE: {image_category}\n"

        if image_category and image_category_desc:
            category_instruction = f"{category_instruction}STYLE MEANING: {image_category_desc}\n"

        if fashion_style and fashion_style_desc:
            category_instruction = f"{category_instruction}FASHION STYLE MEANING: {fashion_style_desc}\n"

        if fashion_outfit and fashion_outfit_desc:
            category_instruction = f"{category_instruction}OUTFIT MEANING: {fashion_outfit_desc}\n"

        if makeup_style and makeup_style_desc:
            category_instruction = f"{category_instruction}MAKEUP MEANING: {makeup_style_desc}\n"

        if lighting_style and lighting_style_desc:
            category_instruction = f"{category_instruction}LIGHTING MEANING: {lighting_style_desc}\n"

        if camera_lens and camera_lens_desc:
            category_instruction = f"{category_instruction}CAMERA/LENS MEANING: {camera_lens_desc}\n"

        if color_palette and color_palette_desc:
            category_instruction = f"{category_instruction}COLOR PALETTE MEANING: {color_palette_desc}\n"
        
        user_prompt = (
            f"Create a {prompt_style} {'face-preserving edit' if retain_face else 'image generation'} prompt.\n"
            f"{category_instruction}"
            f"Attributes: {attrs}\n"
            f"{f'Notes: {custom_text}' if custom_text else ''}\n"
            f"Format: {guidance}\n"
            f"Start with the style/category. Keep under {token_limit} tokens. Output only the prompt:"
        )

        ok, result = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={
                "temperature": float(temperature),
                "num_predict": int(token_limit) * 4,
                "seed": int(seed),
            },
            timeout=120,
        )

        if not ok:
            return f"[Error: {result}]"

        # Clean up common LLM prefixes
        prefixes_to_remove = [
            "Here is",
            "Here's",
            "This prompt",
            "Prompt:",
            f"{style_meta.get('label', '')}:",
            f"{ollama_model}:",
        ]
        for prefix in prefixes_to_remove:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].lstrip(": ")

        return result or "[Empty response from Ollama]"


NODE_CLASS_MAPPINGS = {
    "CharacterPromptBuilder": CharacterPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterPromptBuilder": "Character Prompt Builder",
}
