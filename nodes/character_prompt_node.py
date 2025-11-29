import json
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

try:
    import requests
except ImportError:
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
POSE_RATING_CHOICES = ("SFW only", "NSFW only", "Mixed")
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Caches
_JSON_CACHE: Dict[str, Tuple[int, Any]] = {}
_PROMPT_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 100

logger = logging.getLogger(__name__)


def _load_json(name: str) -> Any:
    """Load JSON file with mtime-based caching."""
    path = DATA_DIR / name
    mtime = int(path.stat().st_mtime_ns)
    cached = _JSON_CACHE.get(name)
    if cached and cached[0] == mtime:
        return cached[1]

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    _JSON_CACHE[name] = (mtime, payload)
    return payload


def _with_random(options: List[str]) -> Tuple[str, ...]:
    """Prepend Random and none options to a list."""
    values: List[str] = [RANDOM_LABEL, NONE_LABEL]
    for option in options:
        if option != NONE_LABEL:
            values.append(option)
    return tuple(values)


def _choose(value: Optional[str], options: List[str], rng: random.Random) -> Optional[str]:
    """Select a value, handling Random and none cases."""
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt != NONE_LABEL]
        selection = rng.choice(pool) if pool else None
    else:
        selection = value
    return None if selection == NONE_LABEL or selection is None else selection


def _split_groups(payload: Any) -> Tuple[List[str], List[str]]:
    """Split payload into SFW and NSFW lists."""
    if isinstance(payload, dict):
        return list(payload.get("sfw") or []), list(payload.get("nsfw") or [])
    elif isinstance(payload, list):
        return list(payload), []
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
    option_map = _load_json("character_options.json")
    prompt_styles = _load_json("prompt_styles.json")
    region_options = _load_json("regions.json")
    country_options = _load_json("countries.json")
    culture_options = _load_json("cultures.json")
    
    pose_sfw, pose_nsfw = _split_groups(option_map.get("pose_style"))
    image_sfw, image_nsfw = _split_groups(option_map.get("image_category"))
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
        ollama_models = cls._collect_ollama_models()
        
        return {
            "required": {
                # === LLM Settings ===
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
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
            llm_response = self._invoke_llm(
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                prompt_style=prompt_style,
                retain_face=retain_face,
                style_meta=style_meta,
                selections=resolved,
                custom_text=custom_text,
                temperature=temperature,
                max_tokens=max_tokens,
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

        return final_prompt, negative_prompt, final_prompt

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        """Fetch available Ollama models from the API."""
        try:
            if requests is None:
                logger.warning("[CharacterPromptBuilder] 'requests' library not installed")
                return ["install_requests_library"]
            
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return models if models else ["no_models_found"]
        except requests.exceptions.ConnectionError:
            logger.warning(f"[CharacterPromptBuilder] Cannot connect to Ollama at {ollama_url}")
            return ["ollama_not_running"]
        except requests.exceptions.Timeout:
            logger.warning("[CharacterPromptBuilder] Ollama request timeout")
            return ["ollama_timeout"]
        except Exception as e:
            logger.exception(f"[CharacterPromptBuilder] Error fetching Ollama models: {e}")
            return ["ollama_error"]

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        retain_face: bool,
        style_meta: Dict,
        selections: Dict,
        custom_text: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Invoke Ollama LLM with token limit enforcement."""
        
        # Extract image_category as PRIMARY style directive
        image_category = selections.get("image_category")
        
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
        
        base_system = (
            "You are a text-to-image prompt engineer. Create concise, vivid prompts. "
            f"{style_directive} "
            "Your first words must establish the image style/category. "
            "Never start with 'Here', 'This', 'Prompt', or model names. "
            "Output ONLY the final prompt text - no explanations, no markdown, no meta commentary. "
            "Never include '<think>' tags or reasoning traces."
        )
        
        if retain_face:
            system_prompt = (
                f"{base_system} "
                "This is for face-preserving image editing. "
                "ALWAYS start with 'Retain the facial features from the original image.' "
                "Then describe outfit, pose, and setting changes. "
                "Do NOT describe facial features - they are preserved from original."
            )
        else:
            system_prompt = (
                f"{base_system} "
                "ALWAYS describe: clothing (type, color, fabric), pose (body position, stance), "
                "and technical aspects (lighting, camera angle)."
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
        
        user_prompt = (
            f"Create a {prompt_style} {'face-preserving edit' if retain_face else 'image generation'} prompt.\n"
            f"{category_instruction}"
            f"Attributes: {attrs}\n"
            f"{f'Notes: {custom_text}' if custom_text else ''}\n"
            f"Format: {guidance}\n"
            f"Start with the style/category. Keep under {token_limit} tokens. Output only the prompt:"
        )

        try:
            if requests is None:
                return "[Please install 'requests': pip install requests]"
            
            payload = {
                "model": ollama_model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": token_limit * 4,  # Rough token-to-char estimate
                }
            }
            
            response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            
            # Clean up common LLM prefixes
            prefixes_to_remove = [
                "Here is", "Here's", "This prompt", "Prompt:", 
                f"{style_meta.get('label', '')}:", f"{ollama_model}:"
            ]
            for prefix in prefixes_to_remove:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].lstrip(": ")
            
            return result or "[Empty response from Ollama]"
            
        except requests.exceptions.ConnectionError:
            return "[Ollama server not running. Please start Ollama.]"
        except requests.exceptions.Timeout:
            return "[Ollama request timed out]"
        except Exception as e:
            logger.exception(f"[CharacterPromptBuilder] Error invoking LLM: {e}")
            return f"[Error: {str(e)}]"


NODE_CLASS_MAPPINGS = {
    "CharacterPromptBuilder": CharacterPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterPromptBuilder": "Character Prompt Builder",
}
