import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


_JSON_CACHE: Dict[str, Tuple[int, Any]] = {}


def _load_json(name: str) -> Any:
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
    values: List[str] = [RANDOM_LABEL, NONE_LABEL]
    for option in options:
        if option == NONE_LABEL:
            continue
        values.append(option)
    return tuple(values)


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



def _choose(value: Optional[str], options: List[str], rng: random.Random) -> Optional[str]:
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt != NONE_LABEL]
        if not pool:
            pool = options[:]
        selection = rng.choice(pool)
    else:
        selection = value

    if selection == NONE_LABEL or selection is None:
        return None
    return selection


class CharacterPromptBuilder:
    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "preview")
    FUNCTION = "build_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        option_map = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        region_options = _load_json("regions.json")
        country_options = _load_json("countries.json")
        culture_options = _load_json("cultures.json")
        upcycled_values = option_map.get("upcycled_materials") or [NONE_LABEL]
        lighting_styles = option_map.get("lighting_style") or [NONE_LABEL]
        camera_lenses = option_map.get("camera_lens") or [NONE_LABEL]
        color_palettes = option_map.get("color_palette") or [NONE_LABEL]
        ollama_models = cls._collect_ollama_models()

        return {
            "required": {
                "character_name": ("STRING", {"default": ""}),
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (tuple(prompt_styles.keys()), {"default": "SDXL"}),
                "retain_face": ("BOOLEAN", {"default": False}),
                "image_category": (_with_random(option_map["image_category"]), {"default": RANDOM_LABEL}),
                "gender": (_with_random(option_map["gender"]), {"default": RANDOM_LABEL}),
                "age_group": (_with_random(option_map["age_group"]), {"default": RANDOM_LABEL}),
                "body_type": (_with_random(option_map["body_type"]), {"default": RANDOM_LABEL}),
                "hair_color": (_with_random(option_map["hair_color"]), {"default": RANDOM_LABEL}),
                "hair_style": (_with_random(option_map["hair_style"]), {"default": RANDOM_LABEL}),
                "eye_color": (_with_random(option_map["eye_color"]), {"default": RANDOM_LABEL}),
                "facial_expression": (_with_random(option_map["facial_expression"]), {"default": RANDOM_LABEL}),
                "face_angle": (_with_random(option_map["face_angle"]), {"default": RANDOM_LABEL}),
                "camera_angle": (_with_random(option_map["camera_angle"]), {"default": RANDOM_LABEL}),
                "pose_style": (_with_random(option_map["pose_style"]), {"default": RANDOM_LABEL}),
                "makeup_style": (_with_random(option_map["makeup_style"]), {"default": RANDOM_LABEL}),
                "fashion_style": (_with_random(option_map["fashion_style"]), {"default": RANDOM_LABEL}),
                "upcycled_fashion": (_with_random(upcycled_values), {"default": NONE_LABEL}),
                "background_style": (_with_random(option_map["background_style"]), {"default": RANDOM_LABEL}),
                "lighting_style": (_with_random(lighting_styles), {"default": RANDOM_LABEL}),
                "camera_lens": (_with_random(camera_lenses), {"default": RANDOM_LABEL}),
                "color_palette": (_with_random(color_palettes), {"default": RANDOM_LABEL}),
                "region": (_with_random(region_options["regions"]), {"default": RANDOM_LABEL}),
                "country": (_with_random(country_options["countries"]), {"default": RANDOM_LABEL}),
                "culture": (_with_random(culture_options["cultures"]), {"default": RANDOM_LABEL}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            }
        }

    def build_prompt(
        self,
        character_name: str,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        retain_face: bool,
        image_category: str,
        gender: str,
        age_group: str,
        body_type: str,
        hair_color: str,
        hair_style: str,
        eye_color: str,
        facial_expression: str,
        face_angle: str,
        camera_angle: str,
        pose_style: str,
        makeup_style: str,
        fashion_style: str,
        background_style: str,
        lighting_style: str,
        camera_lens: str,
        color_palette: str,
        upcycled_fashion: str,
        region: str,
        country: str,
        culture: str,
        custom_text: str,
        seed: int = 0,
    ):
        option_map = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        region_options = _load_json("regions.json")
        country_options = _load_json("countries.json")
        culture_options = _load_json("cultures.json")
        rng = random.Random(seed)
        upcycled_values = option_map.get("upcycled_materials") or [NONE_LABEL]
        lighting_styles = option_map.get("lighting_style") or [NONE_LABEL]
        camera_lenses = option_map.get("camera_lens") or [NONE_LABEL]
        color_palettes = option_map.get("color_palette") or [NONE_LABEL]

        resolved = {
            "character_name": character_name.strip() or None,
            "image_category": _choose(image_category, option_map["image_category"], rng),
            "gender": _choose(gender, option_map["gender"], rng),
            "age_group": _choose(age_group, option_map["age_group"], rng),
            "body_type": _choose(body_type, option_map["body_type"], rng),
            "hair_color": _choose(hair_color, option_map["hair_color"], rng),
            "hair_style": _choose(hair_style, option_map["hair_style"], rng),
            "eye_color": _choose(eye_color, option_map["eye_color"], rng),
            "facial_expression": _choose(facial_expression, option_map["facial_expression"], rng),
            "face_angle": _choose(face_angle, option_map["face_angle"], rng),
            "camera_angle": _choose(camera_angle, option_map["camera_angle"], rng),
            "pose_style": _choose(pose_style, option_map["pose_style"], rng),
            "makeup_style": _choose(makeup_style, option_map["makeup_style"], rng),
            "fashion_style": _choose(fashion_style, option_map["fashion_style"], rng),
            "lighting_style": _choose(lighting_style, lighting_styles, rng),
            "camera_lens": _choose(camera_lens, camera_lenses, rng),
            "color_palette": _choose(color_palette, color_palettes, rng),
            "upcycled_fashion": _choose(upcycled_fashion, upcycled_values, rng),
            "background_style": _choose(background_style, option_map["background_style"], rng),
            "region": _choose(region, region_options["regions"], rng),
            "country": _choose(country, country_options["countries"], rng),
            "culture": _choose(culture, culture_options["cultures"], rng),
        }

        resolved_region = resolved.get("region")
        if resolved_region and country == RANDOM_LABEL:
            resolved["country"] = f"choose a culturally authentic country within {resolved_region}"

        style_meta = prompt_styles[prompt_style]
        llm_response = self._invoke_llm(
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            prompt_style=prompt_style,
            retain_face=retain_face,
            style_meta=style_meta,
            selections=resolved,
            custom_text=custom_text.strip(),
        )

        negative_prompt = style_meta.get("negative_prompt", "")
        suffix = custom_text.strip()
        final_prompt = llm_response if not suffix else f"{llm_response}\n\n{suffix}"
        return final_prompt, negative_prompt, final_prompt

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        """
        Fetch available Ollama models from the API using HTTP requests.
        This is the proper way to query Ollama models as shown in ollama_base.py reference.
        """
        try:
            if requests is None:
                print("[CharacterPromptBuilder] 'requests' library not installed")
                return ["install_requests_library"]
            
            # Query the /api/tags endpoint to get available models
            tags_url = f"{ollama_url}/api/tags"
            print(f"[CharacterPromptBuilder] Querying Ollama at: {tags_url}")
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            print(f"[CharacterPromptBuilder] Found {len(models)} Ollama models: {models}")
            return models if models else ["no_models_found"]
        except requests.exceptions.ConnectionError as e:
            print(f"[CharacterPromptBuilder] Cannot connect to Ollama at {ollama_url}: {e}")
            return ["ollama_not_running"]
        except requests.exceptions.Timeout as e:
            print(f"[CharacterPromptBuilder] Ollama request timeout: {e}")
            return ["ollama_timeout"]
        except Exception as e:
            print(f"[CharacterPromptBuilder] Error fetching Ollama models: {type(e).__name__}: {e}")
            return ["ollama_error"]

    @staticmethod
    def _invoke_llm(ollama_url: str, ollama_model: str, prompt_style: str, retain_face: bool, style_meta: Dict, selections: Dict, custom_text: str) -> str:
        """
        Invoke Ollama LLM using HTTP API (proper method) instead of subprocess.
        """
        if retain_face:
            system_prompt = (
                "You are a text-to-image prompt engineer for face-preserving image editing models (Flux Kontext, Qwen Image Edit). "
                "Create concise prompts that preserve the original face while modifying other aspects. "
                "ALWAYS start prompts with 'Retain the facial features from the original image.' Then describe outfit, pose, setting changes. "
                "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', the model/style name (Flux, SDXL, Qwen, HiDream, etc.), or any meta preface. "
                "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s). "
                "Ensure the character's cultural identity aligns with any provided 'region' or 'country' selections. "
                "Use 'culture' selection to guide traditional or culturally-specific outfit choices and styling. "
                "If 'upcycled_fashion' is provided, weave that sustainable couture concept into garment construction, textures, and detailing. "
                "Treat 'fashion_style' as the definitive reference for wardrobe aesthetic, garment silhouettes, accessories, and fabrics. "
                "Use 'makeup_style' for cosmetic direction and 'background_style' for scene context. "
                "Never include reasoning traces, deliberation markers, or text enclosed in '<think>' or similar tags."
            )
        else:
            system_prompt = (
                "You are a text-to-image prompt engineer. Create concise, vivid prompts that honor all specified attributes. "
                "Be specific and descriptive but avoid excessive verbosity. "
                "ALWAYS describe clothing details (garment type, color, fabric) and pose (body position, stance). "
                "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', the model/style name (Flux, SDXL, Qwen, HiDream, etc.), or any meta preface. "
                "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s). "
                "Ensure the character's cultural identity aligns with any provided 'region' or 'country' selections. "
                "Use 'culture' selection to guide traditional or culturally-specific outfit choices and styling. "
                "If 'upcycled_fashion' is provided, weave that sustainable couture concept into garment construction, textures, and detailing. "
                "Treat 'fashion_style' as the definitive reference for wardrobe aesthetic, garment silhouettes, accessories, and fabrics. "
                "Use 'makeup_style' for cosmetic direction and 'background_style' for scene context. "
                "Never include reasoning traces, deliberation markers, or text enclosed in '<think>' or similar tags."
            )

        # Build attribute list, filtering out None values
        attr_parts = []
        for key, value in selections.items():
            if value is not None:
                attr_parts.append(f"{key.replace('_', ' ')}: {value}")
        
        if retain_face:
            lines = [
                f"Create a {prompt_style} face-preserving image edit prompt using these attributes:",
                ", ".join(attr_parts),
            ]
        else:
            lines = [
                f"Create a {prompt_style} image generation prompt using these attributes:",
                ", ".join(attr_parts),
            ]
        
        lines.append(
            "\nAttribute glossary: 'fashion_style' defines the outfit concept (clothing pieces, accessories, textures); "
            "'makeup_style' guides cosmetics; 'background_style' sets the environment; 'pose_style' directs body language; "
            "'lighting_style' sets illumination mood; 'camera_lens' informs framing and depth; "
            "'color_palette' locks overall chroma direction; "
            "'region' establishes cultural/geographic origin; 'country' specifies national identity when provided; "
            "'culture' guides traditional or culturally-specific outfit choices and styling; "
            "'upcycled_fashion' highlights sustainable couture concepts and material innovations when present."
        )

        if custom_text:
            lines.append(f"\nAdditional notes: {custom_text}")

        if retain_face:
            lines.extend([
                f"\nFormat: {style_meta.get('guidance', 'Single paragraph with comma-separated descriptors')}",
                "CRITICAL - Start with: 'Retain the facial features from the original image.'",
                "Then describe:",
                "  * New outfit/clothing: specific garment types, colors, fabrics, style details",
                "  * New pose: exact body position, limb placement, gesture, stance",
                "  * New setting/background, lighting, atmosphere",
                "Do NOT describe facial features (eyes, nose, mouth, face shape) - these are preserved from original",
                "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', or similar",
                "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
                "Remove any reasoning or planning text; do not include '<think>' or similar tags",
                "Output only the final prompt text:"
            ])
        else:
            lines.extend([
                f"\nFormat: {style_meta.get('guidance', 'Single paragraph with comma-separated descriptors')}",
                "CRITICAL - You MUST describe:",
                "  * Outfit/clothing: specific garment types, colors, fabrics, style details",
                "  * Pose: exact body position, limb placement, gesture, stance",
                "  * Lighting, atmosphere, camera angle",
                "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', or similar",
                "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
                "Remove any reasoning or planning text; do not include '<think>' or similar tags",
                "Output only the final prompt text:"
            ])

        user_prompt = "\n".join(lines)

        try:
            if requests is None:
                return "[Please install 'requests' library: pip install requests]"
            
            # Use the HTTP API endpoint for generation
            generate_url = f"{ollama_url}/api/generate"
            payload = {
                "model": ollama_model,
                "prompt": f"{user_prompt}",
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                }
            }
            
            response = requests.post(generate_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "[Empty response from Ollama]").strip()
            
        except requests.exceptions.ConnectionError:
            return "[Ollama server not running. Please start Ollama.]"
        except requests.exceptions.Timeout:
            return "[Ollama request timed out]"
        except Exception as e:
            print(f"[CharacterPromptBuilder] Error invoking LLM: {e}")
            return f"[Error: {str(e)}]"

NODE_CLASS_MAPPINGS = {
    "CharacterPromptBuilder": CharacterPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterPromptBuilder": "Character Prompt Builder",
}
