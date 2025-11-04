import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@lru_cache(maxsize=None)
def _load_json(name: str) -> Dict:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _with_random(options: List[str], allow_none: bool = True) -> Tuple[str, ...]:
    values: List[str] = [RANDOM_LABEL]
    if allow_none:
        values.append(NONE_LABEL)
    for option in options:
        if allow_none and option == NONE_LABEL:
            continue
        values.append(option)
    return tuple(values)


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


class FashionSupermodelNode:
    """
    ComfyUI node to generate professional fashion supermodel prompts featuring
    contemporary fashion styles from specific countries or broader regions with glamorous styling.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fashion_prompt", "preview")
    FUNCTION = "generate_fashion_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        character_options = _load_json("character_options.json")
        country_options = _load_json("countries.json")
        region_options = _load_json("regions.json")
        prompt_styles = _load_json("prompt_styles.json")
        glamour_options_data = _load_json("glamour_options.json")
        ollama_models = cls._collect_ollama_models()

        style_options = [style_key for style_key in prompt_styles.keys()]
        glamour_options = glamour_options_data["glamour_options"]

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (style_options, {"default": "SDXL"}),
                "country": (_with_random(country_options["countries"]), {"default": RANDOM_LABEL}),
                "region": (_with_random(region_options["regions"]), {"default": NONE_LABEL}),
                "glamour_enhancement": (_with_random(glamour_options, allow_none=False), {"default": RANDOM_LABEL}),
                "gender": (_with_random(character_options["gender"]), {"default": RANDOM_LABEL}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            }
        }

    def generate_fashion_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        country: str,
        region: str,
        glamour_enhancement: str,
        gender: str,
        custom_text: str,
        seed: int = 0,
    ) -> Tuple[str]:
        character_options = _load_json("character_options.json")
        country_options = _load_json("countries.json")
        region_options = _load_json("regions.json")
        prompt_styles = _load_json("prompt_styles.json")
        glamour_options_data = _load_json("glamour_options.json")

        glamour_options = glamour_options_data["glamour_options"]

        rng = random.Random(seed)

        resolved_country = _choose(country, country_options["countries"], rng)
        resolved_region = _choose(region, region_options["regions"], rng)
        resolved_glamour = _choose(glamour_enhancement, glamour_options, rng)
        resolved_gender = _choose(gender, character_options["gender"], rng)

        # Determine location: prefer region if specified, otherwise use country, fallback to generic
        if resolved_region:
            location = resolved_region
            location_type = "regional"
        elif resolved_country:
            location = resolved_country
            location_type = "country"
        else:
            location = "contemporary"
            location_type = "generic"

        gender_prefix = f"{resolved_gender} " if resolved_gender else ""

        # Get style configuration
        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        token_limit = style_config["token_limit"]

        # Build the prompt instruction for the LLM
        if location_type == "generic":
            system_prompt = f"""You are a professional text-to-image prompt engineer specializing in contemporary fashion and modern style. Create vivid, detailed prompts for high-end fashion supermodel photography featuring current fashion trends and styles.

TARGET MODEL: {style_label}
FORMATTING STYLE: {style_guidance}
TOKEN LIMIT: {token_limit} tokens maximum

Your prompts should include:
1. Subject description (supermodel with appropriate gender/identity)
2. Contemporary fashion details (modern clothing styles, fabrics, colors, patterns)
3. Accessories and jewelry (current fashion trends)
4. Makeup and hair styling (contemporary beauty standards)
5. Pose (glamorous, confident, fashion-forward)
6. Background/setting (modern fashion editorial or neutral backdrop)
7. Lighting (professional studio lighting, high-end fashion photography)
8. Overall mood and atmosphere

CRITICAL: Follow the formatting style EXACTLY as specified for {style_label}. Keep within {token_limit} tokens. Focus on contemporary high-fashion aesthetics. Output only the prompt, no explanations or meta-commentary. Never include reasoning traces, deliberation markers, or text enclosed in '<think>' or similar tags. Begin with a vivid descriptor, never the model/style name (Flux, SDXL, Qwen, HiDream, etc.)."""
        else:
            system_prompt = f"""You are a professional text-to-image prompt engineer specializing in contemporary fashion and modern style. Create vivid, detailed prompts for high-end fashion supermodel photography featuring modern clothing styles from specific locations.

TARGET MODEL: {style_label}
FORMATTING STYLE: {style_guidance}
TOKEN LIMIT: {token_limit} tokens maximum

Your prompts should include:
1. Subject description (supermodel with appropriate gender/identity)
2. Location-specific fashion details (contemporary clothing styles, fabrics, colors, patterns from the {location_type})
3. Accessories and jewelry (modern designs inspired by the location)
4. Makeup and hair styling (contemporary yet location-influenced)
5. Pose (glamorous, confident, fashion-forward)
6. Background/setting (modern location-inspired or neutral fashion backdrop)
7. Lighting (professional studio lighting, high-end fashion photography)
8. Overall mood and atmosphere

CRITICAL: Follow the formatting style EXACTLY as specified for {style_label}. Keep within {token_limit} tokens. Focus on contemporary fashion while incorporating location-specific influences. Output only the prompt, no explanations or meta-commentary. Never include reasoning traces, deliberation markers, or text enclosed in '<think>' or similar tags. Begin with a vivid descriptor, never the model/style name (Flux, SDXL, Qwen, HiDream, etc.)."""

        if location_type == "generic":
            user_prompt = f"""Create a professional fashion photography prompt for a {gender_prefix}supermodel wearing contemporary fashion with {resolved_glamour} glamour enhancement.

Requirements:
- Contemporary fashion trends as the foundation
- Enhance the glamour with {resolved_glamour} styling approach
- Professional studio lighting setup
- Glamorous pose suitable for high-fashion editorial
- Detailed description of modern fashion elements enhanced with glamour
- Contemporary accessories and jewelry with current trends
- Professional makeup and hair styling that complements the fashion
- Sophisticated background that maintains fashion editorial aesthetic
- High-end fashion photography aesthetic
- Confident, powerful supermodel presence

IMPORTANT: Format this prompt EXACTLY according to {style_label} style: {style_guidance}
Keep it under {token_limit} tokens.

Generate the prompt now:"""
        else:
            user_prompt = f"""Create a professional fashion photography prompt for a {gender_prefix}supermodel wearing contemporary {location} fashion with {resolved_glamour} glamour enhancement.

Requirements:
- Contemporary {location} fashion as the foundation
- Enhance the glamour with {resolved_glamour} styling approach
- Professional studio lighting setup
- Glamorous pose suitable for high-fashion editorial
- Detailed description of location-inspired fashion elements enhanced with modern glamour
- Contemporary accessories and jewelry with location inspiration
- Professional makeup and hair styling that complements the fashion
- Sophisticated background that honors the location while maintaining fashion editorial aesthetic
- High-end fashion photography aesthetic
- Confident, powerful supermodel presence

IMPORTANT: Format this prompt EXACTLY according to {style_label} style: {style_guidance}
Keep it under {token_limit} tokens.

Generate the prompt now:"""

        # Add custom text if provided
        if custom_text.strip():
            user_prompt = f"{user_prompt}\n\nAdditional custom requirements: {custom_text.strip()}"

        # Ensure URL has the /api/generate endpoint
        generate_url = ollama_url
        if not generate_url.endswith("/api/generate"):
            generate_url = generate_url.rstrip("/") + "/api/generate"

        payload = {
            "model": ollama_model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": token_limit + 50,
                "temperature": 0.8,
            }
        }

        print(f"[FashionSupermodel] Generating prompt for {location} {location_type} fashion")
        print(f"[FashionSupermodel] Glamour enhancement: {resolved_glamour}")
        print(f"[FashionSupermodel] Gender: {resolved_gender or 'unspecified'}")
        print(f"[FashionSupermodel] Prompt style: {style_label} (max {token_limit} tokens)")
        print(f"[FashionSupermodel] Using model: {ollama_model}")

        response = self._invoke_ollama(generate_url, payload)

        if not response or response.startswith("[ERROR"):
            error_msg = f"Failed to generate prompt: {response}"
            return (error_msg, error_msg)

        return (response, response)

    @staticmethod
    def _invoke_ollama(ollama_url: str, payload: Dict) -> Optional[str]:
        if requests is None:
            raise RuntimeError("'requests' is required for Ollama integration. Install optional dependencies.")
        try:
            print(f"[FashionSupermodel] Sending request to {ollama_url}")
            print(f"[FashionSupermodel] Model: {payload.get('model')}")

            response = requests.post(ollama_url, json=payload, timeout=120)

            print(f"[FashionSupermodel] Response status: {response.status_code}")

            response.raise_for_status()
            data = response.json()

            result = (data.get("response") or "").strip()
            print(f"[FashionSupermodel] Received response ({len(result)} chars): {result[:100]}...")

            if not result:
                print("[FashionSupermodel] WARNING: Empty response from Ollama")
                return "[Empty response from LLM]"

            return result
        except requests.exceptions.HTTPError as exc:
            error_msg = f"[FashionSupermodel] HTTP error: {exc}"
            print(error_msg)
            if hasattr(exc.response, 'text'):
                print(f"[FashionSupermodel] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            error_msg = f"[FashionSupermodel] Connection error: {exc}"
            print(error_msg)
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            error_msg = f"[FashionSupermodel] Timeout error: {exc}"
            print(error_msg)
            return "[ERROR: Request timed out]"
        except Exception as exc:
            error_msg = f"[FashionSupermodel] Error invoking Ollama: {exc}"
            print(error_msg)
            return f"[ERROR: {str(exc)}]"

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        if requests is None:
            return ["install_requests_library"]
        try:
            tags_url = f"{ollama_url}/api/tags"
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            all_models = [model["name"] for model in models_data.get("models", [])]

            if not all_models:
                return ["no_models_found"]

            return all_models
        except requests.exceptions.ConnectionError:
            return ["ollama_not_running"]
        except requests.exceptions.Timeout:
            return ["ollama_timeout"]
        except Exception as exc:
            print(f"[FashionSupermodel] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "WizdroidFashionSupermodel": FashionSupermodelNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidFashionSupermodel": "Fashion Supermodel",
}
