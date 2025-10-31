import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _load_json(name: str) -> Dict:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _with_random(options: List[str]) -> Tuple[str, ...]:
    return tuple([RANDOM_LABEL] + options)


def _choose(value: str, options: List[str]) -> Optional[str]:
    if value == RANDOM_LABEL:
        return random.choice(options)
    return value


class FashionSupermodelNode:
    """
    ComfyUI node to generate professional fashion supermodel prompts featuring
    traditional cultural outfits from various countries with glamorous styling.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fashion_prompt",)
    FUNCTION = "generate_fashion_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        character_options = _load_json("character_options.json")
        country_options = _load_json("countries.json")
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
                "glamour_enhancement": (_with_random(glamour_options), {"default": RANDOM_LABEL}),
                "gender": (_with_random(character_options["gender"]), {"default": RANDOM_LABEL}),
            }
        }

    def generate_fashion_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        country: str,
        glamour_enhancement: str,
        gender: str,
    ) -> Tuple[str]:
        character_options = _load_json("character_options.json")
        country_options = _load_json("countries.json")
        prompt_styles = _load_json("prompt_styles.json")
        glamour_options_data = _load_json("glamour_options.json")

        glamour_options = glamour_options_data["glamour_options"]

        resolved_country = _choose(country, country_options["countries"])
        resolved_glamour = _choose(glamour_enhancement, glamour_options)
        resolved_gender = _choose(gender, character_options["gender"])

        # Get style configuration
        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        token_limit = style_config["token_limit"]

        # Build the prompt instruction for the LLM
        system_prompt = f"""You are a professional text-to-image prompt engineer specializing in fashion photography and cultural traditional attire. Create vivid, detailed prompts for high-end fashion supermodel photography featuring traditional cultural outfits.

TARGET MODEL: {style_label}
FORMATTING STYLE: {style_guidance}
TOKEN LIMIT: {token_limit} tokens maximum

Your prompts should include:
1. Subject description (supermodel with appropriate gender/identity)
2. Traditional outfit details (garments, fabrics, colors, patterns, cultural elements)
3. Accessories and jewelry (culturally appropriate)
4. Makeup and hair styling (glamorous yet culturally respectful)
5. Pose (glamorous, confident, fashion-forward)
6. Background/setting (culturally relevant or neutral fashion backdrop)
7. Lighting (professional studio lighting, high-end fashion photography)
8. Overall mood and atmosphere

CRITICAL: Follow the formatting style EXACTLY as specified for {style_label}. Keep within {token_limit} tokens. Focus on fashion photography aesthetics while respecting cultural authenticity. Output only the prompt, no explanations or meta-commentary."""

        user_prompt = f"""Create a professional fashion photography prompt for a {resolved_gender} supermodel wearing traditional {resolved_country} outfit with {resolved_glamour} glamour enhancement.

Requirements:
- Traditional {resolved_country} outfit as the foundation
- Enhance the glamour with {resolved_glamour} styling approach
- Professional studio lighting setup
- Glamorous pose suitable for high-fashion editorial
- Detailed description of traditional outfit elements enhanced with modern glamour
- Culturally appropriate accessories and jewelry with luxurious details
- Professional makeup and hair styling that complements the traditional outfit
- Sophisticated background that honors the culture while maintaining fashion editorial aesthetic
- High-end fashion photography aesthetic
- Confident, powerful supermodel presence

IMPORTANT: Format this prompt EXACTLY according to {style_label} style: {style_guidance}
Keep it under {token_limit} tokens.

Generate the prompt now:"""

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

        print(f"[FashionSupermodel] Generating prompt for {resolved_country} traditional outfit")
        print(f"[FashionSupermodel] Glamour enhancement: {resolved_glamour}")
        print(f"[FashionSupermodel] Gender: {resolved_gender}")
        print(f"[FashionSupermodel] Prompt style: {style_label} (max {token_limit} tokens)")
        print(f"[FashionSupermodel] Using model: {ollama_model}")

        response = self._invoke_ollama(generate_url, payload)

        if not response or response.startswith("[ERROR"):
            return (f"Failed to generate prompt: {response}",)

        return (response,)

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
    "WizdroidFashionSupermodel": "Fashion Supermodel (Wizdroid)",
}
