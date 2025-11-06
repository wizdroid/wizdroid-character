import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class UpcycledFashionNode:
    """
    ComfyUI node to generate professional upcycled fashion supermodel prompts featuring
    everyday objects transformed into glamorous designer wear.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("upcycled_fashion_prompt", "preview")
    FUNCTION = "generate_upcycled_fashion_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        character_options = _load_json("character_options.json")
        upcycled_materials = _load_json("upcycled_materials.json")
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
                "upcycled_material": (_with_random(upcycled_materials["upcycled_materials"], allow_none=False), {"default": RANDOM_LABEL}),
                "glamour_enhancement": (_with_random(glamour_options, allow_none=False), {"default": RANDOM_LABEL}),
                "gender": (_with_random(character_options["gender"]), {"default": RANDOM_LABEL}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            }
        }

    def generate_upcycled_fashion_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        upcycled_material: str,
        glamour_enhancement: str,
        gender: str,
        custom_text: str,
        seed: int = 0,
    ) -> Tuple[str]:
        character_options = _load_json("character_options.json")
        upcycled_materials = _load_json("upcycled_materials.json")
        prompt_styles = _load_json("prompt_styles.json")
        glamour_options_data = _load_json("glamour_options.json")

        glamour_options = glamour_options_data["glamour_options"]

        rng = random.Random(seed)

        # Each entry already pairs garment, material, styling, and makeup for coherent looks
        resolved_material = _choose(upcycled_material, upcycled_materials["upcycled_materials"], rng)
        resolved_glamour = _choose(glamour_enhancement, glamour_options, rng)
        resolved_gender = _choose(gender, character_options["gender"], rng)

        gender_prefix = f"{resolved_gender} " if resolved_gender else ""

        # Get style configuration
        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        token_limit = style_config["token_limit"]

        # Build the prompt instruction for the LLM
        system_prompt = f"""You are a professional text-to-image prompt engineer specializing in upcycled fashion and sustainable design. Create vivid, detailed prompts for high-end fashion supermodel photography featuring everyday objects transformed into glamorous designer wear.

TARGET MODEL: {style_label}
FORMATTING STYLE: {style_guidance}
TOKEN LIMIT: {token_limit} tokens maximum

Your prompts should include:
1. Subject description (supermodel with appropriate gender/identity)
2. Upcycled material transformation (how everyday objects become designer garments)
3. Design innovation (creative use of materials, textures, shapes)
4. Glamour styling (professional makeup, hair, accessories)
5. Pose (confident, fashion-forward runway poses)
6. Setting (high-fashion editorial environment)
7. Lighting (studio lighting emphasizing material textures)
8. Overall aesthetic (sustainable luxury, innovative design)

CRITICAL: Follow the formatting style EXACTLY as specified for {style_label}. Keep within {token_limit} tokens. Focus on creative upcycling while maintaining high-fashion aesthetic. Output only the prompt, no explanations or meta-commentary. Never include reasoning traces, deliberation markers, or text enclosed in '<think>' or similar tags. Begin with a vivid descriptor, never the model/style name (Flux, SDXL, Qwen, HiDream, etc.)."""

        user_prompt = f"""Create a professional upcycled fashion photography prompt for a {gender_prefix}supermodel showcasing the following sustainable couture concept.

    Primary concept: {resolved_material}
    Glamour direction: {resolved_glamour}

    Requirements:
    - Translate the concept details into an elevated runway-ready garment
    - Describe the couture construction, texture play, and silhouette innovation
    - Align makeup and hair styling with the provided concept aesthetic
    - Specify statement accessories or props that reinforce the sustainable narrative
    - Set the scene and lighting to spotlight material transformation and luxury finish
    - Maintain a confident, editorial pose with museum-quality composition
    - Ensure overall tone celebrates avant-garde eco-conscious fashion

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

        print(f"[UpcycledFashion] Generating prompt for {resolved_material} upcycled material")
        print(f"[UpcycledFashion] Glamour enhancement: {resolved_glamour}")
        print(f"[UpcycledFashion] Gender: {resolved_gender or 'unspecified'}")
        print(f"[UpcycledFashion] Prompt style: {style_label} (max {token_limit} tokens)")
        print(f"[UpcycledFashion] Using model: {ollama_model}")

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
            print(f"[UpcycledFashion] Sending request to {ollama_url}")
            print(f"[UpcycledFashion] Model: {payload.get('model')}")

            response = requests.post(ollama_url, json=payload, timeout=120)

            print(f"[UpcycledFashion] Response status: {response.status_code}")

            response.raise_for_status()
            data = response.json()

            result = (data.get("response") or "").strip()
            print(f"[UpcycledFashion] Received response ({len(result)} chars): {result[:100]}...")

            if not result:
                print("[UpcycledFashion] WARNING: Empty response from Ollama")
                return "[Empty response from LLM]"

            return result
        except requests.exceptions.HTTPError as exc:
            error_msg = f"[UpcycledFashion] HTTP error: {exc}"
            print(error_msg)
            if hasattr(exc.response, 'text'):
                print(f"[UpcycledFashion] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            error_msg = f"[UpcycledFashion] Connection error: {exc}"
            print(error_msg)
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            error_msg = f"[UpcycledFashion] Timeout error: {exc}"
            print(error_msg)
            return "[ERROR: Request timed out]"
        except Exception as exc:
            error_msg = f"[UpcycledFashion] Error invoking Ollama: {exc}"
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
            print(f"[UpcycledFashion] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "WizdroidUpcycledFashion": UpcycledFashionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidUpcycledFashion": "Upcycled Fashion",
}