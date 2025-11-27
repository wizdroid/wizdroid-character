import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _load_json(name: str) -> Any:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _with_random(options: List[str]) -> Tuple[str, ...]:
    values: List[str] = [RANDOM_LABEL, NONE_LABEL]
    for option in options:
        if option == NONE_LABEL:
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


def _get_background_groups(payload: Any) -> Dict[str, List[str]]:
    groups = {
        "studio_controlled": [],
        "public_exotic_real": [],
        "imaginative_surreal": [],
    }
    if isinstance(payload, dict):
        for key in groups:
            groups[key] = list(payload.get(key, []) or [])
    elif isinstance(payload, list):
        groups["studio_controlled"] = list(payload)
    return groups


class BackgroundGeneratorNode:
    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("background_prompt", "character_style_inspire")
    FUNCTION = "generate_background"

    @classmethod
    def INPUT_TYPES(cls):
        option_map = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        color_palettes = option_map.get("color_palette") or [NONE_LABEL]
        camera_lenses = option_map.get("camera_lens") or [NONE_LABEL]
        background_groups = _get_background_groups(option_map.get("background_style"))
        studio_backgrounds = background_groups["studio_controlled"] or [NONE_LABEL]
        real_backgrounds = background_groups["public_exotic_real"] or [NONE_LABEL]
        imaginative_backgrounds = background_groups["imaginative_surreal"] or [NONE_LABEL]

        themes = (
            "ethereal",
            "dreamlike",
            "cosmic",
            "gothic",
            "pastoral",
            "surreal",
            "abstract",
            "fantasy",
            "retro-futuristic",
        )

        ollama_models = cls._collect_ollama_models()

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (tuple(prompt_styles.keys()), {"default": "SDXL"}),
                "background_stage_style": (_with_random(studio_backgrounds), {"default": RANDOM_LABEL}),
                "background_location_style": (_with_random(real_backgrounds), {"default": NONE_LABEL}),
                "background_imaginative_style": (_with_random(imaginative_backgrounds), {"default": NONE_LABEL}),
                "fantasy_theme": (_with_random(list(themes)), {"default": RANDOM_LABEL}),
                "surreal_level": ("INT", {"default": 7, "min": 0, "max": 10}),
                "color_palette": (_with_random(color_palettes), {"default": RANDOM_LABEL}),
                "camera_lens": (_with_random(camera_lenses), {"default": RANDOM_LABEL}),
                "custom_inspire": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def generate_background(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        background_stage_style: str,
        background_location_style: str,
        background_imaginative_style: str,
        fantasy_theme: str,
        surreal_level: int,
        color_palette: str,
        camera_lens: str,
        custom_inspire: str,
        seed: int = 0,
    ) -> Tuple[str, str]:
        option_map = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")

        rng = random.Random(seed)
        background_groups = _get_background_groups(option_map.get("background_style"))
        studio_backgrounds = background_groups["studio_controlled"] or [NONE_LABEL]
        real_backgrounds = background_groups["public_exotic_real"] or [NONE_LABEL]
        imaginative_backgrounds = background_groups["imaginative_surreal"] or [NONE_LABEL]
        color_options = option_map.get("color_palette") or [NONE_LABEL]
        lens_options = option_map.get("camera_lens") or [NONE_LABEL]

        resolved_stage = _choose(background_stage_style, studio_backgrounds, rng)
        resolved_location = _choose(background_location_style, real_backgrounds, rng)
        resolved_imag = _choose(background_imaginative_style, imaginative_backgrounds, rng)
        resolved_theme = fantasy_theme if fantasy_theme != RANDOM_LABEL else None
        resolved_palette = _choose(color_palette, color_options, rng)
        resolved_lens = _choose(camera_lens, lens_options, rng)

        style_meta = prompt_styles.get(prompt_style, prompt_styles.get("SDXL", {}))

        # Build the LLM system prompt
        system_prompt = (
            "You are an LLM that generates vivid, cinematic background descriptions for image generation."
            " Respond with JSON containing two keys: 'background_prompt' and 'character_style_inspiration'."
            " 'background_prompt' should be a single evaluatable prompt describing the environment, lighting, mood, colors, and any fantastical/ surreal elements."
            " 'character_style_inspiration' should be a short description focusing on clothing, materials, color palette, accessories, and character silhouette inspired by the background."
            " Be concise and output valid JSON only."
        )

        # Compose user prompt
        user_parts = []
        if resolved_stage:
            user_parts.append(f"studio background: {resolved_stage}")
        if resolved_location:
            user_parts.append(f"real location: {resolved_location}")
        if resolved_imag:
            user_parts.append(f"imaginative layer: {resolved_imag}")
        if resolved_theme:
            user_parts.append(f"theme: {resolved_theme}")
        if surreal_level is not None:
            user_parts.append(f"surrealism: {surreal_level}/10")
        if resolved_palette:
            user_parts.append(f"color palette: {resolved_palette}")
        if resolved_lens:
            user_parts.append(f"camera lens: {resolved_lens}")
        if custom_inspire.strip():
            user_parts.append(f"custom: {custom_inspire.strip()}")

        prompt_description = ", ".join(user_parts)

        user_prompt = (
            f"Create a {prompt_style} background and character style inspiration. Respond in JSON with keys 'background_prompt' and 'character_style_inspiration'.\n"
            f"{prompt_description}\n"
            "Return valid JSON only. Ensure each field is short (1-3 sentences) and actionable for an image generation model."
        )

        # Build payload for Ollama
        generate_url = ollama_url
        if not generate_url.endswith("/api/generate"):
            generate_url = generate_url.rstrip("/") + "/api/generate"

        payload = {
            "model": ollama_model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.8},
        }

        try:
            if requests is None:
                return ("[Please install 'requests' library: pip install requests]", "")

            response = requests.post(generate_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()

            # Try to parse JSON
            try:
                parsed = json.loads(raw)
                bg_prompt = parsed.get("background_prompt", "")
                char_style = parsed.get("character_style_inspiration", "")
                return (bg_prompt.strip(), char_style.strip())
            except Exception:
                # If not JSON, return full text as background and empty style
                return (raw, "")

        except requests.exceptions.ConnectionError:
            return ("[Ollama server not running. Please start Ollama.]", "")
        except requests.exceptions.Timeout:
            return ("[Ollama request timed out]", "")
        except Exception as e:
            logging.getLogger(__name__).exception(f"[BackgroundGenerator] Error invoking LLM: {e}")
            return (f"[Error: {str(e)}]", "")

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        if requests is None:
            return ["install_requests_library"]
        try:
            tags_url = f"{ollama_url}/api/tags"
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            return models if models else ["no_models_found"]
        except requests.exceptions.ConnectionError:
            return ["ollama_not_running"]
        except requests.exceptions.Timeout:
            return ["ollama_timeout"]
        except Exception as exc:
            logging.getLogger(__name__).exception(f"[BackgroundGeneratorNode] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "BackgroundGeneratorNode": BackgroundGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundGeneratorNode": "Background Generator",
}
