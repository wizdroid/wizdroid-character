import json
import random
import logging
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

THEME_CHOICES = (
    "astral myth",
    "eldritch gothic",
    "dreamcore pastel",
    "storm-forged citadel",
    "sunken neon reef",
    "glacial cathedral",
    "lush bioluminescent jungle",
    "clockwork metropolis",
    "void labyrinth",
    "whimsical storybook",
)

TIME_OF_DAY_CHOICES = (
    "dawn",
    "golden hour",
    "midnight",
    "blue hour",
    "eclipse",
    "eternal night",
    "binary sunset",
)

WEATHER_CHOICES = (
    "clear and crystalline",
    "mist-laden",
    "electrical storm",
    "arcane aurora",
    "ember snow",
    "meteor shower",
)

MOOD_CHOICES = (
    "serene",
    "foreboding",
    "majestic",
    "whimsical",
    "chaotic",
    "sacred",
    "melancholic",
)

SCALE_CHOICES = (
    "intimate vignette",
    "sweeping vista",
    "colossal panorama",
)

CREATURE_CHOICES = (
    "none",
    "ambient sprites",
    "mystic guardians",
    "majestic beasts",
    "monstrous apex",
)

ARCHITECTURE_CHOICES = (
    "organic biomorphic",
    "ancient ruins",
    "brutalist monoliths",
    "floating pagodas",
    "shipwreck palaces",
    "crystal spires",
)

FOCAL_CHOICES = (
    "levitating core",
    "abyssal gateway",
    "sacred tree",
    "volcanic forge",
    "infinite library",
    "mirror lake",
)


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


def _with_random_tuple(options: Tuple[str, ...]) -> Tuple[str, ...]:
    return (RANDOM_LABEL, NONE_LABEL, *options)


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


def _choose_tuple(value: Optional[str], options: Tuple[str, ...], rng: random.Random) -> Optional[str]:
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt not in (NONE_LABEL, RANDOM_LABEL)]
        if not pool:
            pool = list(options)
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


class BackgroundEditNode:
    CATEGORY = "Wizdroid/backgrounds"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("background_prompt", "style_inspiration")
    FUNCTION = "generate_background_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        option_map = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        background_groups = _get_background_groups(option_map.get("background_style"))
        studio_backgrounds = background_groups["studio_controlled"] or [NONE_LABEL]
        real_backgrounds = background_groups["public_exotic_real"] or [NONE_LABEL]
        imaginative_backgrounds = background_groups["imaginative_surreal"] or [NONE_LABEL]
        color_palettes = option_map.get("color_palette") or [NONE_LABEL]
        lighting_styles = option_map.get("lighting_style") or [NONE_LABEL]
        camera_lenses = option_map.get("camera_lens") or [NONE_LABEL]

        ollama_models = cls._collect_ollama_models()

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (tuple(prompt_styles.keys()), {"default": "SDXL"}),
                "studio_background": (_with_random(studio_backgrounds), {"default": RANDOM_LABEL}),
                "real_background": (_with_random(real_backgrounds), {"default": NONE_LABEL}),
                "imaginative_background": (_with_random(imaginative_backgrounds), {"default": NONE_LABEL}),
                "theme": (_with_random_tuple(THEME_CHOICES), {"default": RANDOM_LABEL}),
                "time_of_day": (_with_random_tuple(TIME_OF_DAY_CHOICES), {"default": RANDOM_LABEL}),
                "weather": (_with_random_tuple(WEATHER_CHOICES), {"default": RANDOM_LABEL}),
                "mood": (_with_random_tuple(MOOD_CHOICES), {"default": RANDOM_LABEL}),
                "scale": (_with_random_tuple(SCALE_CHOICES), {"default": RANDOM_LABEL}),
                "focal_element": (_with_random_tuple(FOCAL_CHOICES), {"default": RANDOM_LABEL}),
                "architecture": (_with_random_tuple(ARCHITECTURE_CHOICES), {"default": RANDOM_LABEL}),
                "creature_presence": (_with_random_tuple(CREATURE_CHOICES), {"default": RANDOM_LABEL}),
                "lighting_style": (_with_random(lighting_styles), {"default": RANDOM_LABEL}),
                "color_palette": (_with_random(color_palettes), {"default": RANDOM_LABEL}),
                "camera_lens": (_with_random(camera_lenses), {"default": RANDOM_LABEL}),
                "custom_notes": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            }
        }

    def generate_background_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        studio_background: str,
        real_background: str,
        imaginative_background: str,
        theme: str,
        time_of_day: str,
        weather: str,
        mood: str,
        scale: str,
        focal_element: str,
        architecture: str,
        creature_presence: str,
        lighting_style: str,
        color_palette: str,
        camera_lens: str,
        custom_notes: str,
        seed: int = 0,
    ) -> Tuple[str, str]:
        option_map = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        rng = random.Random(seed)

        background_groups = _get_background_groups(option_map.get("background_style"))
        studio_backgrounds = background_groups["studio_controlled"] or [NONE_LABEL]
        real_backgrounds = background_groups["public_exotic_real"] or [NONE_LABEL]
        imaginative_backgrounds = background_groups["imaginative_surreal"] or [NONE_LABEL]
        lighting_styles = option_map.get("lighting_style") or [NONE_LABEL]
        color_palettes = option_map.get("color_palette") or [NONE_LABEL]
        camera_lenses = option_map.get("camera_lens") or [NONE_LABEL]

        resolved_studio = _choose(studio_background, studio_backgrounds, rng)
        resolved_real = _choose(real_background, real_backgrounds, rng)
        resolved_imag = _choose(imaginative_background, imaginative_backgrounds, rng)
        resolved_theme = _choose_tuple(theme, _with_random_tuple(THEME_CHOICES), rng)
        resolved_time = _choose_tuple(time_of_day, _with_random_tuple(TIME_OF_DAY_CHOICES), rng)
        resolved_weather = _choose_tuple(weather, _with_random_tuple(WEATHER_CHOICES), rng)
        resolved_mood = _choose_tuple(mood, _with_random_tuple(MOOD_CHOICES), rng)
        resolved_scale = _choose_tuple(scale, _with_random_tuple(SCALE_CHOICES), rng)
        resolved_focal = _choose_tuple(focal_element, _with_random_tuple(FOCAL_CHOICES), rng)
        resolved_architecture = _choose_tuple(architecture, _with_random_tuple(ARCHITECTURE_CHOICES), rng)
        resolved_creature = _choose_tuple(creature_presence, _with_random_tuple(CREATURE_CHOICES), rng)
        resolved_lighting = _choose(lighting_style, lighting_styles, rng)
        resolved_palette = _choose(color_palette, color_palettes, rng)
        resolved_lens = _choose(camera_lens, camera_lenses, rng)

        style_meta = prompt_styles.get(prompt_style, prompt_styles.get("SDXL", {}))

        attr_parts = []
        for label, value in (
            ("studio background", resolved_studio),
            ("public location", resolved_real),
            ("imaginative layer", resolved_imag),
            ("theme", resolved_theme),
            ("time of day", resolved_time),
            ("weather", resolved_weather),
            ("mood", resolved_mood),
            ("scale", resolved_scale),
            ("focal element", resolved_focal),
            ("architecture", resolved_architecture),
            ("creature presence", resolved_creature),
            ("lighting", resolved_lighting),
            ("color palette", resolved_palette),
            ("camera lens", resolved_lens),
        ):
            if value:
                attr_parts.append(f"{label}: {value}")

        if custom_notes.strip():
            attr_parts.append(f"custom notes: {custom_notes.strip()}")

        attribute_blob = ", ".join(attr_parts)

        system_prompt = (
            "You are a concept artist prompt engineer specializing in environment-only imagery."
            " Create imaginative backgrounds with zero humans or humanoid figures."
            " Monsters and creatures are allowed only if specified; describe them as part of the setting."
            " Output two paragraphs separated by a newline:"
            " (1) Final background prompt for image generation (single paragraph)."
            " (2) Character styling inspiration derived from the environment (one sentence)."
            " Do not include markdown, bullet points, negative prompts, or meta commentary."
        )

        user_prompt = (
            f"Create a {prompt_style} surreal background without any humans."
            f" Attributes: {attribute_blob}."
            " Keep emphasis on atmosphere, lighting, geography, architecture, and creature presence."
        )

        payload = {
            "model": ollama_model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.85,
            },
        }

        generate_url = ollama_url
        if not generate_url.endswith("/api/generate"):
            generate_url = generate_url.rstrip("/") + "/api/generate"

        try:
            if requests is None:
                return ("[Please install 'requests' library: pip install requests]", "")

            response = requests.post(generate_url, json=payload, timeout=120)
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

            if "\n" in raw:
                bg_prompt, style_hint = raw.split("\n", 1)
            else:
                bg_prompt, style_hint = raw, ""

            return bg_prompt.strip(), style_hint.strip()
        except requests.exceptions.ConnectionError:
            return ("[Ollama server not running. Please start Ollama.]", "")
        except requests.exceptions.Timeout:
            return ("[Ollama request timed out]", "")
        except Exception as exc:
            logging.getLogger(__name__).exception(f"[BackgroundEditNode] Error invoking LLM: {exc}")
            return (f"[Error: {exc}]", "")

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
            logging.getLogger(__name__).exception(f"[BackgroundEditNode] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "BackgroundEditNode": BackgroundEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundEditNode": "Background Edit",
}
