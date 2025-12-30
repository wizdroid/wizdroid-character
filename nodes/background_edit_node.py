import random
import logging
from typing import Any, Dict, List, Optional, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json
from wizdroid_lib.helpers import choose, choose_tuple, get_background_groups, with_random, with_random_tuple
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_text

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


class BackgroundEditNode:
    CATEGORY = "Wizdroid/backgrounds"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("background_prompt", "style_inspiration")
    FUNCTION = "generate_background_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        option_map = load_json("character_options.json")
        prompt_styles = load_json("prompt_styles.json")
        background_groups = get_background_groups(option_map.get("background_style"))
        studio_backgrounds = background_groups["studio_controlled"] or [NONE_LABEL]
        real_backgrounds = background_groups["public_exotic_real"] or [NONE_LABEL]
        imaginative_backgrounds = background_groups["imaginative_surreal"] or [NONE_LABEL]
        color_palettes = option_map.get("color_palette") or [NONE_LABEL]
        lighting_styles = option_map.get("lighting_style") or [NONE_LABEL]
        camera_lenses = option_map.get("camera_lens") or [NONE_LABEL]

        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "prompt_style": (tuple(prompt_styles.keys()), {"default": "SDXL"}),
                "studio_background": (with_random(studio_backgrounds), {"default": RANDOM_LABEL}),
                "real_background": (with_random(real_backgrounds), {"default": NONE_LABEL}),
                "imaginative_background": (with_random(imaginative_backgrounds), {"default": NONE_LABEL}),
                "theme": (with_random_tuple(THEME_CHOICES), {"default": RANDOM_LABEL}),
                "time_of_day": (with_random_tuple(TIME_OF_DAY_CHOICES), {"default": RANDOM_LABEL}),
                "weather": (with_random_tuple(WEATHER_CHOICES), {"default": RANDOM_LABEL}),
                "mood": (with_random_tuple(MOOD_CHOICES), {"default": RANDOM_LABEL}),
                "scale": (with_random_tuple(SCALE_CHOICES), {"default": RANDOM_LABEL}),
                "focal_element": (with_random_tuple(FOCAL_CHOICES), {"default": RANDOM_LABEL}),
                "architecture": (with_random_tuple(ARCHITECTURE_CHOICES), {"default": RANDOM_LABEL}),
                "creature_presence": (with_random_tuple(CREATURE_CHOICES), {"default": RANDOM_LABEL}),
                "lighting_style": (with_random(lighting_styles), {"default": RANDOM_LABEL}),
                "color_palette": (with_random(color_palettes), {"default": RANDOM_LABEL}),
                "camera_lens": (with_random(camera_lenses), {"default": RANDOM_LABEL}),
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
        content_rating: str,
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
        option_map = load_json("character_options.json")
        prompt_styles = load_json("prompt_styles.json")
        rng = random.Random(seed)

        background_groups = get_background_groups(option_map.get("background_style"))
        studio_backgrounds = background_groups["studio_controlled"] or [NONE_LABEL]
        real_backgrounds = background_groups["public_exotic_real"] or [NONE_LABEL]
        imaginative_backgrounds = background_groups["imaginative_surreal"] or [NONE_LABEL]
        lighting_styles = option_map.get("lighting_style") or [NONE_LABEL]
        color_palettes = option_map.get("color_palette") or [NONE_LABEL]
        camera_lenses = option_map.get("camera_lens") or [NONE_LABEL]

        resolved_studio = choose(studio_background, studio_backgrounds, rng)
        resolved_real = choose(real_background, real_backgrounds, rng)
        resolved_imag = choose(imaginative_background, imaginative_backgrounds, rng)
        resolved_theme = choose_tuple(theme, with_random_tuple(THEME_CHOICES), rng)
        resolved_time = choose_tuple(time_of_day, with_random_tuple(TIME_OF_DAY_CHOICES), rng)
        resolved_weather = choose_tuple(weather, with_random_tuple(WEATHER_CHOICES), rng)
        resolved_mood = choose_tuple(mood, with_random_tuple(MOOD_CHOICES), rng)
        resolved_scale = choose_tuple(scale, with_random_tuple(SCALE_CHOICES), rng)
        resolved_focal = choose_tuple(focal_element, with_random_tuple(FOCAL_CHOICES), rng)
        resolved_architecture = choose_tuple(architecture, with_random_tuple(ARCHITECTURE_CHOICES), rng)
        resolved_creature = choose_tuple(creature_presence, with_random_tuple(CREATURE_CHOICES), rng)
        resolved_lighting = choose(lighting_style, lighting_styles, rng)
        resolved_palette = choose(color_palette, color_palettes, rng)
        resolved_lens = choose(camera_lens, camera_lenses, rng)

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

        system_prompt = load_system_prompt_text("system_prompts/background_edit_system.txt", content_rating)

        user_prompt = (
            f"Create a {prompt_style} surreal background without any humans."
            f" Attributes: {attribute_blob}."
            " Keep emphasis on atmosphere, lighting, geography, architecture, and creature presence."
        )

        ok, raw = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={
                "temperature": 0.85,
                "num_predict": 512,
            },
            timeout=120,
        )

        if not ok:
            return (f"[Error: {raw}]", "")

        if "\n" in raw:
            bg_prompt, style_hint = raw.split("\n", 1)
        else:
            bg_prompt, style_hint = raw, ""

        if content_rating == "SFW":
            err = enforce_sfw(bg_prompt)
            if err:
                return ("[Blocked: potential NSFW content detected. Switch content_rating to 'Mixed' or 'NSFW'.]", "")

        return bg_prompt.strip(), style_hint.strip()


NODE_CLASS_MAPPINGS = {
    "BackgroundEditNode": BackgroundEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundEditNode": "Background Edit",
}
