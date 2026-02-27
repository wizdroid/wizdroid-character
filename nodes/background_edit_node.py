import random
import logging
from typing import Any, Dict, List, Optional, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json, load_shared
from wizdroid_lib.helpers import choose, choose_tuple, with_random, with_random_tuple
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.registry import DataRegistry
from wizdroid_lib.system_prompts import load_system_prompt_text


class WizdroidBackgroundNode:
    """🧙 Generate surreal background prompts using Ollama LLM."""
    
    CATEGORY = "🧙 Wizdroid/Prompts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("background_prompt", "style_inspiration")
    FUNCTION = "generate_background_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        prompt_styles = DataRegistry.get_prompt_styles() or {}
        bg_data = DataRegistry.get_backgrounds() or {}
        cam_data = DataRegistry.get_camera_lighting() or {}
        bg_edit_data = DataRegistry.get_background_edit() or {}

        backgrounds = bg_data.get("backgrounds", {})
        studio_backgrounds = backgrounds.get("studio_controlled", [NONE_LABEL])
        real_backgrounds = backgrounds.get("public_exotic_real", [NONE_LABEL])
        imaginative_backgrounds = backgrounds.get("imaginative_surreal", [NONE_LABEL])
        color_palettes = [
            item.get("name", item) if isinstance(item, dict) else item
            for item in cam_data.get("color_palettes", [NONE_LABEL])
        ]
        lighting_styles = [
            item.get("name", item) if isinstance(item, dict) else item
            for item in cam_data.get("lighting_styles", [NONE_LABEL])
        ]
        camera_lenses = [
            item.get("name", item) if isinstance(item, dict) else item
            for item in cam_data.get("camera_lenses", [NONE_LABEL])
        ]

        bg_edit = bg_edit_data.get("background_edit", {})
        themes = tuple(bg_edit.get("themes", ["astral myth"]))
        time_of_day = tuple(bg_edit.get("time_of_day", ["dawn"]))
        weather = tuple(bg_edit.get("weather", ["clear and crystalline"]))
        moods = tuple(bg_edit.get("moods", ["serene"]))
        scales = tuple(bg_edit.get("scales", ["sweeping vista"]))
        creatures = tuple(bg_edit.get("creatures", ["none"]))
        architecture = tuple(bg_edit.get("architecture", ["ancient ruins"]))
        focal_elements = tuple(bg_edit.get("focal_elements", ["sacred tree"]))

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
                "theme": (with_random_tuple(themes), {"default": RANDOM_LABEL}),
                "time_of_day": (with_random_tuple(time_of_day), {"default": RANDOM_LABEL}),
                "weather": (with_random_tuple(weather), {"default": RANDOM_LABEL}),
                "mood": (with_random_tuple(moods), {"default": RANDOM_LABEL}),
                "scale": (with_random_tuple(scales), {"default": RANDOM_LABEL}),
                "focal_element": (with_random_tuple(focal_elements), {"default": RANDOM_LABEL}),
                "architecture": (with_random_tuple(architecture), {"default": RANDOM_LABEL}),
                "creature_presence": (with_random_tuple(creatures), {"default": RANDOM_LABEL}),
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
        prompt_styles = DataRegistry.get_prompt_styles() or {}
        bg_data = DataRegistry.get_backgrounds() or {}
        cam_data = DataRegistry.get_camera_lighting() or {}
        bg_edit_data = DataRegistry.get_background_edit() or {}

        rng = random.Random(seed)

        backgrounds = bg_data.get("backgrounds", {})
        studio_backgrounds = backgrounds.get("studio_controlled", [NONE_LABEL])
        real_backgrounds = backgrounds.get("public_exotic_real", [NONE_LABEL])
        imaginative_backgrounds = backgrounds.get("imaginative_surreal", [NONE_LABEL])
        lighting_styles = [
            item.get("name", item) if isinstance(item, dict) else item
            for item in cam_data.get("lighting_styles", [NONE_LABEL])
        ]
        color_palettes = [
            item.get("name", item) if isinstance(item, dict) else item
            for item in cam_data.get("color_palettes", [NONE_LABEL])
        ]
        camera_lenses = [
            item.get("name", item) if isinstance(item, dict) else item
            for item in cam_data.get("camera_lenses", [NONE_LABEL])
        ]

        bg_edit = bg_edit_data.get("background_edit", {})
        themes = tuple(bg_edit.get("themes", ["astral myth"]))
        time_of_day_opts = tuple(bg_edit.get("time_of_day", ["dawn"]))
        weather_opts = tuple(bg_edit.get("weather", ["clear and crystalline"]))
        mood_opts = tuple(bg_edit.get("moods", ["serene"]))
        scale_opts = tuple(bg_edit.get("scales", ["sweeping vista"]))
        creature_opts = tuple(bg_edit.get("creatures", ["none"]))
        arch_opts = tuple(bg_edit.get("architecture", ["ancient ruins"]))
        focal_opts = tuple(bg_edit.get("focal_elements", ["sacred tree"]))

        resolved_studio = choose(studio_background, studio_backgrounds, rng, seed)
        resolved_real = choose(real_background, real_backgrounds, rng, seed)
        resolved_imag = choose(imaginative_background, imaginative_backgrounds, rng, seed)
        resolved_theme = choose_tuple(theme, with_random_tuple(themes), rng)
        resolved_time = choose_tuple(time_of_day, with_random_tuple(time_of_day_opts), rng)
        resolved_weather = choose_tuple(weather, with_random_tuple(weather_opts), rng)
        resolved_mood = choose_tuple(mood, with_random_tuple(mood_opts), rng)
        resolved_scale = choose_tuple(scale, with_random_tuple(scale_opts), rng)
        resolved_focal = choose_tuple(focal_element, with_random_tuple(focal_opts), rng)
        resolved_architecture = choose_tuple(architecture, with_random_tuple(arch_opts), rng)
        resolved_creature = choose_tuple(creature_presence, with_random_tuple(creature_opts), rng)
        resolved_lighting = choose(lighting_style, lighting_styles, rng, seed)
        resolved_palette = choose(color_palette, color_palettes, rng, seed)
        resolved_lens = choose(camera_lens, camera_lenses, rng, seed)

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
    "WizdroidBackground": WizdroidBackgroundNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidBackground": "🧙 Wizdroid: Background",
}
