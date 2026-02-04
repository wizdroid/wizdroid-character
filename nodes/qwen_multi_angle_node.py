"""
Qwen Multi-Angle LoRA Prompt Generator

Generates prompts for fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA

Supports all 96 camera positions:
- 8 Azimuths (horizontal rotation)
- 4 Elevations (vertical angle)
- 3 Distances
"""

from typing import Tuple

from wizdroid_lib.constants import RANDOM_LABEL
from wizdroid_lib.helpers import with_random
import random


# === Camera Position Options ===

AZIMUTH_OPTIONS = [
    "front view",              # 0°
    "front-right quarter view",  # 45°
    "right side view",         # 90°
    "back-right quarter view",  # 135°
    "back view",               # 180°
    "back-left quarter view",   # 225°
    "left side view",          # 270°
    "front-left quarter view",  # 315°
]

ELEVATION_OPTIONS = [
    "low-angle shot",   # -30° - Camera below, looking up
    "eye-level shot",   # 0° - Camera at object level
    "elevated shot",    # 30° - Camera slightly above
    "high-angle shot",  # 60° - Camera high, looking down
]

DISTANCE_OPTIONS = [
    "close-up",     # ×0.6 - Details, textures
    "medium shot",  # ×1.0 - Balanced, standard
    "wide shot",    # ×1.8 - Context, environment
]

EMOTION_OPTIONS = [
    "neutral",
    "happy",
    "smiling",
    "laughing",
    "sad",
    "crying",
    "angry",
    "furious",
    "surprised",
    "shocked",
    "fearful",
    "scared",
    "disgusted",
    "confused",
    "thoughtful",
    "pensive",
    "serious",
    "determined",
    "confident",
    "shy",
    "embarrassed",
    "proud",
    "excited",
    "bored",
    "tired",
    "sleepy",
    "relaxed",
    "calm",
    "peaceful",
    "loving",
    "flirty",
    "seductive",
    "mischievous",
    "playful",
    "curious",
    "worried",
    "anxious",
    "hopeful",
    "melancholic",
    "nostalgic",
]

BODYTYPE_OPTIONS = [
  "slim",
  "slim and fit",
  "slender",
  "lithe",
  "lanky",
  "wiry",
  "curvy",
  "slim thick",
  "thick",
  "voluptuous",
  "chubby",
  "plump",
  "buxom",
  "rubenesque",
  "healthy",
  "balanced",
  "natural",
  "compact",
  "athletic",
  "muscular",
  "toned",
  "fit",
  "buff",
  "swole",
  "bodybuilder",
  "petite",
  "average",
  "plus size",
  "heavyset",
  "stocky",
  "stout",
  "portly",
  "obese",
  "morbidly obese",
  "jabba the hutt level obese",
  "pear shaped",
  "hourglass shaped",
  "apple shaped",
  "inverted triangle",
  "rectangle",
  "triangle",
  "oval",
  "round",
  "spoon",
  "diamond",
  "v-shaped",
  "ectomorph",
  "mesomorph",
  "endomorph",
  "frail",
  "delicate",
  "barrel-chested"
]

SKIN_TONE_OPTIONS = [
  "porcelain",
  "ivory",
  "fair",
  "light",
  "pale",
  "alabaster",
  "rosy",
  "beige",
  "cream",
  "honey",
  "light beige",
  "peach",
  "warm ivory",
  "cool fair",
  "neutral light",
  "medium",
  "olive",
  "tan",
  "golden",
  "caramel",
  "wheat",
  "sandy",
  "bronze",
  "warm medium",
  "cool medium",
  "neutral tan",
  "deep",
  "rich",
  "chestnut",
  "mahogany",
  "cocoa",
  "espresso",
  "ebony",
  "dark",
  "deep brown",
  "warm deep",
  "cool deep",
  "neutral dark",
  "fitzpatrick type i",
  "fitzpatrick type ii",
  "fitzpatrick type iii",
  "fitzpatrick type iv",
  "fitzpatrick type v",
  "fitzpatrick type vi",
  "sun-kissed",
  "freckled fair",
  "glowy olive",
  "toffee",
  "mocha",
  "midnight",
  "ashy pale",
  "ruddy",
  "sallow",
  "tawny",
  "umber",
  "sienna",
  "sepia",
  "walnut",
  "onyx",
  "ghostly white (vampire chic)",
  "sunburnt lobster (oops level)",
  "chocolate fondue",
  "velvet night"
]

EYE_COLOR_OPTIONS = [
  "blue",
  "light blue",
  "ice blue",
  "steel blue",
  "turquoise",
  "aqua",
  "green",
  "emerald green",
  "forest green",
  "olive green",
  "hazel",
  "amber",
  "honey",
  "gold",
  "brown",
  "light brown",
  "chestnut brown",
  "dark brown",
  "chocolate brown",
  "black",
  "gray",
  "slate gray",
  "silver",
  "violet",
  "indigo",
  "lavender",
  "red",
  "albino pink",
  "heterochromia (two different colors)",
  "central heterochromia (ring of different color)",
  "sectoral heterochromia (patch of different color)",
  "fitzpatrick blue-gray",
  "stormy gray",
  "mossy green",
  "caramel brown",
  "whiskey amber",
  "midnight blue",
  "sapphire",
  "jade",
  "topaz",
  "onyx black",
  "zombie white (undead chic)",
  "dragon red (fiery myth)",
  "alien glow (neon weird)",
  "vampire crimson (eternal night)"
]

HAIRSTYLE_OPTIONS = [
  "bald",
  "buzz cut",
  "shaved head",
  "crew cut",
  "high and tight",
  "short cropped",
  "pixie cut",
  "bob",
  "lob (long bob)",
  "shoulder length",
  "mid-length straight",
  "long straight",
  "extra long",
  "ponytail",
  "high ponytail",
  "low ponytail",
  "braids",
  "cornrows",
  "box braids",
  "french braid",
  "dutch braid",
  "fishtail braid",
  "bun",
  "top knot",
  "messy bun",
  "space buns",
  "updo",
  "chignon",
  "wavy",
  "beach waves",
  "curly",
  "loose curls",
  "tight coils",
  "afro",
  "twist out",
  "bantu knots",
  "dreadlocks",
  "locs",
  "faux locs",
  "undercut",
  "side shave",
  "mohawk",
  "faux hawk",
  "mullet",
  "shag",
  "layered",
  "feathered",
  "bangs",
  "side-swept bangs",
  "curtain bangs",
  "pompadour",
  "quiff",
  "slicked back",
  "messy bedhead",
  "top fade",
  "low fade",
  "tapered",
  "hi-top fade",
  "bowl cut",
  "asymmetrical",
  "wolf cut",
  "hime cut",
  "samurai topknot",
  "viking braids",
  "elfin spikes",
  "mermaid waves",
  "unicorn mane",
  "wizard beard flow (if applicable)",
  "dragon spikes",
  "zombie tousled"
]


class WizdroidMultiAngleNode:
    """🧙 Generate multi-angle camera prompts for the Qwen Image Edit LoRA."""

    CATEGORY = "🧙 Wizdroid/Prompts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "preview")
    FUNCTION = "generate_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "azimuth": (with_random(AZIMUTH_OPTIONS), {"default": "front view"}),
                "elevation": (with_random(ELEVATION_OPTIONS), {"default": "eye-level shot"}),
                "distance": (with_random(DISTANCE_OPTIONS), {"default": "medium shot"}),
                "bodytype": (with_random(BODYTYPE_OPTIONS + ["none"]), {"default": "none"}),
                "skin_tone": (with_random(SKIN_TONE_OPTIONS + ["none"]), {"default": "none"}),
                "eye_color": (with_random(EYE_COLOR_OPTIONS + ["none"]), {"default": "none"}),
                "hairstyle": (with_random(HAIRSTYLE_OPTIONS + ["none"]), {"default": "none"}),
                "emotion": (with_random(EMOTION_OPTIONS), {"default": "neutral"}),
                "additional_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Additional instructions (appended after camera prompt)"
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            }
        }

    def generate_prompt(
        self,
        azimuth: str,
        elevation: str,
        distance: str,
        bodytype: str,
        skin_tone: str,
        eye_color: str,
        hairstyle: str,
        emotion: str,
        additional_text: str,
        seed: int = 0,
    ) -> Tuple[str, str]:
        """Generate the multi-angle prompt."""
        
        rng = random.Random(seed)
        
        # Resolve random selections
        resolved_azimuth = self._resolve(azimuth, AZIMUTH_OPTIONS, rng)
        resolved_elevation = self._resolve(elevation, ELEVATION_OPTIONS, rng)
        resolved_distance = self._resolve(distance, DISTANCE_OPTIONS, rng)
        resolved_bodytype = self._resolve(bodytype, BODYTYPE_OPTIONS + ["none"], rng)
        resolved_skin_tone = self._resolve(skin_tone, SKIN_TONE_OPTIONS + ["none"], rng)
        resolved_eye_color = self._resolve(eye_color, EYE_COLOR_OPTIONS + ["none"], rng)
        resolved_hairstyle = self._resolve(hairstyle, HAIRSTYLE_OPTIONS + ["none"], rng)
        resolved_emotion = self._resolve(emotion, EMOTION_OPTIONS, rng)
        
        # Build prompt in the required format: <sks> [azimuth] [elevation] [distance]
        prompt = f"<sks> {resolved_azimuth} {resolved_elevation} {resolved_distance}"
        
        # Add bodytype if not none
        if resolved_bodytype != "none":
            prompt = f"{prompt}, {resolved_bodytype} body type"
        
        # Add skin tone if not none
        if resolved_skin_tone != "none":
            prompt = f"{prompt}, {resolved_skin_tone} skin tone"
        
        # Add eye color if not none
        if resolved_eye_color != "none":
            prompt = f"{prompt}, {resolved_eye_color} eyes"
        
        # Add hairstyle if not none
        if resolved_hairstyle != "none":
            prompt = f"{prompt}, {resolved_hairstyle} hairstyle"
        
        # Add emotion
        prompt = f"{prompt}, {resolved_emotion} expression"
        
        # Append additional text if provided
        if additional_text.strip():
            prompt = f"{prompt}, {additional_text.strip()}"
        
        # Build preview with more details
        preview_lines = [
            f"Prompt: {prompt}",
            "",
            "Camera Settings:",
            f"  • Azimuth: {resolved_azimuth}",
            f"  • Elevation: {resolved_elevation}",
            f"  • Distance: {resolved_distance}",
            f"  • Body Type: {resolved_bodytype if resolved_bodytype != 'none' else 'None'}",
            f"  • Skin Tone: {resolved_skin_tone if resolved_skin_tone != 'none' else 'None'}",
            f"  • Eye Color: {resolved_eye_color if resolved_eye_color != 'none' else 'None'}",
            f"  • Hairstyle: {resolved_hairstyle if resolved_hairstyle != 'none' else 'None'}",
            f"  • Emotion: {resolved_emotion}",
        ]
        
        if additional_text.strip():
            preview_lines.extend([
                "",
                f"Additional: {additional_text.strip()}",
            ])
        
        preview_lines.extend([
            "",
            "Tips:",
            "  • LoRA Strength: 0.8 - 1.0 recommended",
            "  • Base Model: Qwen/Qwen-Image-Edit-2511",
        ])
        preview = "\n".join(preview_lines)
        
        return (prompt, preview)

    @staticmethod
    def _resolve(value: str, options: list, rng: random.Random) -> str:
        """Resolve a value, picking randomly if needed."""
        if value == RANDOM_LABEL:
            return rng.choice(options)
        return value


NODE_CLASS_MAPPINGS = {
    "WizdroidMultiAngle": WizdroidMultiAngleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidMultiAngle": "🧙 Wizdroid: Multi-Angle",
}
