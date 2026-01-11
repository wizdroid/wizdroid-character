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


class QwenMultiAngleNode:
    """
    ComfyUI node to generate prompts for the Qwen Image Edit Multiple Angles LoRA.
    
    Produces prompts in the format: <sks> [azimuth] [elevation] [distance]
    
    Supports all 96 camera positions:
    - 8 Azimuths × 4 Elevations × 3 Distances = 96 poses
    """

    CATEGORY = "Wizdroid/character"
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
        resolved_emotion = self._resolve(emotion, EMOTION_OPTIONS, rng)
        
        # Build prompt in the required format: <sks> [azimuth] [elevation] [distance]
        prompt = f"<sks> {resolved_azimuth} {resolved_elevation} {resolved_distance}"
        
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
    "WizdroidMultiAngle": QwenMultiAngleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidMultiAngle": "Multi-Angle",
}
