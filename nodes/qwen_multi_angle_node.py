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

import random

from wizdroid_lib.constants import RANDOM_LABEL
from wizdroid_lib.data_files import load_json
from wizdroid_lib.helpers import with_random


# === Option Data ===

_OPTIONS = load_json("qwen_multi_angle_options.json")

AZIMUTH_OPTIONS = _OPTIONS["azimuth"]
ELEVATION_OPTIONS = _OPTIONS["elevation"]
DISTANCE_OPTIONS = _OPTIONS["distance"]
EMOTION_OPTIONS = _OPTIONS["emotion"]
BODYTYPE_OPTIONS = _OPTIONS["bodytype"]
SKIN_TONE_OPTIONS = _OPTIONS["skin_tone"]
EYE_COLOR_OPTIONS = _OPTIONS["eye_color"]
HAIRSTYLE_OPTIONS = _OPTIONS["hairstyle"]
OUTFIT_STYLE_OPTIONS = _OPTIONS["outfit_style"]
MAKEUP_STYLE_OPTIONS = _OPTIONS["makeup_style"]
BACKGROUND_OPTIONS = _OPTIONS["background"]
OUTFIT_TYPE_OPTIONS = _OPTIONS["outfit_type"]


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
                "outfit_style": (with_random(OUTFIT_STYLE_OPTIONS + ["none"]), {"default": "none"}),
                "makeup_style": (with_random(MAKEUP_STYLE_OPTIONS + ["none"]), {"default": "none"}),
                "background": (with_random(BACKGROUND_OPTIONS + ["none"]), {"default": "none"}),
                "outfit_type": (with_random(OUTFIT_TYPE_OPTIONS + ["none"]), {"default": "none"}),
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
        outfit_style: str,
        makeup_style: str,
        background: str,
        outfit_type: str,
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
        resolved_outfit_style = self._resolve(outfit_style, OUTFIT_STYLE_OPTIONS + ["none"], rng)
        resolved_makeup_style = self._resolve(makeup_style, MAKEUP_STYLE_OPTIONS + ["none"], rng)
        resolved_background = self._resolve(background, BACKGROUND_OPTIONS + ["none"], rng)
        resolved_outfit_type = self._resolve(outfit_type, OUTFIT_TYPE_OPTIONS + ["none"], rng)
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
        
        # Add outfit style if not none
        if resolved_outfit_style != "none":
            prompt = f"{prompt}, {resolved_outfit_style} outfit style"
        
        # Add outfit type if not none
        if resolved_outfit_type != "none":
            prompt = f"{prompt}, {resolved_outfit_type} outfit"
        
        # Add makeup style if not none
        if resolved_makeup_style != "none":
            prompt = f"{prompt}, {resolved_makeup_style} makeup"
        
        # Add background if not none
        if resolved_background != "none":
            prompt = f"{prompt}, {resolved_background} background"
        
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
            f"  • Outfit Style: {resolved_outfit_style if resolved_outfit_style != 'none' else 'None'}",
            f"  • Outfit Type: {resolved_outfit_type if resolved_outfit_type != 'none' else 'None'}",
            f"  • Makeup Style: {resolved_makeup_style if resolved_makeup_style != 'none' else 'None'}",
            f"  • Background: {resolved_background if resolved_background != 'none' else 'None'}",
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
