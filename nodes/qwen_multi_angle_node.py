"""
Character Edit Prompt Generator (Merged)

Generates prompts for image editing models with multiple reference images:
- Subject reference (face/identity)
- Clothing reference
- Pose reference
- Background/lighting reference
- Style reference

Supports camera positioning for multi-angle editing:
- 8 Azimuths (horizontal rotation)
- 4 Elevations (vertical angle)
- 8 Distances

Merged from character_edit_node.py + qwen_multi_angle_node.py
"""

from typing import Tuple

import json
import random
import logging

from wizdroid_lib.constants import RANDOM_LABEL
from wizdroid_lib.data_files import load_shared, filter_by_gender
from wizdroid_lib.helpers import normalize_option_list, with_random
from wizdroid_lib.registry import DataRegistry

logger = logging.getLogger(__name__)

# Reference image type options
REF_IMAGE_TYPE_OPTIONS = ["none", "clothing", "pose", "background", "lighting", "style"]

# Prompt templates for each reference type
REF_PROMPT_TEMPLATES = {
    "clothing": (
        "Use image {index} STRICTLY as clothing reference — apply this exact outfit with perfect "
        "fabric details, texture, fit, color accuracy, and all accessories/jewelry. No creative "
        "changes to the garment."
    ),
    "pose": (
        "Use image {index} as pose reference — copy the exact body position, limb placement, "
        "camera angle, framing, and spatial arrangement precisely."
    ),
    "background": (
        "Use image {index} as background reference — match the environment, setting, depth, "
        "and atmospheric details exactly."
    ),
    "lighting": (
        "Use image {index} as lighting reference — replicate the exact lighting setup, shadow "
        "direction, highlight placement, color temperature, and overall illumination."
    ),
    "style": (
        "Use image {index} as style reference — replicate its artistic style, color palette, "
        "rendering technique, tonal range, texture quality, and visual aesthetics exactly."
    ),
}


class WizdroidCharacterEditNode:
    """🧙 Generate character edit prompts with multiple reference images for image editing models."""

    CATEGORY = "🧙 Wizdroid/Prompts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "preview")
    FUNCTION = "generate_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        cam = DataRegistry.get_camera_lighting() or {}
        fashion_data = DataRegistry.get_fashion() or {}
        emotions_data = DataRegistry.get_emotions() or {}
        body_data = DataRegistry.get_body_types() or {}
        skin_data = DataRegistry.get_skin_tones() or {}
        eye_data = DataRegistry.get_eye_colors() or {}
        hair_data = DataRegistry.get_hair() or {}
        makeup_data = DataRegistry.get_makeup() or {}
        bg_data = DataRegistry.get_backgrounds() or {}
        poses_data = DataRegistry.get_poses() or {}

        emotions = emotions_data.get("emotions", [])
        body_types = normalize_option_list(body_data.get("body_types", []))
        skin_tones = skin_data.get("skin_tones", [])
        eye_colors = eye_data.get("eye_colors", [])
        hair_styles = hair_data.get("hair_styles", {}).get("any", [])
        makeup_styles = normalize_option_list(makeup_data.get("makeup_styles", []))
        backgrounds = (
            bg_data.get("backgrounds", {}).get("studio_controlled", [])
            + bg_data.get("backgrounds", {}).get("public_exotic_real", [])
            + bg_data.get("backgrounds", {}).get("imaginative_surreal", [])
        )
        outfit_names = normalize_option_list(fashion_data.get("fashion_outfits", []))
        style_names = normalize_option_list(fashion_data.get("fashion_styles", []))
        pose_all = (
            poses_data.get("pose_styles", {}).get("sfw", {}).get("any", [])
            + poses_data.get("pose_styles", {}).get("sfw", {}).get("female", [])
            + poses_data.get("pose_styles", {}).get("sfw", {}).get("male", [])
        )

        azimuth = cam.get("azimuth", ["front view"])
        elevation = cam.get("elevation", ["eye-level shot"])
        distance = cam.get("distance", ["medium shot"])

        return {
            "required": {
                "retain_face": (["enabled", "disabled"], {"default": "enabled"}),
                "gender": (["none", "male", "female"], {"default": "none"}),
                "azimuth": (with_random(azimuth), {"default": "front view"}),
                "elevation": (with_random(elevation), {"default": "eye-level shot"}),
                "distance": (with_random(distance), {"default": "medium shot"}),
                "bodytype": (with_random(body_types + ["none"]), {"default": "none"}),
                "skin_tone": (with_random(skin_tones + ["none"]), {"default": "none"}),
                "eye_color": (with_random(eye_colors + ["none"]), {"default": "none"}),
                "hairstyle": (with_random(hair_styles + ["none"]), {"default": "none"}),
                "outfit_style": (with_random(style_names + ["none"]), {"default": "none"}),
                "makeup_style": (with_random(makeup_styles + ["none"]), {"default": "none"}),
                "background": (with_random(backgrounds + ["none"]), {"default": "none"}),
                "outfit_type": (with_random(outfit_names + ["none"]), {"default": "none"}),
                "emotion": (with_random(emotions), {"default": "neutral"}),
                "pose_style": (with_random(pose_all + ["none"]), {"default": "none"}),
                "input_image_index": ("STRING", {
                    "default": "1",
                    "placeholder": "1-9"
                }),
                "ref_images": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Dynamic reference config (managed by UI)"
                }),
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
        retain_face: str,
        gender: str,
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
        pose_style: str,
        input_image_index: str,
        ref_images: str,
        additional_text: str,
        seed: int = 0,
    ) -> Tuple[str, str]:
        """Generate the multi-angle prompt."""
        cam = DataRegistry.get_camera_lighting() or {}
        fashion_data = DataRegistry.get_fashion() or {}
        emotions_data = DataRegistry.get_emotions() or {}
        body_data = DataRegistry.get_body_types() or {}
        skin_data = DataRegistry.get_skin_tones() or {}
        eye_data = DataRegistry.get_eye_colors() or {}
        hair_data = DataRegistry.get_hair() or {}
        makeup_data = DataRegistry.get_makeup() or {}
        bg_data = DataRegistry.get_backgrounds() or {}
        poses_data = DataRegistry.get_poses() or {}

        rng = random.Random(seed)
        
        azimuth_opts = cam.get("azimuth", ["front view"])
        elevation_opts = cam.get("elevation", ["eye-level shot"])
        distance_opts = cam.get("distance", ["medium shot"])
        emotion_opts = emotions_data.get("emotions", [])
        body_opts = normalize_option_list(
            filter_by_gender(body_data.get("body_types", []), gender)
        )
        skin_opts = skin_data.get("skin_tones", [])
        eye_opts = eye_data.get("eye_colors", [])
        hair_opts = hair_data.get("hair_styles", {}).get("any", [])
        if gender and gender != "none":
            hair_opts = hair_opts + hair_data.get("hair_styles", {}).get(gender, [])
        makeup_opts = normalize_option_list(
            filter_by_gender(makeup_data.get("makeup_styles", []), gender)
        )
        outfit_names = normalize_option_list(
            filter_by_gender(fashion_data.get("fashion_outfits", []), gender)
        )
        style_names = normalize_option_list(
            filter_by_gender(fashion_data.get("fashion_styles", []), gender)
        )
        bg_all = (
            bg_data.get("backgrounds", {}).get("studio_controlled", [])
            + bg_data.get("backgrounds", {}).get("public_exotic_real", [])
            + bg_data.get("backgrounds", {}).get("imaginative_surreal", [])
        )

        # Resolve random selections
        resolved_azimuth = self._resolve(azimuth, azimuth_opts, rng)
        resolved_elevation = self._resolve(elevation, elevation_opts, rng)
        resolved_distance = self._resolve(distance, distance_opts, rng)
        resolved_bodytype = self._resolve(bodytype, body_opts + ["none"], rng)
        resolved_skin_tone = self._resolve(skin_tone, skin_opts + ["none"], rng)
        resolved_eye_color = self._resolve(eye_color, eye_opts + ["none"], rng)
        resolved_hairstyle = self._resolve(hairstyle, hair_opts + ["none"], rng)
        resolved_outfit_style = self._resolve(outfit_style, style_names + ["none"], rng)
        resolved_makeup_style = self._resolve(makeup_style, makeup_opts + ["none"], rng)
        resolved_background = self._resolve(background, bg_all + ["none"], rng)
        resolved_outfit_type = self._resolve(outfit_type, outfit_names + ["none"], rng)
        resolved_emotion = self._resolve(emotion, emotion_opts, rng)
        pose_opts = poses_data.get("pose_styles", {}).get("sfw", {}).get("any", [])
        if gender and gender.lower() in ("female", "male"):
            pose_opts = pose_opts + poses_data.get("pose_styles", {}).get("sfw", {}).get(gender.lower(), [])
        else:
            pose_opts = (
                pose_opts
                + poses_data.get("pose_styles", {}).get("sfw", {}).get("female", [])
                + poses_data.get("pose_styles", {}).get("sfw", {}).get("male", [])
            )
        resolved_pose_style = self._resolve(pose_style, pose_opts + ["none"], rng)
        resolved_input_index = self._resolve_input_index(input_image_index)
        
        # Build natural language prompt (multi-paragraph, high-fashion quality)
        prompt_parts = []
        
        # Build reference image prompts
        ref_prompts = []
        
        # Add subject reference (always first, uses input_image_index)
        if retain_face == "enabled":
            ref_prompts.append(
                f"Use image {resolved_input_index} as subject reference for face and identity — "
                f"preserve the exact facial features, skin tone, eye shape, and hair with zero "
                f"changes to identity."
            )
        
        # Parse dynamic reference images
        ref_assignments = self._parse_ref_images(ref_images)
        
        for img_index, ref_type in ref_assignments:
            if ref_type != "none" and ref_type in REF_PROMPT_TEMPLATES:
                ref_prompts.append(REF_PROMPT_TEMPLATES[ref_type].format(index=img_index))
        
        if ref_prompts:
            ref_block = "\n".join(ref_prompts)
            prompt_parts.append(ref_block.rstrip("."))
        
        # Camera/shot specification
        camera_parts = [p for p in [resolved_azimuth, resolved_elevation, resolved_distance] if p != "none"]
        if camera_parts:
            prompt_parts.append(f"<sks> {', '.join(camera_parts)}")
        else:
            prompt_parts.append("<sks>")
        
        # === PARAGRAPH 1: IDENTITY / PHYSICAL DESCRIPTION ===
        identity_parts = []
        
        if gender == "male":
            identity_parts.append("A striking male")
        elif gender == "female":
            identity_parts.append("A stunning female")
        else:
            identity_parts.append("A stunning person")
        
        physical_traits = []
        if resolved_bodytype != "none":
            physical_traits.append(f"{resolved_bodytype} build")
        if resolved_skin_tone != "none":
            physical_traits.append(f"{resolved_skin_tone} skin")
        if resolved_eye_color != "none":
            physical_traits.append(f"{resolved_eye_color} eyes")
        if resolved_hairstyle != "none":
            physical_traits.append(f"{resolved_hairstyle} hair")
        
        if physical_traits:
            identity_parts.append(f"with {', '.join(physical_traits)}")
        
        prompt_parts.append(" ".join(identity_parts))
        
        # === PARAGRAPH 2: OUTFIT / CLOTHING ===
        outfit_parts = []
        if resolved_outfit_type != "none":
            outfit_parts.append(f"Wearing {resolved_outfit_type}")
        elif resolved_outfit_style != "none":
            outfit_parts.append(f"Dressed in a {resolved_outfit_style} style ensemble")
        
        clothing_ref_idx = None
        for img_index, ref_type in ref_assignments:
            if ref_type == "clothing":
                clothing_ref_idx = img_index
                break
        
        if clothing_ref_idx is not None:
            if outfit_parts:
                outfit_parts.append(
                    f"with perfect fabric details, fit, and accessories exactly matching "
                    f"image {clothing_ref_idx}"
                )
            else:
                outfit_parts.append(
                    f"Wearing the exact outfit from image {clothing_ref_idx} with perfect "
                    f"fabric details, texture, fit, color accuracy, and all accessories"
                )
        
        if outfit_parts:
            prompt_parts.append(". ".join(outfit_parts))
        
        # === PARAGRAPH 3: BEAUTY / MAKEUP & EXPRESSION ===
        beauty_parts = []
        
        if resolved_makeup_style != "none" and resolved_makeup_style != "no makeup natural":
            beauty_parts.append(f"Wearing {resolved_makeup_style} makeup")
        elif resolved_makeup_style == "no makeup natural":
            beauty_parts.append("Natural look, no makeup")
        
        # Expression
        if resolved_emotion != "neutral":
            beauty_parts.append(
                f"Expressing a {resolved_emotion} expression, captivating and natural"
            )
        else:
            beauty_parts.append(
                "Sophisticated and captivating expression"
            )
        
        prompt_parts.append(". ".join(beauty_parts))
        
        # === PARAGRAPH 4: POSE, STYLE & SETTING ===
        scene_parts = []
        
        if resolved_pose_style != "none":
            scene_parts.append(f"Striking a {resolved_pose_style}")
        
        if resolved_background != "none":
            scene_parts.append(f"set against a {resolved_background} setting")
        
        if scene_parts:
            prompt_parts.append(", ".join(scene_parts))
        
        prompt = ". ".join(prompt_parts)
        
        if additional_text.strip():
            prompt = f"{prompt}\n\n{additional_text.strip()}"
        
        # Build preview
        preview_lines = [
            f"Prompt: {prompt}",
            "",
            "Settings:",
            f"  • Retain Face: {retain_face}",
            f"  • Gender: {gender if gender != 'none' else 'None'}",
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
            f"  • Pose Style: {resolved_pose_style if resolved_pose_style != 'none' else 'None'}",
        ]

        if retain_face == "enabled":
            preview_lines.append(f"  • Subject Image: {resolved_input_index}")
        
        ref_preview_lines = []
        for img_index, ref_type in ref_assignments:
            if ref_type != "none":
                ref_preview_lines.append(f"  • Image {img_index}: {ref_type}")
        
        if ref_preview_lines:
            preview_lines.extend(["", "Reference Images:"] + ref_preview_lines)
        
        if additional_text.strip():
            preview_lines.extend(["", f"Additional: {additional_text.strip()}"])
        
        preview = "\n".join(preview_lines)
        
        return (prompt, preview)

    @staticmethod
    def _resolve(value: str, options: list, rng: random.Random) -> str:
        """Resolve a value, picking randomly if needed."""
        if value == RANDOM_LABEL:
            pool = [o for o in options if o != "none"]
            if pool:
                return rng.choice(pool)
            return "none"
        return value

    @staticmethod
    def _resolve_input_index(value: str) -> int:
        """Coerce input image index to an int in the range 1-9, defaulting to 1."""
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            return 1

        if parsed < 1 or parsed > 9:
            return 1

        return parsed

    @staticmethod
    def _parse_ref_images(ref_images_str: str) -> list:
        """Parse reference images string into list of (index, type) tuples.
        
        Supports formats:
        - Simple: "2:clothing,3:pose,4:background"
        - JSON: [{"index":2,"type":"clothing"},...]
        """
        if not ref_images_str or not ref_images_str.strip():
            return []
        
        ref_images_str = ref_images_str.strip()
        result = []
        
        # Try JSON format first
        if ref_images_str.startswith('['):
            try:
                data = json.loads(ref_images_str)
                for item in data:
                    if isinstance(item, dict) and 'index' in item and 'type' in item:
                        idx = int(item['index'])
                        ref_type = str(item['type']).lower()
                        if ref_type in REF_IMAGE_TYPE_OPTIONS and ref_type != 'none':
                            result.append((idx, ref_type))
                return result
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # Try simple format: "2:clothing,3:pose"
        try:
            for part in ref_images_str.split(','):
                part = part.strip()
                if ':' in part:
                    idx_str, ref_type = part.split(':', 1)
                    idx = int(idx_str.strip())
                    ref_type = ref_type.strip().lower()
                    if ref_type in REF_IMAGE_TYPE_OPTIONS and ref_type != 'none':
                        result.append((idx, ref_type))
        except (ValueError, AttributeError):
            pass
        
        return result


NODE_CLASS_MAPPINGS = {
    "WizdroidCharacterEdit": WizdroidCharacterEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidCharacterEdit": "🧙 Wizdroid: Character Edit",
}
