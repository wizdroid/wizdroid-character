from typing import Dict, List

from wizdroid_lib.constants import DEFAULT_OLLAMA_URL

from .video_scene_expander_node import VIDEO_MODELS, _NEGATIVE_PROMPTS

# === Model-specific base negatives ===

_BASE_NEGATIVES: Dict[str, str] = {
    "WAN-T2V": "blurry, low quality, watermark, text overlay, static, flat motion, jittery, inconsistent lighting, distorted faces",
    "WAN-I2V": "blurry, low quality, watermark, static image, no motion, jittery, distorted",
    "LTX-T2V": "worst quality, inconsistent motion, blurry, jittery, distorted, floating limbs, temporal flickering",
    "LTX-I2V": "worst quality, inconsistent motion, blurry, jittery, distorted, static image, no movement",
}

# === Artifact type tokens ===

_ARTIFACT_TOKENS: Dict[str, str] = {
    "motion_blur":    "motion blur, smeared frames",
    "temporal_flicker": "temporal flickering, frame inconsistency",
    "face_distortion": "distorted face, deformed eyes, warped features",
    "body_artifacts": "floating limbs, extra fingers, merged body parts, anatomical errors",
    "compression":    "compression artifacts, pixelation, mosaic artifacts",
    "overexposure":   "overexposed, blown-out highlights, washed out",
    "color_banding":  "color banding, posterization, unnatural palette",
    "watermark_text": "watermark, text overlay, subtitles, logo",
    "shake_noise":    "camera shake, grain noise, visual noise",
    "low_framerate":  "choppy motion, low framerate, stuttering",
}


class WizdroidVideoNegativePromptNode:
    """🎬 Build model-specific negative prompts for video generation."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative_prompt",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_model": (VIDEO_MODELS, {"default": "WAN-T2V"}),
                "include_base": ("BOOLEAN", {"default": True, "label_on": "Include model defaults", "label_off": "Custom only"}),
                "motion_blur": ("BOOLEAN", {"default": False}),
                "temporal_flicker": ("BOOLEAN", {"default": True}),
                "face_distortion": ("BOOLEAN", {"default": True}),
                "body_artifacts": ("BOOLEAN", {"default": True}),
                "compression": ("BOOLEAN", {"default": False}),
                "overexposure": ("BOOLEAN", {"default": False}),
                "color_banding": ("BOOLEAN", {"default": False}),
                "watermark_text": ("BOOLEAN", {"default": True}),
                "shake_noise": ("BOOLEAN", {"default": False}),
                "low_framerate": ("BOOLEAN", {"default": False}),
                "custom_additions": ("STRING", {"multiline": False, "default": "", "placeholder": "Any extra negative terms, comma-separated"}),
            }
        }

    def build(
        self,
        target_model: str,
        include_base: bool,
        motion_blur: bool,
        temporal_flicker: bool,
        face_distortion: bool,
        body_artifacts: bool,
        compression: bool,
        overexposure: bool,
        color_banding: bool,
        watermark_text: bool,
        shake_noise: bool,
        low_framerate: bool,
        custom_additions: str,
    ) -> tuple:
        parts: List[str] = []

        if include_base:
            base = _BASE_NEGATIVES.get(target_model, "blurry, low quality, inconsistent motion")
            parts.append(base)

        flags = {
            "motion_blur": motion_blur,
            "temporal_flicker": temporal_flicker,
            "face_distortion": face_distortion,
            "body_artifacts": body_artifacts,
            "compression": compression,
            "overexposure": overexposure,
            "color_banding": color_banding,
            "watermark_text": watermark_text,
            "shake_noise": shake_noise,
            "low_framerate": low_framerate,
        }

        for key, enabled in flags.items():
            if enabled:
                token = _ARTIFACT_TOKENS.get(key, "")
                if token:
                    # Only add terms not already present in the base
                    existing = parts[0] if parts else ""
                    new_terms = [t.strip() for t in token.split(",") if t.strip() not in existing]
                    if new_terms:
                        parts.append(", ".join(new_terms))

        if custom_additions.strip():
            parts.append(custom_additions.strip())

        negative = ", ".join(parts)
        return (negative,)


NODE_CLASS_MAPPINGS = {"WizdroidVideoNegativePrompt": WizdroidVideoNegativePromptNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidVideoNegativePrompt": "🎬 Video Negative Prompt"}
