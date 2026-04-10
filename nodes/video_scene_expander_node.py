import json
import hashlib
from typing import Dict, Optional, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

# === Constants ===

VIDEO_MODELS = ("WAN-T2V", "WAN-I2V", "LTX-T2V", "LTX-I2V")
MOTION_INTENSITIES = ("calm", "moderate", "dynamic", "explosive")

_MODEL_STYLE_RULES: Dict[str, str] = {
    "WAN-T2V": (
        "Target model: WAN 2.2 Text-to-Video. "
        "Write cinematic prose (80-200 words). Lead with subject appearance, then describe motion with physical "
        "precision (weight, speed, direction). Include how the environment evolves over the clip. "
        "Specify lighting quality (golden hour, neon rim, soft studio diffused). End with camera movement and shot type."
    ),
    "WAN-I2V": (
        "Target model: WAN 2.2 Image-to-Video. "
        "Write cinematic prose (60-150 words). Begin with visual style or mood. Describe what the existing "
        "subject does: motion, expression changes, new interactions. Add environmental atmosphere (weather, "
        "light shift). Close with camera shot type and movement."
    ),
    "LTX-T2V": (
        "Target model: LTX-Video Text-to-Video. "
        "Write a single chronological paragraph, under 200 words. Start DIRECTLY with the main action (no preamble). "
        "Follow this strict order: (1) action, (2) movements/gestures, (3) character/object appearances, "
        "(4) background/environment, (5) camera angle/movement, (6) lighting/colors, (7) any changes during the clip. "
        "Be literal and precise. Think like a cinematographer writing a shot list."
    ),
    "LTX-I2V": (
        "Target model: LTX-Video Image-to-Video. "
        "Write a single chronological paragraph, under 200 words. Start DIRECTLY with what the existing subject "
        "begins doing. Specify exact movements, gestures, gaze direction, body shifts. Add environmental and "
        "ambient changes. State whether the camera holds or moves — be specific. Describe light and color throughout."
    ),
}

_NEGATIVE_PROMPTS: Dict[str, str] = {
    "WAN-T2V": "blurry, low quality, watermark, text overlay, static, flat motion, jittery, inconsistent lighting, distorted faces",
    "WAN-I2V": "blurry, low quality, watermark, static image, no motion, jittery, distorted",
    "LTX-T2V": "worst quality, inconsistent motion, blurry, jittery, distorted, floating limbs, temporal flickering",
    "LTX-I2V": "worst quality, inconsistent motion, blurry, jittery, distorted, static image, no movement",
}

# === Caching ===

_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 50


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _clean_output(text: str) -> str:
    text = text.strip()
    for prefix in ["here is", "here's", "this is", "sure,", "prompt:", "video prompt:", "a video of", "in this video"]:
        if text.lower().startswith(prefix):
            remainder = text[len(prefix):]
            colon_idx = remainder.find(":")
            if colon_idx != -1 and colon_idx < 20:
                text = remainder[colon_idx + 1:].strip()
            else:
                text = remainder.strip()
            break
    return text


# === Node ===

class WizdroidVideoSceneExpanderNode:
    """🎬 Expand short scene ideas into full video prompts for WAN 2.2 and LTX-Video."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_prompt", "negative_prompt", "preview")
    FUNCTION = "expand"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)
        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "target_model": (VIDEO_MODELS, {"default": "WAN-T2V"}),
                "user_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe your scene idea, e.g. 'woman running through a neon city at night'"}),
                "duration_seconds": ("INT", {"default": 5, "min": 2, "max": 60, "step": 1}),
                "motion_intensity": (MOTION_INTENSITIES, {"default": "moderate"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 250, "min": 80, "max": 500, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def expand(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        target_model: str,
        user_text: str,
        duration_seconds: int,
        motion_intensity: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ):
        global _CACHE

        selections = {
            "text": user_text.strip(),
            "model": target_model,
            "duration": duration_seconds,
            "intensity": motion_intensity,
            "temp": temperature,
            "max_tokens": max_tokens,
            "content_rating": content_rating,
            "seed": seed,
        }

        cache_key = _cache_key(selections)
        if cache_key in _CACHE:
            prompt = _CACHE[cache_key]
        else:
            prompt = self._invoke_llm(
                ollama_url, ollama_model, content_rating, target_model,
                user_text, duration_seconds, motion_intensity, temperature, max_tokens,
            )
            if len(_CACHE) >= _MAX_CACHE_SIZE:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[cache_key] = prompt

        negative = _NEGATIVE_PROMPTS.get(target_model, "blurry, low quality, inconsistent motion")
        return prompt, negative, prompt

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        target_model: str,
        user_text: str,
        duration_seconds: int,
        motion_intensity: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        model_rules = _MODEL_STYLE_RULES.get(target_model, _MODEL_STYLE_RULES["WAN-T2V"])
        system_prompt = load_system_prompt_template(
            "system_prompts/video_scene_expander_system.txt",
            content_rating,
            model_style_rules=model_rules,
            duration_seconds=duration_seconds,
        )
        user_prompt = (
            f"Scene idea: {user_text.strip() or '(no input — create something visually compelling and cinematic)'}\n"
            f"Target model: {target_model}\n"
            f"Motion intensity: {motion_intensity}\n"
            f"Clip duration: {duration_seconds} seconds\n\n"
            f"Generate the video prompt now. Output only the prompt:"
        )

        ok, result = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={"temperature": float(temperature), "num_predict": int(max_tokens) * 4},
            timeout=120,
        )

        if not ok:
            return f"[Error: {result}]"

        result = _clean_output(result)

        if content_rating == "SFW":
            if err := enforce_sfw(result):
                return f"[Blocked: {err}]"

        return result or "[Empty response from Ollama]"


NODE_CLASS_MAPPINGS = {"WizdroidVideoSceneExpander": WizdroidVideoSceneExpanderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidVideoSceneExpander": "🎬 Video Scene Expander"}
