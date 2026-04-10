import json
import hashlib
from typing import Dict

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

from .video_scene_expander_node import VIDEO_MODELS, _NEGATIVE_PROMPTS, _MODEL_STYLE_RULES, _clean_output

# === Caching ===

_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 50


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class WizdroidImageToVideoAdapterNode:
    """🎬 Convert an existing image prompt into a video-ready format for WAN 2.2 or LTX-Video."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_prompt", "negative_prompt", "preview")
    FUNCTION = "adapt"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)
        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "target_model": (VIDEO_MODELS, {"default": "WAN-T2V"}),
                "image_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Paste your image prompt here — from Character, Scene, or any other node"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 250, "min": 80, "max": 500, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "motion_hint": ("STRING", {"multiline": False, "default": "", "placeholder": "Optional: hint for motion type, e.g. 'walking slowly', 'wind in trees'"}),
            }
        }

    def adapt(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        target_model: str,
        image_prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        motion_hint: str = "",
    ):
        global _CACHE

        selections = {
            "image_prompt": image_prompt.strip(),
            "model": target_model,
            "motion_hint": motion_hint.strip(),
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
                image_prompt, motion_hint, temperature, max_tokens,
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
        image_prompt: str,
        motion_hint: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        model_rules = _MODEL_STYLE_RULES.get(target_model, _MODEL_STYLE_RULES["WAN-T2V"])
        system_prompt = load_system_prompt_template(
            "system_prompts/image_to_video_adapter_system.txt",
            content_rating,
            model_style_rules=model_rules,
        )

        user_prompt = (
            f"Original image prompt:\n{image_prompt.strip() or '(no input provided)'}\n\n"
            f"Target video model: {target_model}\n"
        )
        if motion_hint.strip():
            user_prompt += f"Motion hint: {motion_hint.strip()}\n"
        user_prompt += "\nConvert to a video prompt now. Output only the video prompt:"

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


NODE_CLASS_MAPPINGS = {"WizdroidImageToVideoAdapter": WizdroidImageToVideoAdapterNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidImageToVideoAdapter": "🎬 Image→Video Adapter"}
