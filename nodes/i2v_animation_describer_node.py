import json
import hashlib
from typing import Dict

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

from .video_scene_expander_node import _NEGATIVE_PROMPTS, _MODEL_STYLE_RULES, _clean_output

# === I2V model options ===

I2V_MODELS = ("WAN-I2V", "LTX-I2V")
ANIMATION_FOCUS_OPTIONS = ("subject", "environment", "both")

# === Caching ===

_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 50


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class WizdroidI2VAnimationDescriberNode:
    """🎬 Turn a static image description into an I2V animation prompt."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("animation_prompt", "negative_prompt", "preview")
    FUNCTION = "describe"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)
        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "target_model": (I2V_MODELS, {"default": "WAN-I2V"}),
                "image_description": ("STRING", {"multiline": True, "default": "", "placeholder": "Paste the image description here, or wire from Photo Aspect Extractor output"}),
                "animation_focus": (ANIMATION_FOCUS_OPTIONS, {"default": "both"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 250, "min": 80, "max": 500, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def describe(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        target_model: str,
        image_description: str,
        animation_focus: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ):
        global _CACHE

        selections = {
            "image_desc": image_description.strip(),
            "model": target_model,
            "focus": animation_focus,
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
                image_description, animation_focus, temperature, max_tokens,
            )
            if len(_CACHE) >= _MAX_CACHE_SIZE:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[cache_key] = prompt

        negative = _NEGATIVE_PROMPTS.get(target_model, "worst quality, inconsistent motion, blurry, jittery, distorted")
        return prompt, negative, prompt

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        target_model: str,
        image_description: str,
        animation_focus: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        model_rules = _MODEL_STYLE_RULES.get(target_model, _MODEL_STYLE_RULES["WAN-I2V"])
        system_prompt = load_system_prompt_template(
            "system_prompts/i2v_animation_system.txt",
            content_rating,
            model_style_rules=model_rules,
            animation_focus=animation_focus,
        )

        user_prompt = (
            f"Image description:\n{image_description.strip() or '(no description provided — create a plausible animation)'}\n\n"
            f"Target model: {target_model}\n"
            f"Animation focus: {animation_focus}\n\n"
            f"Generate the animation video prompt now. Output only the prompt:"
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


NODE_CLASS_MAPPINGS = {"WizdroidI2VAnimationDescriber": WizdroidI2VAnimationDescriberNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidI2VAnimationDescriber": "🎬 I2V Animation Describer"}
