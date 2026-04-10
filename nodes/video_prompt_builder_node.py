import json
import hashlib
import random
from typing import Any, Dict, Optional, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json
from wizdroid_lib.helpers import choose, with_random
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

from .video_scene_expander_node import VIDEO_MODELS, _NEGATIVE_PROMPTS, _MODEL_STYLE_RULES, _clean_output

# === Data caches (mtime-based) ===

_SCENE_TYPES_CACHE: Optional[Tuple[int, Dict]] = None
_MOTION_TYPES_CACHE: Optional[Tuple[int, Dict]] = None

_DEFAULT_SCENE_TYPES = {
    "scene_types": ["urban street", "natural landscape", "interior room"],
    "environments": ["city downtown", "forest", "desert"],
    "time_of_day": [{"name": "golden hour dawn"}, {"name": "bright midday"}, {"name": "night with city lights"}],
    "moods": [{"name": "epic cinematic"}, {"name": "peaceful serene"}, {"name": "tense thriller"}],
}
_DEFAULT_MOTION_TYPES = {
    "motion_styles": [{"name": "natural casual"}, {"name": "slow and deliberate"}, {"name": "explosive action"}],
    "motion_phases": [{"name": "sustained action"}, {"name": "full action sequence"}],
}

_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 50


def _load_cached(filename: str, default: Dict, cache_slot: Optional[Tuple]) -> Tuple[Optional[Tuple], Dict]:
    from wizdroid_lib.paths import DATA_DIR
    path = DATA_DIR / filename
    try:
        if not path.exists():
            return (0, default), default
        mtime = int(path.stat().st_mtime_ns)
    except OSError:
        return (0, default), default
    if cache_slot and cache_slot[0] == mtime:
        return cache_slot, cache_slot[1]
    try:
        payload = load_json(filename)
        return (mtime, payload), payload
    except Exception:
        return (0, default), default


def _names(items: Any) -> list:
    if isinstance(items, list):
        return [i["name"] if isinstance(i, dict) else str(i) for i in items if i]
    return []


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class WizdroidVideoPromptBuilderNode:
    """🎬 Build structured video prompts from categorical inputs for WAN 2.2 and LTX-Video."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_prompt", "negative_prompt", "preview")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        global _SCENE_TYPES_CACHE, _MOTION_TYPES_CACHE

        _SCENE_TYPES_CACHE, scene_data = _load_cached("video_scene_types.json", _DEFAULT_SCENE_TYPES, _SCENE_TYPES_CACHE)
        _MOTION_TYPES_CACHE, motion_data = _load_cached("video_motion_types.json", _DEFAULT_MOTION_TYPES, _MOTION_TYPES_CACHE)

        scene_types = _names(scene_data.get("scene_types", _DEFAULT_SCENE_TYPES["scene_types"]))
        environments = _names(scene_data.get("environments", _DEFAULT_SCENE_TYPES["environments"]))
        times = _names(scene_data.get("time_of_day", _DEFAULT_SCENE_TYPES["time_of_day"]))
        moods = _names(scene_data.get("moods", _DEFAULT_SCENE_TYPES["moods"]))
        motion_styles = _names(motion_data.get("motion_styles", _DEFAULT_MOTION_TYPES["motion_styles"]))

        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "target_model": (VIDEO_MODELS, {"default": "WAN-T2V"}),
                "subject_description": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe the main subject, e.g. 'a young woman in a red coat'"}),
                "scene_type": (with_random(scene_types), {"default": RANDOM_LABEL}),
                "environment": (with_random(environments), {"default": RANDOM_LABEL}),
                "motion_style": (with_random(motion_styles), {"default": RANDOM_LABEL}),
                "time_of_day": (with_random(times), {"default": RANDOM_LABEL}),
                "mood": (with_random(moods), {"default": RANDOM_LABEL}),
                "duration_seconds": ("INT", {"default": 5, "min": 2, "max": 60, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 250, "min": 80, "max": 500, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def build(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        target_model: str,
        subject_description: str,
        scene_type: str,
        environment: str,
        motion_style: str,
        time_of_day: str,
        mood: str,
        duration_seconds: int,
        temperature: float,
        max_tokens: int,
        seed: int,
    ):
        global _SCENE_TYPES_CACHE, _MOTION_TYPES_CACHE, _CACHE

        _SCENE_TYPES_CACHE, scene_data = _load_cached("video_scene_types.json", _DEFAULT_SCENE_TYPES, _SCENE_TYPES_CACHE)
        _MOTION_TYPES_CACHE, motion_data = _load_cached("video_motion_types.json", _DEFAULT_MOTION_TYPES, _MOTION_TYPES_CACHE)

        rng = random.Random(seed)

        resolved_scene = choose(scene_type, _names(scene_data.get("scene_types", [])), rng, seed)
        resolved_env = choose(environment, _names(scene_data.get("environments", [])), rng, seed)
        resolved_motion = choose(motion_style, _names(motion_data.get("motion_styles", [])), rng, seed)
        resolved_time = choose(time_of_day, _names(scene_data.get("time_of_day", [])), rng, seed)
        resolved_mood = choose(mood, _names(scene_data.get("moods", [])), rng, seed)

        selections = {
            "subject": subject_description.strip(),
            "scene": resolved_scene,
            "env": resolved_env,
            "motion": resolved_motion,
            "time": resolved_time,
            "mood": resolved_mood,
            "duration": duration_seconds,
            "target_model": target_model,
            "content_rating": content_rating,
            "temp": temperature,
        }

        cache_key = _cache_key(selections)
        if cache_key in _CACHE:
            prompt = _CACHE[cache_key]
        else:
            prompt = self._invoke_llm(ollama_url, ollama_model, content_rating, target_model,
                                      selections, temperature, max_tokens)
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
        selections: Dict,
        temperature: float,
        max_tokens: int,
    ) -> str:
        model_rules = _MODEL_STYLE_RULES.get(target_model, _MODEL_STYLE_RULES["WAN-T2V"])
        system_prompt = load_system_prompt_template(
            "system_prompts/video_prompt_builder_system.txt",
            content_rating,
            model_style_rules=model_rules,
            duration_seconds=selections.get("duration", 5),
        )

        parts = []
        if subj := selections.get("subject"):
            parts.append(f"Subject: {subj}")
        if v := selections.get("scene"):
            parts.append(f"Scene type: {v}")
        if v := selections.get("env"):
            parts.append(f"Environment: {v}")
        if v := selections.get("motion"):
            parts.append(f"Motion style: {v}")
        if v := selections.get("time"):
            parts.append(f"Time of day: {v}")
        if v := selections.get("mood"):
            parts.append(f"Mood: {v}")
        parts.append(f"Target model: {target_model}")
        parts.append(f"Duration: {selections.get('duration', 5)} seconds")

        user_prompt = "\n".join(parts) + "\n\nGenerate the video prompt now. Output only the prompt:"

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


NODE_CLASS_MAPPINGS = {"WizdroidVideoPromptBuilder": WizdroidVideoPromptBuilderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidVideoPromptBuilder": "🎬 Video Prompt Builder"}
