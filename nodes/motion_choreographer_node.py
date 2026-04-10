import json
import hashlib
from typing import Any, Dict, Optional, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json
from wizdroid_lib.helpers import with_random, choose
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template
import random

from .video_scene_expander_node import _clean_output

# === Data caches ===

_MOTION_CACHE: Optional[Tuple[int, Dict]] = None
_DEFAULT_MOTION = {
    "motion_styles": [{"name": "natural casual"}, {"name": "slow and deliberate"}, {"name": "explosive action"}],
    "motion_phases": [{"name": "sustained action"}, {"name": "full action sequence"}],
}

_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 50


def _load_motion_data(cache_slot):
    from wizdroid_lib.paths import DATA_DIR
    path = DATA_DIR / "video_motion_types.json"
    try:
        if not path.exists():
            return (0, _DEFAULT_MOTION), _DEFAULT_MOTION
        mtime = int(path.stat().st_mtime_ns)
    except OSError:
        return (0, _DEFAULT_MOTION), _DEFAULT_MOTION
    if cache_slot and cache_slot[0] == mtime:
        return cache_slot, cache_slot[1]
    try:
        payload = load_json("video_motion_types.json")
        return (mtime, payload), payload
    except Exception:
        return (0, _DEFAULT_MOTION), _DEFAULT_MOTION


def _names(items: Any) -> list:
    if isinstance(items, list):
        return [i["name"] if isinstance(i, dict) else str(i) for i in items if i]
    return []


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class WizdroidMotionChoreographerNode:
    """🎬 Generate precise motion descriptions for injecting into video prompts."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("motion_description", "preview")
    FUNCTION = "choreograph"

    @classmethod
    def INPUT_TYPES(cls):
        global _MOTION_CACHE
        _MOTION_CACHE, motion_data = _load_motion_data(_MOTION_CACHE)

        motion_styles = _names(motion_data.get("motion_styles", _DEFAULT_MOTION["motion_styles"]))
        motion_phases = _names(motion_data.get("motion_phases", _DEFAULT_MOTION["motion_phases"]))
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "subject_description": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe the subject, e.g. 'an elderly monk in brown robes'"}),
                "motion_style": (with_random(motion_styles), {"default": RANDOM_LABEL}),
                "motion_phase": (with_random(motion_phases), {"default": "sustained action"}),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def choreograph(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        subject_description: str,
        motion_style: str,
        motion_phase: str,
        temperature: float,
        seed: int,
    ):
        global _MOTION_CACHE, _CACHE
        _MOTION_CACHE, motion_data = _load_motion_data(_MOTION_CACHE)

        rng = random.Random(seed)
        motion_styles = _names(motion_data.get("motion_styles", []))
        motion_phases = _names(motion_data.get("motion_phases", []))

        resolved_style = choose(motion_style, motion_styles, rng, seed)
        resolved_phase = choose(motion_phase, motion_phases, rng, seed)

        selections = {
            "subject": subject_description.strip(),
            "style": resolved_style,
            "phase": resolved_phase,
            "temp": temperature,
            "content_rating": content_rating,
            "seed": seed,
        }

        cache_key = _cache_key(selections)
        if cache_key in _CACHE:
            desc = _CACHE[cache_key]
        else:
            desc = self._invoke_llm(
                ollama_url, ollama_model, content_rating,
                subject_description, resolved_style, resolved_phase, temperature,
            )
            if len(_CACHE) >= _MAX_CACHE_SIZE:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[cache_key] = desc

        return desc, desc

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        subject: str,
        motion_style: str,
        motion_phase: str,
        temperature: float,
    ) -> str:
        system_prompt = load_system_prompt_template(
            "system_prompts/motion_choreographer_system.txt",
            content_rating,
            motion_style=motion_style or "natural casual",
            motion_phase=motion_phase or "sustained action",
        )

        user_prompt = (
            f"Subject: {subject.strip() or '(a person)'}\n"
            f"Motion style: {motion_style or 'natural casual'}\n"
            f"Motion phase: {motion_phase or 'sustained action'}\n\n"
            f"Generate the motion description (1-3 sentences, output only):"
        )

        ok, result = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={"temperature": float(temperature), "num_predict": 256},
            timeout=60,
        )

        if not ok:
            return f"[Error: {result}]"

        result = _clean_output(result)

        if content_rating == "SFW":
            if err := enforce_sfw(result):
                return f"[Blocked: {err}]"

        return result or "[Empty response from Ollama]"


NODE_CLASS_MAPPINGS = {"WizdroidMotionChoreographer": WizdroidMotionChoreographerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidMotionChoreographer": "🎬 Motion Choreographer"}
