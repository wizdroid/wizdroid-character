import json
import hashlib
import random
from typing import Any, Dict, Optional, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json
from wizdroid_lib.helpers import with_random, choose
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

from .video_scene_expander_node import _clean_output

# === Data cache ===

_CAMERA_CACHE: Optional[Tuple[int, Dict]] = None
_DEFAULT_CAMERA = {
    "shot_types": [{"name": "medium shot"}, {"name": "close-up"}, {"name": "establishing wide shot"}],
    "camera_movements": [{"name": "static hold"}, {"name": "slow pan right"}, {"name": "dolly push in"}],
    "lens_types": [{"name": "standard 50mm"}, {"name": "wide angle"}, {"name": "medium telephoto 85mm"}],
}

_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 50


def _load_camera_data(cache_slot):
    from wizdroid_lib.paths import DATA_DIR
    path = DATA_DIR / "video_camera_data.json"
    try:
        if not path.exists():
            return (0, _DEFAULT_CAMERA), _DEFAULT_CAMERA
        mtime = int(path.stat().st_mtime_ns)
    except OSError:
        return (0, _DEFAULT_CAMERA), _DEFAULT_CAMERA
    if cache_slot and cache_slot[0] == mtime:
        return cache_slot, cache_slot[1]
    try:
        payload = load_json("video_camera_data.json")
        return (mtime, payload), payload
    except Exception:
        return (0, _DEFAULT_CAMERA), _DEFAULT_CAMERA


def _names(items: Any) -> list:
    if isinstance(items, list):
        return [i["name"] if isinstance(i, dict) else str(i) for i in items if i]
    return []


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class WizdroidCinematicShotDesignerNode:
    """🎬 Design cinematic camera shots and movements for video prompts."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("shot_description", "preview")
    FUNCTION = "design"

    @classmethod
    def INPUT_TYPES(cls):
        global _CAMERA_CACHE
        _CAMERA_CACHE, camera_data = _load_camera_data(_CAMERA_CACHE)

        shot_types = _names(camera_data.get("shot_types", _DEFAULT_CAMERA["shot_types"]))
        movements = _names(camera_data.get("camera_movements", _DEFAULT_CAMERA["camera_movements"]))
        lenses = _names(camera_data.get("lens_types", _DEFAULT_CAMERA["lens_types"]))
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "subject_description": ("STRING", {"multiline": True, "default": "", "placeholder": "Brief subject hint, e.g. 'a woman standing in a doorway'"}),
                "shot_type": (with_random(shot_types), {"default": RANDOM_LABEL}),
                "camera_movement": (with_random(movements), {"default": RANDOM_LABEL}),
                "lens_type": (with_random(lenses), {"default": RANDOM_LABEL}),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def design(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        subject_description: str,
        shot_type: str,
        camera_movement: str,
        lens_type: str,
        temperature: float,
        seed: int,
    ):
        global _CAMERA_CACHE, _CACHE
        _CAMERA_CACHE, camera_data = _load_camera_data(_CAMERA_CACHE)

        rng = random.Random(seed)
        resolved_shot = choose(shot_type, _names(camera_data.get("shot_types", [])), rng, seed)
        resolved_move = choose(camera_movement, _names(camera_data.get("camera_movements", [])), rng, seed)
        resolved_lens = choose(lens_type, _names(camera_data.get("lens_types", [])), rng, seed)

        selections = {
            "subject": subject_description.strip(),
            "shot": resolved_shot,
            "movement": resolved_move,
            "lens": resolved_lens,
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
                subject_description, resolved_shot, resolved_move, resolved_lens, temperature,
            )
            if len(_CACHE) >= _MAX_CACHE_SIZE:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[cache_key] = desc

        return desc, desc

    @staticmethod
    def _invoke_llm(
        ollama_url: str, ollama_model: str, content_rating: str,
        subject: str, shot_type: str, camera_movement: str, lens_type: str, temperature: float,
    ) -> str:
        system_prompt = load_system_prompt_template(
            "system_prompts/cinematic_shot_designer_system.txt",
            content_rating,
            shot_type=shot_type or "medium shot",
            camera_movement=camera_movement or "static hold",
            lens_type=lens_type or "standard 50mm",
        )

        user_prompt = (
            f"Subject/scene hint: {subject.strip() or '(a subject in a scene)'}\n"
            f"Shot type: {shot_type or 'medium shot'}\n"
            f"Camera movement: {camera_movement or 'static hold'}\n"
            f"Lens type: {lens_type or 'standard 50mm'}\n\n"
            f"Generate the shot description (1-3 sentences, output only):"
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


NODE_CLASS_MAPPINGS = {"WizdroidCinematicShotDesigner": WizdroidCinematicShotDesignerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidCinematicShotDesigner": "🎬 Cinematic Shot Designer"}
