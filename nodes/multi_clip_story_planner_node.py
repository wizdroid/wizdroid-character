import json
import hashlib
import re
from typing import Dict, List, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

from .video_scene_expander_node import VIDEO_MODELS, _MODEL_STYLE_RULES, _NEGATIVE_PROMPTS, _clean_output

# === Story genres ===

STORY_GENRES = ("action", "drama", "comedy", "documentary", "fantasy", "sci-fi", "thriller", "romance", "horror")

# === Caching ===

_CACHE: Dict[str, Tuple] = {}
_MAX_CACHE_SIZE = 30


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _parse_clip(text: str, n: int) -> str:
    """Extract content of [CLIP_N] tag."""
    pattern = rf"\[CLIP_{n}\]\s*(.*?)(?=\[CLIP_|\[SUMMARY\]|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _parse_summary(text: str) -> str:
    pattern = r"\[SUMMARY\]\s*(.*?)(?=\[CLIP_|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


class WizdroidMultiClipStoryPlannerNode:
    """🎬 Plan a multi-clip story sequence with individual prompts for each clip."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("clip_1", "clip_2", "clip_3", "clip_4", "clip_5", "clip_6", "story_summary", "preview")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)
        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "target_model": (VIDEO_MODELS, {"default": "WAN-T2V"}),
                "story_concept": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe the story, e.g. 'a detective tracking a thief through a rain-soaked city'"}),
                "num_clips": ("INT", {"default": 3, "min": 2, "max": 6, "step": 1}),
                "story_genre": (STORY_GENRES, {"default": "drama"}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 600, "min": 200, "max": 1200, "step": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def plan(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        target_model: str,
        story_concept: str,
        num_clips: int,
        story_genre: str,
        temperature: float,
        max_tokens: int,
        seed: int,
    ):
        global _CACHE

        selections = {
            "concept": story_concept.strip(),
            "num_clips": num_clips,
            "genre": story_genre,
            "model": target_model,
            "temp": temperature,
            "content_rating": content_rating,
            "seed": seed,
        }

        cache_key = _cache_key(selections)
        if cache_key in _CACHE:
            clips, summary = _CACHE[cache_key]
        else:
            clips, summary = self._invoke_llm(
                ollama_url, ollama_model, content_rating, target_model,
                story_concept, num_clips, story_genre, temperature, max_tokens,
            )
            if len(_CACHE) >= _MAX_CACHE_SIZE:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[cache_key] = (clips, summary)

        # Ensure exactly 6 outputs
        while len(clips) < 6:
            clips.append("")

        preview_lines = [f"Story: {summary}", ""]
        for i, clip in enumerate(clips[:num_clips], 1):
            preview_lines.append(f"[Clip {i}]\n{clip}")

        return (*clips[:6], summary, "\n\n".join(preview_lines))

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        target_model: str,
        story_concept: str,
        num_clips: int,
        story_genre: str,
        temperature: float,
        max_tokens: int,
    ):
        model_rules = _MODEL_STYLE_RULES.get(target_model, _MODEL_STYLE_RULES["WAN-T2V"])
        system_prompt = load_system_prompt_template(
            "system_prompts/multi_clip_story_planner_system.txt",
            content_rating,
            model_style_rules=model_rules,
            num_clips=num_clips,
            story_genre=story_genre,
        )

        user_prompt = (
            f"Story concept: {story_concept.strip() or '(create an original compelling story)'}\n"
            f"Number of clips: {num_clips}\n"
            f"Genre: {story_genre}\n"
            f"Target model: {target_model}\n\n"
            f"Generate the [SUMMARY] followed by [CLIP_1] through [CLIP_{num_clips}]:"
        )

        ok, result = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={"temperature": float(temperature), "num_predict": int(max_tokens) * 4},
            timeout=180,
        )

        if not ok:
            err = f"[Error: {result}]"
            return [err] + [""] * 5, ""

        summary = _parse_summary(result)
        clips: List[str] = []
        for i in range(1, num_clips + 1):
            clip = _parse_clip(result, i)
            clip = _clean_output(clip) if clip else ""
            if content_rating == "SFW" and clip:
                if err := enforce_sfw(clip):
                    clip = f"[Blocked: {err}]"
            clips.append(clip or f"[Empty clip {i}]")

        return clips, summary or "(no summary)"


NODE_CLASS_MAPPINGS = {"WizdroidMultiClipStoryPlanner": WizdroidMultiClipStoryPlannerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidMultiClipStoryPlanner": "🎬 Multi-Clip Story Planner"}
