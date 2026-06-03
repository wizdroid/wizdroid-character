import json
import hashlib
import re
from typing import Dict

from wizdroid_lib.constants import DEFAULT_OLLAMA_URL
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

from .video_scene_expander_node import VIDEO_MODELS, _NEGATIVE_PROMPTS, _MODEL_STYLE_RULES, _clean_output

# === Arc types ===

ARC_TYPES = ("buildup", "action-peak", "discovery", "journey", "transformation")

# === Caching ===

_CACHE: Dict[str, tuple] = {}
_MAX_CACHE_SIZE = 50


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _parse_tagged(text: str, tag: str) -> str:
    """Extract content between [TAG] and the next [ or end of string."""
    pattern = rf"\[{re.escape(tag)}\]\s*(.*?)(?=\[|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


class WizdroidTemporalScenePlannerNode:
    """🎬 Plan video prompts with a full beginning-to-end temporal arc."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_prompt",)
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)
        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "target_model": (VIDEO_MODELS, {"default": "WAN-T2V"}),
                "concept": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe the concept, e.g. 'a samurai facing a setting sun before battle'"}),
                "arc_type": (ARC_TYPES, {"default": "buildup"}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 300, "min": 100, "max": 600, "step": 10}),
                "use_ai": ("BOOLEAN", {"default": True}),
                "spiciness": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "detail_level": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "fantasy": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def plan(
        self,
        ollama_url: str,
        ollama_model: str,
        target_model: str,
        concept: str,
        arc_type: str,
        temperature: float,
        max_tokens: int,
        use_ai: bool,
        seed: int,
        spiciness: int = 0,
        detail_level: int = 5,
        fantasy: int = 0,
    ):
        global _CACHE

        selections = {
            "concept": concept.strip(),
            "model": target_model,
            "arc": arc_type,
            "temp": temperature,
                        "seed": seed,
        }

        cache_key = _cache_key(selections)
        if use_ai and cache_key in _CACHE:
            video_prompt, start_desc, end_desc = _CACHE[cache_key]
        elif use_ai:
            video_prompt, start_desc, end_desc = self._invoke_llm(
                ollama_url, ollama_model, target_model,
                concept, arc_type, temperature, max_tokens, spiciness,
            )
            if len(_CACHE) >= _MAX_CACHE_SIZE:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[cache_key] = (video_prompt, start_desc, end_desc)
        else:
            video_prompt = concept.strip() or "(concept)"
            start_desc = ""
            end_desc = ""

        return (video_prompt,)

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        target_model: str,
        concept: str,
        arc_type: str,
        temperature: float,
        max_tokens: int,
        spiciness: int = 0,
    ):
        model_rules = _MODEL_STYLE_RULES.get(target_model, _MODEL_STYLE_RULES["WAN-T2V"])
        system_prompt = load_system_prompt_template(
            "system_prompts/temporal_scene_planner_system.txt",
            
            model_style_rules=model_rules,
            arc_type=arc_type,
        )

        user_prompt = (
            f"Concept: {concept.strip() or '(no input — create a compelling cinematic arc)'}\n"
            f"Target model: {target_model}\n"
            f"Arc type: {arc_type}\n"
            f"{'' if spiciness == 0 else f'INSTRUCTION - Tone: {['','mildly suggestive','playful sensuality','romantic intimacy','sensual and warm','openly sensual','boldly erotic','unabashedly erotic','raw and explicit','extremely explicit','MAXIMUM SPICINESS'][spiciness]}.\\n'}"
            f"Generate the temporal video prompt with [PROMPT], [START], and [END] sections:"
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
            err = f"[Error: {result}]"
            return err, "", ""

        video_prompt = _parse_tagged(result, "PROMPT") or _clean_output(result)
        start_desc = _parse_tagged(result, "START")
        end_desc = _parse_tagged(result, "END")

        return (
            video_prompt or "[Empty response from Ollama]",
            start_desc,
            end_desc,
        )


NODE_CLASS_MAPPINGS = {"WizdroidTemporalScenePlanner": WizdroidTemporalScenePlannerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidTemporalScenePlanner": "🎬 Temporal Scene Planner"}
