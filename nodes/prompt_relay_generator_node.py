import json
import hashlib
import re
from typing import Dict, List, Tuple

from wizdroid_lib.constants import DEFAULT_OLLAMA_URL
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_template

from .video_scene_expander_node import _clean_output

# === Caching ===

_CACHE: Dict[str, Tuple] = {}
_MAX_CACHE_SIZE = 50


def _cache_key(data: Dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _parse_segment(text: str, n: int) -> str:
    """Extract content of [SEGMENT_N] tag."""
    pattern = rf"\[SEGMENT_{n}\]\s*(.*?)(?=\[SEGMENT_|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


class WizdroidPromptRelayGeneratorNode:
    """🎬 Generate time-segmented prompt sequences for WAN Prompt Relay temporal control."""

    CATEGORY = "🧙 Wizdroid/Video"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("segment_1", "segment_2", "segment_3", "segment_4", "timecodes")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)
        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "concept": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe the full scene concept, e.g. 'a butterfly emerging from a cocoon in a sunlit forest'"}),
                "num_segments": ("INT", {"default": 3, "min": 2, "max": 4, "step": 1}),
                "total_duration_seconds": ("INT", {"default": 10, "min": 3, "max": 60, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 400, "min": 150, "max": 800, "step": 10}),
                "use_ai": ("BOOLEAN", {"default": True}),
                "spiciness": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "detail_level": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "fantasy": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def generate(
        self,
        ollama_url: str,
        ollama_model: str,
        concept: str,
        num_segments: int,
        total_duration_seconds: int,
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
            "num_segments": num_segments,
            "duration": total_duration_seconds,
            "temp": temperature,
                        "seed": seed,
        }

        cache_key = _cache_key(selections)
        if use_ai and cache_key in _CACHE:
            segments, timecodes = _CACHE[cache_key]
        elif use_ai:
            segments, timecodes = self._invoke_llm(
                ollama_url, ollama_model,
                concept, num_segments, total_duration_seconds, temperature, max_tokens,
            )
            if len(_CACHE) >= _MAX_CACHE_SIZE:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[cache_key] = (segments, timecodes)
        else:
            segments = [concept.strip() or "(concept)"]
            timecodes = "0:00 - " + str(total_duration_seconds) + "s | Relay prompt"

        # Pad to 4 outputs
        while len(segments) < 4:
            segments.append("")

        return segments[0], segments[1], segments[2], segments[3], timecodes

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        concept: str,
        num_segments: int,
        total_duration: int,
        temperature: float,
        max_tokens: int,
    ):
        system_prompt = load_system_prompt_template(
            "system_prompts/prompt_relay_system.txt",
            
            num_segments=num_segments,
        )

        seg_duration = total_duration / num_segments
        tc_parts = [f"Segment {i+1}: {i * seg_duration:.1f}s – {(i+1) * seg_duration:.1f}s" for i in range(num_segments)]
        timecodes = " | ".join(tc_parts)

        user_prompt = (
            f"Concept: {concept.strip() or '(create a compelling cinematic sequence)'}\n"
            f"Number of segments: {num_segments}\n"
            f"Total duration: {total_duration} seconds\n"
            f"Timecode structure: {timecodes}\n\n"
            f"Generate exactly {num_segments} segment prompts in [SEGMENT_1] through [SEGMENT_{num_segments}] tags:"
        )

        ok, result = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={"temperature": float(temperature), "num_predict": int(max_tokens) * 4},
            timeout=150,
        )

        if not ok:
            err = f"[Error: {result}]"
            return [err, "", "", ""], timecodes

        segments: List[str] = []
        for i in range(1, num_segments + 1):
            seg = _parse_segment(result, i)
            seg = _clean_output(seg) if seg else ""
            segments.append(seg or f"[Empty segment {i}]")

        return segments, timecodes


NODE_CLASS_MAPPINGS = {"WizdroidPromptRelayGenerator": WizdroidPromptRelayGeneratorNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidPromptRelayGenerator": "🎬 Prompt Relay Generator"}
