import json
from typing import Any, Dict, List, Optional, Tuple

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import apply_content_policy
DEFAULT_CONTEST_JSON = "contest.json"
def _load_json(filename: str) -> Any:
    return load_json(filename)


def _load_contest() -> Dict[str, Any]:
    """Load the current contest definition.

    Intention: keep a single common JSON filename that you replace per contest.

    Supported schema:
    - {"categories": [{"name": str, "description": str}, ...], "system_prompt": str}
    - (legacy) {"categories": ["Name", ...], "system_prompt": str}
    """

    payload: Dict[str, Any] = {}
    try:
        payload = _load_json(DEFAULT_CONTEST_JSON) or {}
    except Exception:  # noqa: BLE001
        payload = {}

    categories_raw = payload.get("categories") or payload.get("subjects") or []
    categories: List[Dict[str, str]] = []

    if isinstance(categories_raw, list):
        for item in categories_raw:
            if isinstance(item, str):
                categories.append({"name": item.strip(), "description": ""})
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("title") or "").strip()
                desc = str(item.get("description") or item.get("text") or "").strip()
                if name:
                    categories.append({"name": name, "description": desc})

    payload["_categories"] = categories
    return payload


class WizdroidContestPromptNode:
    """🧙 Generate contest-ready prompts from data/contest.json using Ollama LLM."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:  # noqa: N802
        contest = _load_contest()
        cats = [c.get("name", "").strip() for c in (contest.get("_categories") or [])]
        cats = [c for c in cats if c]
        if not cats:
            cats = ["missing_contest_json"]

        models = collect_models(DEFAULT_OLLAMA_URL)

        return {
            "required": {
                "ollama_url": (
                    "STRING",
                    {
                        "default": DEFAULT_OLLAMA_URL,
                        "multiline": False,
                    },
                ),
                "ollama_model": (
                    tuple(models),
                    {
                        "default": models[0] if models else "model_not_available",
                    },
                ),
                "content_rating": (
                    CONTENT_RATING_CHOICES,
                    {
                        "default": "SFW",
                    },
                ),
                "category": (
                    tuple(cats),
                    {
                        "default": cats[0],
                    },
                ),
                "extra_notes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional: add any extra constraints (e.g. 'no text, no watermark', '3D render', 'portrait framing')",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 128,
                        "max": 4096,
                        "step": 16,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "debug")
    FUNCTION = "generate"
    CATEGORY = "🧙 Wizdroid/Prompts"

    def generate(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        category: str,
        extra_notes: str,
        seed: int,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Tuple[str, str]:
        cfg = _load_contest()
        categories: List[Dict[str, str]] = list(cfg.get("_categories") or [])

        cat = (category or "").strip()
        cat_obj: Optional[Dict[str, str]] = next((c for c in categories if c.get("name") == cat), None)
        if not cat_obj:
            available = ", ".join([c.get("name", "") for c in categories if c.get("name")])
            return ("", f"ContestPrompt error: invalid category '{cat}'. Available: {available}")

        category_text = (cat_obj.get("description") or "").strip()

        # The category description is the "subject text" the user wants.
        # Keep the node generic: any contest can swap contest.json and restart.
        user_prompt = (
            f"Category: {cat}\n"
            f"Category description:\n{category_text}\n\n"
            f"Extra notes (optional): {extra_notes.strip()}\n\n"
            "Generate ONE random, contest-ready, visually unambiguous text-to-image prompt for this category. "
            "Output only the prompt text."
        ).strip()

        system_prompt = apply_content_policy(str(cfg.get("system_prompt") or ""), content_rating)
        ok, full = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={
                "temperature": float(temperature),
                "seed": int(seed),
                "num_predict": int(max_tokens),
            },
            timeout=120,
        )
        if not ok:
            return ("", f"ContestPrompt error: {full}")

        if content_rating == "SFW":
            err = enforce_sfw(full)
            if err:
                return (
                    "",
                    "ContestPrompt blocked: potential NSFW content detected. Switch content_rating to 'Mixed' or 'NSFW' or revise category/notes.",
                )

        return (full, f"Category: {cat}\n\n{category_text}")


NODE_CLASS_MAPPINGS = {
    "WizdroidContestPrompt": WizdroidContestPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidContestPrompt": "🧙 Wizdroid: Contest Prompt",
}
