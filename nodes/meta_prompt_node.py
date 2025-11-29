import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

_JSON_CACHE: Dict[str, Tuple[int, Any]] = {}


def _safe_requests_post(url: str, json_body: Dict[str, Any], timeout: int = 60) -> Tuple[bool, str]:
    """Send a POST request and return (ok, text_or_error).

    This helper avoids raising exceptions inside the node and returns a readable
    error message instead.
    """

    if requests is None:
        return False, "request_error: requests library not installed"

    try:
        resp = requests.post(url, json=json_body, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return False, f"request_error: {type(e).__name__}: {e}"

    if resp.status_code != 200:
        return False, f"http_error: status {resp.status_code}: {resp.text[:512]}"

    return True, resp.text


def _collect_ollama_models(ollama_url: str) -> List[str]:
    """Best-effort discovery of available Ollama models.

    Returns a simple list of model names or a small fallback list if discovery
    fails. This matches the pattern used by other nodes in the package.
    """

    if requests is None:
        return ["install_requests_library"]

    try:
        resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code != 200:
            return ["model_not_available"]
        data = resp.json()
        models = [m.get("name", "unknown") for m in data.get("models", [])]
        return models or ["no_models_found"]
    except Exception:  # noqa: BLE001
        return ["ollama_not_running"]


def _load_json(name: str) -> Any:
    path = DATA_DIR / name
    mtime = int(path.stat().st_mtime_ns)
    cached = _JSON_CACHE.get(name)
    if cached and cached[0] == mtime:
        return cached[1]

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    _JSON_CACHE[name] = (mtime, payload)
    return payload


def _with_random(options: List[str]) -> Tuple[str, ...]:
    values: List[str] = [RANDOM_LABEL, NONE_LABEL]
    for option in options:
        if option != NONE_LABEL:
            values.append(option)
    return tuple(values)


def _choose(value: str, options: List[str], rng: random.Random) -> Optional[str]:
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt != NONE_LABEL]
        choice = rng.choice(pool) if pool else None
    else:
        choice = value
    if choice in (NONE_LABEL, None):
        return None
    return choice


def _get_meta_options() -> Dict[str, List[str]]:
    region_payload = _load_json("regions.json")
    meta_payload = _load_json("meta_prompt_options.json")
    return {
        "regions": list(region_payload.get("regions", [])),
        "futuristic_settings": list(meta_payload.get("futuristic_settings", [])),
        "ancient_eras": list(meta_payload.get("ancient_eras", [])),
        "mythological_elements": list(meta_payload.get("mythological_elements", [])),
        "visual_styles": list(meta_payload.get("visual_styles", [])),
    }


META_PROMPT_SYSTEM = """You are a prompt generator for an image AI. Your job is to take a short list of user keywords plus optional context directives and expand them into a single, vivid, well-structured prompt.

Requirements:
1. Input: The user will provide only a few words or short phrases.
2. Output: Produce one single prompt as plain text. Do not add explanations, headings, or lists.
3. Content: Use your imagination to add concrete visual detail: setting, mood, lighting, camera framing, clothing, materials, and background elements that fit the keywords. Keep style coherent with the implied genre and ALWAYS integrate any provided directives (regional influence, futuristic/ancient cues, mythological elements, visual style). Avoid explicit artist or franchise names.
4. Length: The final prompt must be between 150 and 1024 tokens. If there are few keywords, expand with descriptive detail; if there are many, focus on the most important and compress wording.
5. Tone: Write as if you are describing the image directly to the image model, using a natural flowing sentence or comma-separated phrases, not bullet points.

Now wait for the user’s keywords and output only the final expanded prompt that follows these rules."""


class MetaPromptGeneratorNode:
    """Generate a detailed image prompt from loose keywords using Ollama.

    This node takes a short text input (keywords or fragments) and asks an
    Ollama-hosted LLM to expand it into a 150–200 token imaginative prompt.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:  # noqa: N802
        models = _collect_ollama_models(DEFAULT_OLLAMA_URL)
        options = _get_meta_options()

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
                    models,
                    {
                        "default": models[0] if models else "model_not_available",
                    },
                ),
                "keywords": (
                    "STRING",
                    {
                        "default": "female knight, rainy neon city, blue coat",
                        "multiline": True,
                    },
                ),
                "regional_style": (
                    _with_random(options["regions"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "futuristic_setting": (
                    _with_random(options["futuristic_settings"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "ancient_setting": (
                    _with_random(options["ancient_eras"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "mythological_element": (
                    _with_random(options["mythological_elements"]),
                    {
                        "default": NONE_LABEL,
                    },
                ),
                "visual_style": (
                    _with_random(options["visual_styles"]),
                    {
                        "default": NONE_LABEL,
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
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.05,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Wizdroid/character"

    def generate_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        keywords: str,
        regional_style: str,
        futuristic_setting: str,
        ancient_setting: str,
        mythological_element: str,
        visual_style: str,
        seed: int,
        max_tokens: int = 1024,
        temperature: float = 0.8,
    ) -> Tuple[str]:
        """Call Ollama to expand user keywords into a detailed prompt."""

        # Use seed to introduce slight variation for batch mode consistency
        rng = random.Random(seed)
        meta_options = _get_meta_options()

        selected_directives = {
            "region": _choose(regional_style, meta_options["regions"], rng),
            "futuristic": _choose(futuristic_setting, meta_options["futuristic_settings"], rng),
            "ancient": _choose(ancient_setting, meta_options["ancient_eras"], rng),
            "mythology": _choose(mythological_element, meta_options["mythological_elements"], rng),
            "style": _choose(visual_style, meta_options["visual_styles"], rng),
        }

        directive_parts: List[str] = []
        if selected_directives["region"]:
            directive_parts.append(f"Regional inspiration: {selected_directives['region']}")
        if selected_directives["futuristic"]:
            directive_parts.append(f"Futuristic setting: {selected_directives['futuristic']}")
        if selected_directives["ancient"]:
            directive_parts.append(f"Historical inspiration: {selected_directives['ancient']}")
        if selected_directives["mythology"]:
            directive_parts.append(f"Mythological element: {selected_directives['mythology']}")
        if selected_directives["style"]:
            directive_parts.append(f"Visual style: {selected_directives['style']}")

        temp_variation = rng.uniform(-0.1, 0.1)  # Small temperature variation
        adjusted_temperature = max(0.0, min(1.5, temperature + temp_variation))

        api_url = f"{ollama_url.rstrip('/')}/api/generate"

        keyword_payload = keywords.strip() or "dreamlike character vignette"
        if directive_parts:
            context_block = "Context directives:\n" + "\n".join(f"- {part}" for part in directive_parts)
            prompt_input = f"{keyword_payload}\n{context_block}"
        else:
            prompt_input = keyword_payload

        payload = {
            "model": ollama_model,
            "stream": False,
            "options": {
                "temperature": float(adjusted_temperature),
                # Ollama uses num_predict as max tokens for completion
                "num_predict": int(max_tokens),
            },
            "system": META_PROMPT_SYSTEM,
            "prompt": prompt_input,
        }

        ok, text = _safe_requests_post(api_url, payload)
        if not ok:
            # Return the error string as the prompt so the user can see what happened
            return (f"MetaPrompt error: {text}",)

        # Ollama /api/generate returns JSON per call when stream=False
        try:
            data = json.loads(text)
            output = data.get("response", "").strip()
        except Exception:  # noqa: BLE001
            output = text.strip()

        if not output:
            output = "MetaPrompt error: empty response from Ollama"

        return (output,)


META_PROMPT_NODE_CLASS_MAPPINGS = {
    "MetaPromptGenerator": MetaPromptGeneratorNode,
}

META_PROMPT_NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaPromptGenerator": "Meta Prompt Generator",
}
