import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@lru_cache(maxsize=None)
def _load_json(name: str) -> Dict:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _with_random(options: List[str], allow_none: bool = True) -> Tuple[str, ...]:
    values: List[str] = [RANDOM_LABEL]
    if allow_none:
        values.append(NONE_LABEL)
    for option in options:
        if allow_none and option == NONE_LABEL:
            continue
        values.append(option)
    return tuple(values)


def _choose(value: Optional[str], options: List[str], rng: random.Random) -> Optional[str]:
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt != NONE_LABEL]
        if not pool:
            pool = options[:]
        selection = rng.choice(pool)
    else:
        selection = value

    if selection == NONE_LABEL or selection is None:
        return None
    return selection


class UniformFashionNode:
    """Generate luxe editorial prompts for capsule-based uniform concepts."""

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("uniform_prompt", "preview")
    FUNCTION = "generate_uniform_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        character_options = _load_json("character_options.json")
        uniform_options = _load_json("uniform_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        glamour_options_data = _load_json("glamour_options.json")
        ollama_models = cls._collect_ollama_models()

        style_options = [style_key for style_key in prompt_styles.keys()]
        glamour_options = glamour_options_data["glamour_options"]

        collections = uniform_options["collections"]
        collection_labels = [collection["label"] for collection in collections]
        silhouette_directions = uniform_options["silhouette_directions"]

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (style_options, {"default": "SDXL"}),
                "uniform_collection": (_with_random(collection_labels, allow_none=False), {"default": RANDOM_LABEL}),
                "silhouette_direction": (_with_random(silhouette_directions, allow_none=False), {"default": RANDOM_LABEL}),
                "glamour_enhancement": (_with_random(glamour_options, allow_none=False), {"default": RANDOM_LABEL}),
                "gender": (_with_random(character_options["gender"]), {"default": RANDOM_LABEL}),
                "include_environment": ("BOOLEAN", {"default": True}),
                "include_follow_up": ("BOOLEAN", {"default": True}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            }
        }

    def generate_uniform_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        uniform_collection: str,
        silhouette_direction: str,
        glamour_enhancement: str,
        gender: str,
        include_environment: bool,
        include_follow_up: bool,
        custom_text: str,
        seed: int = 0,
    ) -> Tuple[str, str]:
        character_options = _load_json("character_options.json")
        uniform_options = _load_json("uniform_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        glamour_options_data = _load_json("glamour_options.json")

        collections = uniform_options["collections"]
        collection_map = {collection["label"]: collection for collection in collections}
        collection_labels = [collection["label"] for collection in collections]
        silhouette_directions = uniform_options["silhouette_directions"]
        glamour_options = glamour_options_data["glamour_options"]

        rng = random.Random(seed)

        resolved_collection_label = _choose(uniform_collection, collection_labels, rng)
        resolved_silhouette = _choose(silhouette_direction, silhouette_directions, rng)
        resolved_glamour = _choose(glamour_enhancement, glamour_options, rng)
        resolved_gender = _choose(gender, character_options["gender"], rng)

        if resolved_collection_label is None:
            resolved_collection_label = collection_labels[0]

        collection = collection_map[resolved_collection_label]

        gender_prefix = f"{resolved_gender} " if resolved_gender else ""

        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        token_limit = style_config["token_limit"]

        capsule_lines = [
            f"Service context: {collection['service_context']}",
            f"Signature garments: {', '.join(collection['signature_garments'])}",
            f"Material palette: {', '.join(collection['material_palette'])}",
            f"Insignia highlights: {', '.join(collection['insignia_details'])}",
            f"Glamour accents: {', '.join(collection['glamour_accents'])}",
            f"Accessory focus: {', '.join(collection['accessory_focus'])}",
            f"Beauty direction: {', '.join(collection['beauty_direction'])}",
            f"Silhouette direction: {resolved_silhouette}",
            f"Glamour enhancement: {resolved_glamour}",
        ]

        if include_environment:
            capsule_lines.append(f"Environment staging: {', '.join(collection['environment_staging'])}")

        uniform_brief = "\n".join(capsule_lines)

        system_prompt = f"""You are a professional prompt engineer crafting luxury editorial descriptions for high-fashion uniform capsules. Keep responses under {token_limit} tokens.

RULES:
- Start with an evocative descriptor, never a model or style name such as Flux, SDXL, Qwen, HiDream.
- Highlight uniform authenticity while explaining couture-level tailoring and styling.
- Cover outfit construction, insignia, accessories, pose, lighting, and environment when supplied.
- Maintain respectful, aspirational language; avoid caricature or camp unless explicitly requested.
- Output only the final prompt—no lists, markdown, or meta commentary.
- Never include reasoning traces, deliberation markers, or any '<think>' text."""

        user_prompt_lines = [
            f"Design a {prompt_style} fashion image prompt for a {gender_prefix}model wearing the '{resolved_collection_label}' uniform capsule.",
            "Uniform capsule brief:",
            uniform_brief,
            "\nSynthesis instructions:",
            "- Translate the brief into a runway-ready ensemble with confident editorial pose.",
            "- Showcase materials, insignia, and glamour accents while preserving service credibility.",
            "- Integrate the selected silhouette direction and glamour enhancement.",
            "- Describe lighting and setting to reinforce the professional narrative.",
            "- Exclude negative prompts, markdown syntax, or explanatory commentary.",
        ]

        if custom_text.strip():
            user_prompt_lines.append(f"Additional user notes: {custom_text.strip()}")

        user_prompt_lines.append(f"\nFormat: {style_guidance}\nToken limit: {token_limit} tokens maximum")
        user_prompt = "\n".join(user_prompt_lines)

        generate_url = ollama_url
        if not generate_url.endswith("/api/generate"):
            generate_url = generate_url.rstrip("/") + "/api/generate"

        payload = {
            "model": ollama_model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": token_limit + 90,
                "temperature": 0.76,
            }
        }

        print(f"[UniformFashion] Generating prompt for capsule: {resolved_collection_label}")
        print(f"[UniformFashion] Silhouette: {resolved_silhouette}")
        print(f"[UniformFashion] Glamour: {resolved_glamour}")
        print(f"[UniformFashion] Gender: {resolved_gender or 'unspecified'}")
        print(f"[UniformFashion] Prompt style: {style_label} (max {token_limit} tokens)")
        print(f"[UniformFashion] Model: {ollama_model}")

        response = self._invoke_ollama(generate_url, payload)

        if not response or response.startswith("[ERROR"):
            error_msg = f"Failed to generate uniform prompt: {response}"
            return (error_msg, error_msg)

        follow_up = "\n".join(collection.get("follow_up", [])) if include_follow_up else ""
        preview = response if not follow_up else f"{response}\n\nUniform follow-up refinements:\n{follow_up}"
        return (response, preview)

    @staticmethod
    def _invoke_ollama(ollama_url: str, payload: Dict) -> Optional[str]:
        if requests is None:
            raise RuntimeError("'requests' is required for Ollama integration. Install optional dependencies.")
        try:
            print(f"[UniformFashionNode] Sending request to {ollama_url}")
            print(f"[UniformFashionNode] Model: {payload.get('model')}")

            response = requests.post(ollama_url, json=payload, timeout=120)

            print(f"[UniformFashionNode] Response status: {response.status_code}")

            response.raise_for_status()
            data = response.json()

            result = (data.get("response") or "").strip()
            print(f"[UniformFashionNode] Received response ({len(result)} chars): {result[:100]}...")

            if not result:
                print("[UniformFashionNode] WARNING: Empty response from Ollama")
                return "[Empty response from LLM]"

            return result
        except requests.exceptions.HTTPError as exc:
            error_msg = f"[UniformFashionNode] HTTP error: {exc}"
            print(error_msg)
            if hasattr(exc.response, 'text'):
                print(f"[UniformFashionNode] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            error_msg = f"[UniformFashionNode] Connection error: {exc}"
            print(error_msg)
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            error_msg = f"[UniformFashionNode] Timeout error: {exc}"
            print(error_msg)
            return "[ERROR: Request timed out]"
        except Exception as exc:
            error_msg = f"[UniformFashionNode] Error invoking Ollama: {exc}"
            print(error_msg)
            return f"[ERROR: {str(exc)}]"

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        if requests is None:
            return ["install_requests_library"]
        try:
            tags_url = f"{ollama_url}/api/tags"
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            all_models = [model["name"] for model in models_data.get("models", [])]

            if not all_models:
                return ["no_models_found"]

            return all_models
        except requests.exceptions.ConnectionError:
            return ["ollama_not_running"]
        except requests.exceptions.Timeout:
            return ["ollama_timeout"]
        except Exception as exc:
            print(f"[UniformFashionNode] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "WizdroidUniformFashion": UniformFashionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidUniformFashion": "Uniform Fashion",
}
