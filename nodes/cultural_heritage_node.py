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


def _with_random(options: List[str], allow_none: bool = False) -> Tuple[str, ...]:
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


class CulturalHeritageCapsuleNode:
    """Blend cultural heritage fashion with modern glamour for editorial prompts."""

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("heritage_prompt", "preview", "follow_up")
    FUNCTION = "generate_heritage_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        capsules_payload = _load_json("cultural_capsules.json")
        glamour_payload = _load_json("glamour_options.json")
        character_options = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        ollama_models = cls._collect_ollama_models()

        capsules = capsules_payload["capsules"]
        capsule_labels = [capsule["label"] for capsule in capsules]
        silhouette_directions = capsules_payload["silhouette_directions"]
        glamour_options = glamour_payload["glamour_options"]

        style_options = [style_key for style_key in prompt_styles.keys()]

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (style_options, {"default": "SDXL"}),
                "heritage_capsule": (_with_random(capsule_labels), {"default": RANDOM_LABEL}),
                "silhouette_direction": (_with_random(silhouette_directions, allow_none=False), {"default": RANDOM_LABEL}),
                "glamour_enhancement": (_with_random(glamour_options, allow_none=False), {"default": RANDOM_LABEL}),
                "gender": (_with_random(character_options["gender"]), {"default": RANDOM_LABEL}),
                "include_motifs": ("BOOLEAN", {"default": True}),
                "include_environment": ("BOOLEAN", {"default": True}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "widget": "seed"}),
            },
        }

    def generate_heritage_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        heritage_capsule: str,
        silhouette_direction: str,
        glamour_enhancement: str,
        gender: str,
        include_motifs: bool,
        include_environment: bool,
        custom_text: str,
        seed: int = 0,
    ) -> Tuple[str, str, str]:
        capsules_payload = _load_json("cultural_capsules.json")
        glamour_payload = _load_json("glamour_options.json")
        character_options = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")

        capsules = capsules_payload["capsules"]
        capsule_map = {capsule["label"]: capsule for capsule in capsules}
        capsule_labels = [capsule["label"] for capsule in capsules]
        glamour_options = glamour_payload["glamour_options"]
        silhouette_directions = capsules_payload["silhouette_directions"]
        gender_options = character_options["gender"]

        rng = random.Random(seed)

        resolved_capsule_label = _choose(heritage_capsule, capsule_labels, rng)
        resolved_silhouette = _choose(silhouette_direction, silhouette_directions, rng)
        resolved_glamour = _choose(glamour_enhancement, glamour_options, rng)
        resolved_gender = _choose(gender, gender_options, rng)

        if resolved_capsule_label is None:
            # Fallback to deterministic first capsule to avoid runtime errors.
            resolved_capsule_label = capsule_labels[0]

        capsule = capsule_map[resolved_capsule_label]

        gender_prefix = f"{resolved_gender} " if resolved_gender else ""

        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        token_limit = style_config["token_limit"]

        capsule_lines = [
            f"Heritage focus: {capsule['heritage_context']}",
            f"Signature garments: {', '.join(capsule['signature_garments'])}",
            f"Traditional materials: {', '.join(capsule['traditional_materials'])}",
            f"Ornamentation cues: {', '.join(capsule['ornamentation'])}",
            f"Symbolic colors: {', '.join(capsule['symbolic_colors'])}",
            f"Modern infusions: {', '.join(capsule['modern_fashion_infusions'])}",
            f"Silhouette direction: {resolved_silhouette}",
            f"Glamour enhancement: {resolved_glamour}",
        ]

        if include_motifs:
            capsule_lines.append(f"Motif emphasis: {', '.join(capsule['fusion_strategies'])}")
            capsule_lines.append(f"Accessory focus: {', '.join(capsule['accessory_focus'])}")

        if include_environment:
            capsule_lines.append(f"Environment staging: {', '.join(capsule['environment_staging'])}")

        heritage_brief = "\n".join(capsule_lines)

        system_prompt = f"""You are a couture fashion prompt engineer. Blend cultural heritage accuracy with contemporary high fashion while respecting the source culture. Keep responses under {token_limit} tokens.

RULES:
- Begin the prompt with a vivid descriptor (never the model/style name such as Flux, SDXL, Qwen, HiDream).
- Reference key heritage garments, materials, and motifs with reverence.
- Explain how modern silhouettes, tailoring, and glamour reinterpret the heritage pieces.
- Always include outfit specifics, accessories, pose, lighting, and setting when requested.
- Avoid stereotypes, costuming clichés, or trivializing language.
- Output only the final prompt text, no explanations, headings, bullet markers, or meta commentary.
- Never include reasoning traces, deliberation markers, or text inside '<think>' tags."""

        user_prompt_lines = [
            f"Create a {prompt_style} fashion image prompt for a {gender_prefix}model inspired by the '{resolved_capsule_label}' capsule.",
            "Heritage brief:",
            heritage_brief,
            "\nSynthesis instructions:",
            "- Fuse the listed heritage elements with the modern silhouette and glamour direction.",
            "- Highlight textiles, tailoring, and accessories that honor the culture while feeling runway ready.",
            "- Maintain a polished editorial tone with confident posture and cinematic lighting.",
            "- Ensure the description feels luxurious, respectful, and contemporary.",
            "- Exclude negative prompts, markdown, lists, or explanatory text.",
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
                "num_predict": token_limit + 120,
                "temperature": 0.7,
            }
        }

        print(f"[HeritageCapsule] Generating prompt for capsule: {resolved_capsule_label}")
        print(f"[HeritageCapsule] Silhouette: {resolved_silhouette}")
        print(f"[HeritageCapsule] Glamour: {resolved_glamour}")
        print(f"[HeritageCapsule] Gender: {resolved_gender or 'unspecified'}")
        print(f"[HeritageCapsule] Prompt style: {style_label} (max {token_limit} tokens)")
        print(f"[HeritageCapsule] Model: {ollama_model}")

        response = self._invoke_ollama(generate_url, payload)

        if not response or response.startswith("[ERROR"):
            error_msg = f"Failed to generate heritage prompt: {response}"
            return (error_msg, error_msg, error_msg)

        follow_up = "\n".join(capsule.get("follow_up", [])) if capsule.get("follow_up") else ""
        return (response, response, follow_up)

    @staticmethod
    def _invoke_ollama(ollama_url: str, payload: Dict) -> Optional[str]:
        if requests is None:
            raise RuntimeError("'requests' is required for Ollama integration. Install optional dependencies.")
        try:
            print(f"[HeritageCapsule] Sending request to {ollama_url}")
            response = requests.post(ollama_url, json=payload, timeout=120)
            print(f"[HeritageCapsule] Response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            result = (data.get("response") or "").strip()
            print(f"[HeritageCapsule] Received response ({len(result)} chars)")
            if not result:
                print("[HeritageCapsule] WARNING: Empty response from Ollama")
                return "[Empty response from LLM]"
            return result
        except requests.exceptions.HTTPError as exc:
            print(f"[HeritageCapsule] HTTP error: {exc}")
            if hasattr(exc.response, "text"):
                print(f"[HeritageCapsule] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            print(f"[HeritageCapsule] Connection error: {exc}")
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            print(f"[HeritageCapsule] Timeout error: {exc}")
            return "[ERROR: Request timed out]"
        except Exception as exc:
            print(f"[HeritageCapsule] Error invoking Ollama: {exc}")
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
            models = [model["name"] for model in models_data.get("models", [])]
            return models if models else ["no_models_found"]
        except requests.exceptions.ConnectionError:
            return ["ollama_not_running"]
        except requests.exceptions.Timeout:
            return ["ollama_timeout"]
        except Exception as exc:
            print(f"[HeritageCapsule] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "WizdroidCulturalHeritageCapsule": CulturalHeritageCapsuleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidCulturalHeritageCapsule": "Cultural Heritage Capsule (Wizdroid)",
}
