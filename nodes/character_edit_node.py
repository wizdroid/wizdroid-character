import json
import random
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


def _load_json(name: str) -> Dict:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _with_random(options: List[str]) -> Tuple[str, ...]:
    return tuple([RANDOM_LABEL, NONE_LABEL] + options)


def _choose(value: str, options: List[str]) -> Optional[str]:
    if value == RANDOM_LABEL:
        return random.choice(options)
    if value == NONE_LABEL:
        return None
    return value


class CharacterEditNode:
    """
    ComfyUI node to generate prompts for editing character images, focusing on
    face angles, camera angles, and poses using Flux Kontext or Qwen Image Edit models.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("edit_prompt", "preview")
    FUNCTION = "generate_edit_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        character_options = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        ollama_models = cls._collect_ollama_models()

        style_options = [style_key for style_key in prompt_styles.keys()]

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (style_options, {"default": "SDXL"}),
                "retain_face": ("BOOLEAN", {"default": True}),
                "target_face_angle": (_with_random(character_options["face_angle"]), {"default": RANDOM_LABEL}),
                "target_camera_angle": (_with_random(character_options["camera_angle"]), {"default": RANDOM_LABEL}),
                "target_pose": (_with_random(character_options["pose_style"]), {"default": RANDOM_LABEL}),
                "gender": (_with_random(character_options["gender"]), {"default": RANDOM_LABEL}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "preview_prompt": ("STRING", {"widget": "text", "multiline": True, "readonly": True, "default": "Generated prompt will appear here..."}),
            }
        }

    def generate_edit_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        retain_face: bool,
        target_face_angle: str,
        target_camera_angle: str,
        target_pose: str,
        gender: str,
        custom_text: str,
    ) -> Tuple[str]:
        character_options = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")

        resolved_face_angle = _choose(target_face_angle, character_options["face_angle"])
        resolved_camera_angle = _choose(target_camera_angle, character_options["camera_angle"])
        resolved_pose = _choose(target_pose, character_options["pose_style"])
        resolved_gender = _choose(gender, character_options["gender"])

        # Get style configuration
        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        token_limit = style_config["token_limit"]

        # Build the prompt instruction for the LLM
        if retain_face:
            system_prompt = (
                "You are a text-to-image prompt engineer for face-preserving image editing models (Flux Kontext, Qwen Image Edit). "
                f"Create concise prompts under {token_limit} tokens that preserve the original face while modifying other aspects. "
                "ALWAYS start prompts with 'Retain the facial features from the original image.' Then describe angle, pose, and positioning changes. "
                "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', or any meta preface. "
                "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s)."
            )
        else:
            system_prompt = (
                "You are a text-to-image prompt engineer for image editing models (Flux Kontext, Qwen Image Edit). "
                f"Create concise prompts under {token_limit} tokens for editing character images. "
                "Focus on face angles, camera angles, and poses. Be specific and descriptive but avoid excessive verbosity. "
                "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', or any meta preface. "
                "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s)."
            )

        # Build attribute list, filtering out None values
        attr_parts = []
        if resolved_face_angle:
            attr_parts.append(f"face angle: {resolved_face_angle}")
        if resolved_camera_angle:
            attr_parts.append(f"camera angle: {resolved_camera_angle}")
        if resolved_pose:
            attr_parts.append(f"pose: {resolved_pose}")
        if resolved_gender:
            attr_parts.append(f"gender: {resolved_gender}")

        if retain_face:
            lines = [
                f"Create a {prompt_style} face-preserving image edit prompt for changing angles and pose:",
                ", ".join(attr_parts),
            ]
        else:
            lines = [
                f"Create a {prompt_style} image edit prompt for changing character angles and pose:",
                ", ".join(attr_parts),
            ]

        if custom_text.strip():
            lines.append(f"\nAdditional notes: {custom_text.strip()}")

        if retain_face:
            lines.extend([
                f"\nFormat: {style_guidance}",
                f"Token limit: {token_limit} tokens maximum",
                "CRITICAL - Start with: 'Retain the facial features from the original image.'",
                "Then describe:",
                "  * New face angle and head positioning",
                "  * New camera angle and perspective",
                "  * New pose and body positioning",
                "Do NOT describe facial features (eyes, nose, mouth, face shape) - these are preserved from original",
                "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', or similar",
                "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
                "Output only the final prompt text:"
            ])
        else:
            lines.extend([
                f"\nFormat: {style_guidance}",
                f"Token limit: {token_limit} tokens maximum",
                "CRITICAL - Describe:",
                "  * Face angle and head positioning",
                "  * Camera angle and perspective",
                "  * Pose and body positioning",
                "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', or similar",
                "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
                "Output only the final prompt text:"
            ])

        user_prompt = "\n".join(lines)

        # Ensure URL has the /api/generate endpoint
        generate_url = ollama_url
        if not generate_url.endswith("/api/generate"):
            generate_url = generate_url.rstrip("/") + "/api/generate"

        payload = {
            "model": ollama_model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": token_limit + 50,
                "temperature": 0.8,
            }
        }

        print(f"[CharacterEdit] Generating edit prompt")
        print(f"[CharacterEdit] Retain face: {retain_face}")
        print(f"[CharacterEdit] Target face angle: {resolved_face_angle}")
        print(f"[CharacterEdit] Target camera angle: {resolved_camera_angle}")
        print(f"[CharacterEdit] Target pose: {resolved_pose}")
        print(f"[CharacterEdit] Gender: {resolved_gender}")
        print(f"[CharacterEdit] Prompt style: {style_label} (max {token_limit} tokens)")
        print(f"[CharacterEdit] Using model: {ollama_model}")

        response = self._invoke_ollama(generate_url, payload)

        if not response or response.startswith("[ERROR"):
            error_msg = f"Failed to generate prompt: {response}"
            return (error_msg, error_msg)

        return (response, response)

    @staticmethod
    def _invoke_ollama(ollama_url: str, payload: Dict) -> Optional[str]:
        if requests is None:
            raise RuntimeError("'requests' is required for Ollama integration. Install optional dependencies.")
        try:
            print(f"[CharacterEdit] Sending request to {ollama_url}")
            print(f"[CharacterEdit] Model: {payload.get('model')}")

            response = requests.post(ollama_url, json=payload, timeout=120)

            print(f"[CharacterEdit] Response status: {response.status_code}")

            response.raise_for_status()
            data = response.json()

            result = (data.get("response") or "").strip()
            print(f"[CharacterEdit] Received response ({len(result)} chars): {result[:100]}...")

            if not result:
                print("[CharacterEdit] WARNING: Empty response from Ollama")
                return "[Empty response from LLM]"

            return result
        except requests.exceptions.HTTPError as exc:
            error_msg = f"[CharacterEdit] HTTP error: {exc}"
            print(error_msg)
            if hasattr(exc.response, 'text'):
                print(f"[CharacterEdit] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            error_msg = f"[CharacterEdit] Connection error: {exc}"
            print(error_msg)
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            error_msg = f"[CharacterEdit] Timeout error: {exc}"
            print(error_msg)
            return "[ERROR: Request timed out]"
        except Exception as exc:
            error_msg = f"[CharacterEdit] Error invoking Ollama: {exc}"
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
            print(f"[CharacterEdit] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "WizdroidCharacterEdit": CharacterEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidCharacterEdit": "Character Edit (Wizdroid)",
}