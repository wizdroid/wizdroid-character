import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import requests
except ImportError:
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


def _choose(value: str, options: List[str]) -> str:
    if value == RANDOM_LABEL:
        return random.choice(options)
    if value == NONE_LABEL:
        return None
    return value


class CharacterPromptBuilder:
    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "follow_up")
    FUNCTION = "build_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        option_map = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        ollama_models = cls._collect_ollama_models()

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (tuple(prompt_styles.keys()), {"default": "SDXL"}),
                "retain_face": ("BOOLEAN", {"default": False}),
                "image_category": (_with_random(option_map["image_category"]), {"default": RANDOM_LABEL}),
                "gender": (_with_random(option_map["gender"]), {"default": RANDOM_LABEL}),
                "age_group": (_with_random(option_map["age_group"]), {"default": RANDOM_LABEL}),
                "body_type": (_with_random(option_map["body_type"]), {"default": RANDOM_LABEL}),
                "hair_color": (_with_random(option_map["hair_color"]), {"default": RANDOM_LABEL}),
                "hair_style": (_with_random(option_map["hair_style"]), {"default": RANDOM_LABEL}),
                "eye_color": (_with_random(option_map["eye_color"]), {"default": RANDOM_LABEL}),
                "facial_expression": (_with_random(option_map["facial_expression"]), {"default": RANDOM_LABEL}),
                "face_angle": (_with_random(option_map["face_angle"]), {"default": RANDOM_LABEL}),
                "camera_angle": (_with_random(option_map["camera_angle"]), {"default": RANDOM_LABEL}),
                "pose_style": (_with_random(option_map["pose_style"]), {"default": RANDOM_LABEL}),
                "makeup_style": (_with_random(option_map["makeup_style"]), {"default": RANDOM_LABEL}),
                "fashion_style": (_with_random(option_map["fashion_style"]), {"default": RANDOM_LABEL}),
                "background_style": (_with_random(option_map["background_style"]), {"default": RANDOM_LABEL}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def build_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        retain_face: bool,
        image_category: str,
        gender: str,
        age_group: str,
        body_type: str,
        hair_color: str,
        hair_style: str,
        eye_color: str,
        facial_expression: str,
        face_angle: str,
        camera_angle: str,
        pose_style: str,
        makeup_style: str,
        fashion_style: str,
        background_style: str,
        custom_text: str,
    ):
        option_map = _load_json("character_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        followup = self._pick_followup_questions()

        resolved = {
            "image_category": _choose(image_category, option_map["image_category"]),
            "gender": _choose(gender, option_map["gender"]),
            "age_group": _choose(age_group, option_map["age_group"]),
            "body_type": _choose(body_type, option_map["body_type"]),
            "hair_color": _choose(hair_color, option_map["hair_color"]),
            "hair_style": _choose(hair_style, option_map["hair_style"]),
            "eye_color": _choose(eye_color, option_map["eye_color"]),
            "facial_expression": _choose(facial_expression, option_map["facial_expression"]),
            "face_angle": _choose(face_angle, option_map["face_angle"]),
            "camera_angle": _choose(camera_angle, option_map["camera_angle"]),
            "pose_style": _choose(pose_style, option_map["pose_style"]),
            "makeup_style": _choose(makeup_style, option_map["makeup_style"]),
            "fashion_style": _choose(fashion_style, option_map["fashion_style"]),
            "background_style": _choose(background_style, option_map["background_style"]),
        }

        style_meta = prompt_styles[prompt_style]
        llm_response = self._invoke_llm(
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            prompt_style=prompt_style,
            retain_face=retain_face,
            style_meta=style_meta,
            selections=resolved,
            custom_text=custom_text.strip(),
        )

        negative_prompt = style_meta.get("negative_prompt", "")
        return llm_response, negative_prompt, "\n".join(followup)

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        """
        Fetch available Ollama models from the API using HTTP requests.
        This is the proper way to query Ollama models as shown in ollama_base.py reference.
        """
        try:
            if requests is None:
                print("[CharacterPromptBuilder] 'requests' library not installed")
                return ["install_requests_library"]
            
            # Query the /api/tags endpoint to get available models
            tags_url = f"{ollama_url}/api/tags"
            print(f"[CharacterPromptBuilder] Querying Ollama at: {tags_url}")
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            print(f"[CharacterPromptBuilder] Found {len(models)} Ollama models: {models}")
            return models if models else ["no_models_found"]
        except requests.exceptions.ConnectionError as e:
            print(f"[CharacterPromptBuilder] Cannot connect to Ollama at {ollama_url}: {e}")
            return ["ollama_not_running"]
        except requests.exceptions.Timeout as e:
            print(f"[CharacterPromptBuilder] Ollama request timeout: {e}")
            return ["ollama_timeout"]
        except Exception as e:
            print(f"[CharacterPromptBuilder] Error fetching Ollama models: {type(e).__name__}: {e}")
            return ["ollama_error"]

    @staticmethod
    def _invoke_llm(ollama_url: str, ollama_model: str, prompt_style: str, retain_face: bool, style_meta: Dict, selections: Dict, custom_text: str) -> str:
        """
        Invoke Ollama LLM using HTTP API (proper method) instead of subprocess.
        """
        token_limit = style_meta.get('token_limit', 200)
        
        if retain_face:
            system_prompt = (
                "You are a text-to-image prompt engineer for face-preserving image editing models (Flux Kontext, Qwen Image Edit). "
                f"Create concise prompts under {token_limit} tokens that preserve the original face while modifying other aspects. "
                "ALWAYS start prompts with 'Retain the facial features from the original image.' Then describe outfit, pose, setting changes. "
                "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', or any meta preface. "
                "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s)."
            )
        else:
            system_prompt = (
                "You are a text-to-image prompt engineer. Create concise, vivid prompts that honor all specified attributes. "
                f"Keep output under {token_limit} tokens. Be specific and descriptive but avoid excessive verbosity. "
                "ALWAYS describe clothing details (garment type, color, fabric) and pose (body position, stance). "
                "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', or any meta preface. "
                "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s)."
            )

        # Build attribute list, filtering out None values
        attr_parts = []
        for key, value in selections.items():
            if value is not None:
                attr_parts.append(f"{key.replace('_', ' ')}: {value}")
        
        if retain_face:
            lines = [
                f"Create a {prompt_style} face-preserving image edit prompt using these attributes:",
                ", ".join(attr_parts),
            ]
        else:
            lines = [
                f"Create a {prompt_style} image generation prompt using these attributes:",
                ", ".join(attr_parts),
            ]
        
        if custom_text:
            lines.append(f"\nAdditional notes: {custom_text}")

        if retain_face:
            lines.extend([
                f"\nFormat: {style_meta.get('guidance', 'Single paragraph with comma-separated descriptors')}",
                f"Token limit: {token_limit} tokens maximum",
                "CRITICAL - Start with: 'Retain the facial features from the original image.'",
                "Then describe:",
                "  * New outfit/clothing: specific garment types, colors, fabrics, style details",
                "  * New pose: exact body position, limb placement, gesture, stance",
                "  * New setting/background, lighting, atmosphere",
                "Do NOT describe facial features (eyes, nose, mouth, face shape) - these are preserved from original",
                "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', or similar",
                "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
                "Output only the final prompt text:"
            ])
        else:
            lines.extend([
                f"\nFormat: {style_meta.get('guidance', 'Single paragraph with comma-separated descriptors')}",
                f"Token limit: {token_limit} tokens maximum",
                "CRITICAL - You MUST describe:",
                "  * Outfit/clothing: specific garment types, colors, fabrics, style details",
                "  * Pose: exact body position, limb placement, gesture, stance",
                "  * Lighting, atmosphere, camera angle",
                "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', or similar",
                "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
                "Output only the final prompt text:"
            ])

        user_prompt = "\n".join(lines)

        try:
            if requests is None:
                return "[Please install 'requests' library: pip install requests]"
            
            # Use the HTTP API endpoint for generation
            generate_url = f"{ollama_url}/api/generate"
            payload = {
                "model": ollama_model,
                "prompt": f"{user_prompt}",
                "system": system_prompt,
                "stream": False
            }
            
            response = requests.post(generate_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "[Empty response from Ollama]").strip()
            
        except requests.exceptions.ConnectionError:
            return "[Ollama server not running. Please start Ollama.]"
        except requests.exceptions.Timeout:
            return "[Ollama request timed out]"
        except Exception as e:
            print(f"[CharacterPromptBuilder] Error invoking LLM: {e}")
            return f"[Error: {str(e)}]"

    @staticmethod
    def _pick_followup_questions() -> List[str]:
        payload = _load_json("followup_questions.json")
        bag: List[str] = []
        for values in payload.values():
            bag.extend(values)
        random.shuffle(bag)
        return bag[:3] if bag else []


NODE_CLASS_MAPPINGS = {
    "CharacterPromptBuilder": CharacterPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterPromptBuilder": "Character Prompt Builder",
}
