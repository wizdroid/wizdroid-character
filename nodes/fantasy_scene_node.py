import json
import random
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


class FantasySceneBuilder:
    CATEGORY = "Wizdroid/fantasy"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "preview")
    FUNCTION = "build_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        option_map = _load_json("fantasy_options.json")
        prompt_styles = _load_json("prompt_styles.json")
        ollama_models = cls._collect_ollama_models()

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (tuple(prompt_styles.keys()), {"default": "SDXL"}),
                "fantasy_theme": (_with_random(option_map["fantasy_theme"]), {"default": RANDOM_LABEL}),
                "subject_type": (_with_random(option_map["subject_type"]), {"default": RANDOM_LABEL}),
                "environment": (_with_random(option_map["environment"]), {"default": RANDOM_LABEL}),
                "atmosphere": (_with_random(option_map["atmosphere"]), {"default": RANDOM_LABEL}),
                "lighting": (_with_random(option_map["lighting"]), {"default": RANDOM_LABEL}),
                "visual_style": (_with_random(option_map["visual_style"]), {"default": RANDOM_LABEL}),
                "texture": (_with_random(option_map["texture"]), {"default": RANDOM_LABEL}),
                "composition": (_with_random(option_map["composition"]), {"default": RANDOM_LABEL}),
                "special_elements": (_with_random(option_map["special_elements"]), {"default": RANDOM_LABEL}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def build_prompt(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        fantasy_theme: str,
        subject_type: str,
        environment: str,
        atmosphere: str,
        lighting: str,
        visual_style: str,
        texture: str,
        composition: str,
        special_elements: str,
        custom_text: str,
    ):
        option_map = _load_json("fantasy_options.json")
        prompt_styles = _load_json("prompt_styles.json")

        resolved = {
            "fantasy_theme": _choose(fantasy_theme, option_map["fantasy_theme"]),
            "subject_type": _choose(subject_type, option_map["subject_type"]),
            "environment": _choose(environment, option_map["environment"]),
            "atmosphere": _choose(atmosphere, option_map["atmosphere"]),
            "lighting": _choose(lighting, option_map["lighting"]),
            "visual_style": _choose(visual_style, option_map["visual_style"]),
            "texture": _choose(texture, option_map["texture"]),
            "composition": _choose(composition, option_map["composition"]),
            "special_elements": _choose(special_elements, option_map["special_elements"]),
        }

        style_meta = prompt_styles[prompt_style]
        llm_response = self._invoke_llm(
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            prompt_style=prompt_style,
            style_meta=style_meta,
            selections=resolved,
            custom_text=custom_text.strip(),
        )

        negative_prompt = style_meta.get("negative_prompt", "")
        return llm_response, negative_prompt, llm_response

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        """
        Fetch available Ollama models from the API using HTTP requests.
        This is the proper way to query Ollama models as shown in ollama_base.py reference.
        """
        try:
            if requests is None:
                print("[FantasySceneBuilder] 'requests' library not installed")
                return ["install_requests_library"]
            
            # Query the /api/tags endpoint to get available models
            tags_url = f"{ollama_url}/api/tags"
            print(f"[FantasySceneBuilder] Querying Ollama at: {tags_url}")
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            print(f"[FantasySceneBuilder] Found {len(models)} Ollama models: {models}")
            return models if models else ["no_models_found"]
        except requests.exceptions.ConnectionError as e:
            print(f"[FantasySceneBuilder] Cannot connect to Ollama at {ollama_url}: {e}")
            return ["ollama_not_running"]
        except requests.exceptions.Timeout as e:
            print(f"[FantasySceneBuilder] Ollama request timeout: {e}")
            return ["ollama_timeout"]
        except Exception as e:
            print(f"[FantasySceneBuilder] Error fetching Ollama models: {type(e).__name__}: {e}")
            return ["ollama_error"]

    @staticmethod
    def _invoke_llm(ollama_url: str, ollama_model: str, prompt_style: str, style_meta: Dict, selections: Dict, custom_text: str) -> str:
        """
        Invoke Ollama LLM using HTTP API to generate fantasy scene prompts.
        """
        token_limit = style_meta.get('token_limit', 200)
        
        # Check if subject is "no subject" to determine focus
        has_subject = selections.get("subject_type", "no subject") != "no subject"
        
        if has_subject:
            system_prompt = (
                "You are a dark fantasy and horror prompt engineer for text-to-image AI. Create vivid, atmospheric prompts "
                f"that evoke haunting scenes with characters or creatures. Keep output under {token_limit} tokens. "
                "Use sensory details, texture descriptions, and atmospheric elements. Balance subject description with environment."
            )
        else:
            system_prompt = (
                "You are a fantasy scene prompt engineer for text-to-image AI. Create vivid, atmospheric prompts "
                f"that evoke otherworldly scenes. Keep output under {token_limit} tokens. Use sensory details, "
                "texture descriptions, and atmospheric elements. Focus on environment and mood, not characters."
            )

        # Build attribute list, filtering out None values
        attr_parts = []
        for key, value in selections.items():
            if value is None:
                continue  # Skip None values
            if key == "subject_type" and value == "no subject":
                continue  # Skip "no subject" in the attribute list
            attr_parts.append(f"{key.replace('_', ' ')}: {value}")
        
        lines = [
            f"Create a {prompt_style} fantasy scene prompt using these elements:",
            ", ".join(attr_parts),
        ]
        
        if custom_text:
            lines.append(f"\nAdditional notes: {custom_text}")

        lines.extend([
            f"\nFormat: {style_meta.get('guidance', 'Semicolon-separated descriptive clauses')}",
            f"Token limit: {token_limit} tokens maximum",
            "Include: textures, lighting, atmosphere, special effects, composition",
            "Exclude: negative prompt content, markdown, explanations",
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
            print(f"[FantasySceneBuilder] Error invoking LLM: {e}")
            return f"[Error: {str(e)}]"


NODE_CLASS_MAPPINGS = {
    "FantasySceneBuilder": FantasySceneBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FantasySceneBuilder": "Fantasy Scene Builder",
}
