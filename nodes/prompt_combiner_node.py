import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _load_json(name: str) -> Dict:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_token_limit(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


class PromptCombinerNode:
    """
    ComfyUI node to combine multiple text prompts into one coherent prompt
    using different model styles from prompt_styles.json.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("combined_prompt", "preview")
    FUNCTION = "combine_prompts"

    @classmethod
    def INPUT_TYPES(cls):
        prompt_styles = _load_json("prompt_styles.json")
        ollama_models = cls._collect_ollama_models()

        style_options = [style_key for style_key in prompt_styles.keys()]

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (style_options, {"default": "SDXL"}),
                "input_prompt_1": ("STRING", {"multiline": True, "default": ""}),
                "input_prompt_2": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "input_prompt_3": ("STRING", {"multiline": True, "default": ""}),
                "input_prompt_4": ("STRING", {"multiline": True, "default": ""}),
                "input_prompt_5": ("STRING", {"multiline": True, "default": ""}),
                "custom_instructions": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def combine_prompts(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        input_prompt_1: str,
        input_prompt_2: str,
        input_prompt_3: str = "",
        input_prompt_4: str = "",
        input_prompt_5: str = "",
        custom_instructions: str = "",
        token_limit_override: str = "0",
    ) -> Tuple[str]:
        prompt_styles = _load_json("prompt_styles.json")

        # Collect all non-empty input prompts
        input_prompts = [
            prompt.strip() for prompt in [input_prompt_1, input_prompt_2, input_prompt_3, input_prompt_4, input_prompt_5]
            if prompt.strip()
        ]

        if len(input_prompts) < 2:
            return ("[ERROR: Need at least 2 input prompts to combine]",)

        # Get style configuration
        style_config = prompt_styles.get(prompt_style, prompt_styles["SDXL"])
        style_label = style_config["label"]
        style_guidance = style_config["guidance"]
        token_limit = style_config["token_limit"]

        # Build the prompt instruction for the LLM
        system_prompt = (
            "You are a text-to-image prompt engineer specializing in combining multiple prompts into coherent, unified descriptions. "
            "Create concise prompts that merge the provided prompts into one cohesive description. "
            "Maintain the artistic style and key elements from all input prompts while eliminating redundancy. "
            "Your first word must be a vivid descriptor (adjective or noun), never 'Here', 'This', 'Prompt', or any meta preface. "
            "Do not include introductions, explanations, or meta commentary—output only the usable prompt sentence(s). "
            "Never include reasoning traces, deliberation markers, or text enclosed in '<think>' or similar tags."
        )

        # Build the user prompt
        lines = [
            f"Combine these {len(input_prompts)} prompts into one coherent {prompt_style} prompt:",
        ]

        for i, prompt in enumerate(input_prompts, 1):
            lines.append(f"Prompt {i}: {prompt}")

        if custom_instructions.strip():
            lines.append(f"\nAdditional combination instructions: {custom_instructions.strip()}")

        lines.extend([
            f"\nFormat: {style_guidance}",
            "CRITICAL - Create one unified prompt that:",
            "  * Merges all key elements from input prompts",
            "  * Eliminates redundancy and contradictions",
            "  * Maintains artistic coherence",
            "  * Preserves important details from each prompt",
            "Begin output with a descriptive adjective or noun; never start with 'Here', 'Here's', 'This prompt', the model/style name (Flux, SDXL, Qwen, HiDream, etc.), or similar",
            "Exclude: negative prompt content, markdown, explanations, prefaces, or statements like 'Here is a prompt'",
            "Remove any reasoning or planning text; do not include '<think>' or similar tags",
            "Output only the final combined prompt text:"
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
                "temperature": 0.7,
            }
        }

        print(f"[PromptCombiner] Combining {len(input_prompts)} prompts")
        print(f"[PromptCombiner] Prompt style: {style_label}")
        print(f"[PromptCombiner] Using model: {ollama_model}")
        for i, prompt in enumerate(input_prompts, 1):
            print(f"[PromptCombiner] Input {i}: {prompt[:50]}...")

        response = self._invoke_ollama(generate_url, payload)

        if not response or response.startswith("[ERROR"):
            error_msg = f"Failed to combine prompts: {response}"
            return (error_msg, error_msg)

        return (response, response)

    @staticmethod
    def _invoke_ollama(ollama_url: str, payload: Dict) -> Optional[str]:
        if requests is None:
            raise RuntimeError("'requests' is required for Ollama integration. Install optional dependencies.")
        try:
            print(f"[PromptCombiner] Sending request to {ollama_url}")
            print(f"[PromptCombiner] Model: {payload.get('model')}")

            response = requests.post(ollama_url, json=payload, timeout=120)

            print(f"[PromptCombiner] Response status: {response.status_code}")

            response.raise_for_status()
            data = response.json()

            result = (data.get("response") or "").strip()
            print(f"[PromptCombiner] Received response ({len(result)} chars): {result[:100]}...")

            if not result:
                print("[PromptCombiner] WARNING: Empty response from Ollama")
                return "[Empty response from LLM]"

            return result
        except requests.exceptions.HTTPError as exc:
            error_msg = f"[PromptCombiner] HTTP error: {exc}"
            print(error_msg)
            if hasattr(exc.response, 'text'):
                print(f"[PromptCombiner] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            error_msg = f"[PromptCombiner] Connection error: {exc}"
            print(error_msg)
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            error_msg = f"[PromptCombiner] Timeout error: {exc}"
            print(error_msg)
            return "[ERROR: Request timed out]"
        except Exception as exc:
            error_msg = f"[PromptCombiner] Error invoking Ollama: {exc}"
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
            print(f"[PromptCombiner] Error fetching Ollama models: {exc}")
            return ["ollama_error"]


NODE_CLASS_MAPPINGS = {
    "WizdroidPromptCombiner": PromptCombinerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidPromptCombiner": "Prompt Combiner",
}