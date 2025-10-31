import base64
import io
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

TorchTensor = Any

from PIL import Image

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
VISION_KEYWORDS = {
    "vision",
    "vl",
    "llava",
    "bakllava",
    "moondream",
    "cogvlm",
    "blip",
    "instructblip",
    "minigpt",
    "mplug",
    "qwen-vl",
    "florence",
    "idefics",
    "fuyu",
}


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


def _extract_tensor(image_input) -> Optional[TorchTensor]:
    if torch is None:
        raise RuntimeError("'torch' is required to process image tensors. Install optional dependencies.")
    if image_input is None:
        return None
    if isinstance(image_input, dict):
        image_input = image_input.get("image")
    if isinstance(image_input, (list, tuple)):
        if not image_input:
            return None
        image_input = image_input[0]
    if isinstance(image_input, torch.Tensor):
        return image_input
    return None


def _tensor_to_pil(tensor: TorchTensor) -> Optional[Image.Image]:
    if torch is None:
        raise RuntimeError("'torch' is required to process image tensors. Install optional dependencies.")
    if tensor is None:
        return None
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        return None
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[-1] not in (1, 3, 4):
        return None
    array = tensor.clamp(0, 1).mul(255).byte().numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def _image_to_base64(image_input) -> Optional[str]:
    tensor = _extract_tensor(image_input)
    if tensor is None:
        return None
    pil_img = _tensor_to_pil(tensor)
    if pil_img is None:
        return None
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _describe_subject(gender: Optional[str], age: Optional[str]) -> str:
    noun = "person"
    if gender:
        lower = gender.lower()
        if any(term in lower for term in ["female", "woman", "girl"]):
            noun = "woman"
        elif any(term in lower for term in ["male", "man", "boy"]):
            noun = "man"
        elif "non-binary" in lower or "gender" in lower:
            noun = f"{gender} person"
        else:
            noun = f"{gender} person"
    parts = []
    if age:
        parts.append(age)
    parts.append(noun)
    subject_phrase = " ".join(parts).strip()
    if not subject_phrase:
        subject_phrase = "person"
    return subject_phrase


class PoseExtractionNode:
    """
    ComfyUI node to extract pose descriptions from character images using Ollama vision models.
    Analyzes an image to generate detailed pose prompts for text-to-image generation.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pose_prompt",)
    FUNCTION = "extract_pose"

    @classmethod
    def INPUT_TYPES(cls):
        option_map = _load_json("character_options.json")
        ollama_models = cls._collect_ollama_models()

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "style": (_with_random(option_map["image_category"]), {"default": RANDOM_LABEL}),
                "gender": (_with_random(option_map["gender"]), {"default": RANDOM_LABEL}),
                "age_group": (_with_random(option_map["age_group"]), {"default": RANDOM_LABEL}),
                "character_image": ("IMAGE",),
                "custom_prompt_1": ("STRING", {"default": "", "multiline": True}),
                "custom_prompt_2": ("STRING", {"default": "", "multiline": True}),
                "custom_prompt_3": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def extract_pose(
        self,
        ollama_url: str,
        ollama_model: str,
        style: str,
        gender: str,
        age_group: str,
        character_image,
        custom_prompt_1: str = "",
        custom_prompt_2: str = "",
        custom_prompt_3: str = "",
    ) -> Tuple[str]:
        option_map = _load_json("character_options.json")

        resolved_style = _choose(style, option_map["image_category"]) if style else None
        resolved_gender = _choose(gender, option_map["gender"])
        resolved_age = _choose(age_group, option_map["age_group"])

        subject_phrase = _describe_subject(resolved_gender, resolved_age)
        style_phrase = resolved_style or "photorealistic"

        character_b64 = _image_to_base64(character_image)

        print(f"[PoseExtractionNode] Image conversion results:")
        print(f"  - Character: {len(character_b64) if character_b64 else 0} chars")

        if not character_b64:
            raise ValueError("Character image must be provided and valid")

        # Analyze character image
        print(f"[PoseExtractionNode] Analyzing character image...")
        
        character_desc = self._analyze_single_image(
            ollama_url, ollama_model, character_b64,
            "Describe the body pose, stance, posture, limb placement, hand gestures, body position, camera angle, and framing in this image. Be specific and detailed."
        )
        
        if character_desc.startswith("[ERROR"):
            return (f"Failed to analyze image: {character_desc}",)
        
        print(f"[PoseExtractionNode] Analysis complete")
        print(f"  - Character pose: {character_desc[:80]}...")
        
        # Generate final pose description prompt
        gender_req = f"- Gender: {resolved_gender}\n" if resolved_gender else ""
        age_req = f"- Age: {resolved_age}\n" if resolved_age else ""
        
        synthesis_prompt = f"""Based on the following pose analysis, create a concise text-to-image prompt describing the body pose and positioning.

Pose analysis: {character_desc}

Requirements:
- Start with 'the {subject_phrase} in'
- Focus on the body pose, stance, posture, limb placement, hand gestures, and camera angle
- Be specific about body positioning and framing
- Style: {style_phrase}
{gender_req}{age_req}- Keep within 100 tokens
- No meta commentary, just the pose description"""

        system_prompt = (
            "You are a text-to-image prompt engineer specializing in body pose and positioning descriptions. "
            "Create concise, detailed prompts focusing on stance, posture, limb placement, and camera angles. "
            "Output only the prompt, no explanations."
        )
        
        # Ensure URL has the /api/generate endpoint for synthesis
        synthesis_url = ollama_url
        if not synthesis_url.endswith("/api/generate"):
            synthesis_url = synthesis_url.rstrip("/") + "/api/generate"

        payload = {
            "model": ollama_model,
            "prompt": synthesis_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": 200,
                "temperature": 0.7,
            }
        }

        print(f"[PoseExtractionNode] Calling Ollama vision model: {ollama_model}")
        response = self._invoke_ollama(synthesis_url, payload)
        
        # Concatenate custom prompts at the beginning
        custom_parts = []
        if custom_prompt_1.strip():
            custom_parts.append(custom_prompt_1.strip())
        if custom_prompt_2.strip():
            custom_parts.append(custom_prompt_2.strip())
        if custom_prompt_3.strip():
            custom_parts.append(custom_prompt_3.strip())
        
        if custom_parts:
            custom_prefix = ", ".join(custom_parts)
            final_prompt = f"{custom_prefix}, {response}" if response else custom_prefix
            print(f"[PoseExtractionNode] Added custom prompts at beginning")
            return (final_prompt,)
        
        return (response or "",)
    
    def _analyze_single_image(
        self,
        ollama_url: str,
        ollama_model: str,
        image_b64: str,
        prompt: str
    ) -> str:
        """Analyze a single image with vision model."""
        # Ensure URL has the /api/generate endpoint
        if not ollama_url.endswith("/api/generate"):
            ollama_url = ollama_url.rstrip("/") + "/api/generate"
        
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "images": [image_b64],
            "options": {
                "num_predict": 150,
                "temperature": 0.5,
            }
        }
        
        return self._invoke_ollama(ollama_url, payload) or "[ERROR: No response]"

    @staticmethod
    def _invoke_ollama(ollama_url: str, payload: Dict) -> Optional[str]:
        if requests is None:
            raise RuntimeError("'requests' is required for Ollama integration. Install optional dependencies.")
        try:
            print(f"[PoseExtractionNode] Sending request to {ollama_url}")
            print(f"[PoseExtractionNode] Model: {payload.get('model')}")
            print(f"[PoseExtractionNode] Images count: {len(payload.get('images', []))}")
            
            response = requests.post(ollama_url, json=payload, timeout=120)
            
            print(f"[PoseExtractionNode] Response status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
            result = (data.get("response") or "").strip()
            print(f"[PoseExtractionNode] Received response ({len(result)} chars): {result[:100]}...")
            
            if not result:
                print("[PoseExtractionNode] WARNING: Empty response from Ollama")
                return "[Empty response from vision model]"
            
            return result
        except requests.exceptions.HTTPError as exc:
            error_msg = f"[PoseExtractionNode] HTTP error: {exc}"
            print(error_msg)
            if hasattr(exc.response, 'text'):
                print(f"[PoseExtractionNode] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            error_msg = f"[PoseExtractionNode] Connection error: {exc}"
            print(error_msg)
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            error_msg = f"[PoseExtractionNode] Timeout error: {exc}"
            print(error_msg)
            return "[ERROR: Request timed out]"
        except Exception as exc:
            error_msg = f"[PoseExtractionNode] Error invoking Ollama: {exc}"
            print(error_msg)
            return f"[ERROR: {str(exc)}]"

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        if requests is None:
            return ["install_requests_library"]
        try:
            tags_url = f"{ollama_url}/api/tags"
            models = PoseExtractionNode._fetch_filter_models(tags_url)
            if models:
                return models

            if PoseExtractionNode._pull_florence_model():
                models = PoseExtractionNode._fetch_filter_models(tags_url)
                if models:
                    return models

            return ["no_vision_models"]
        except requests.exceptions.ConnectionError:
            return ["ollama_not_running"]
        except requests.exceptions.Timeout:
            return ["ollama_timeout"]
        except Exception as exc:
            print(f"[PoseExtractionNode] Error fetching Ollama models: {exc}")
            return ["ollama_error"]

    @staticmethod
    def _fetch_filter_models(tags_url: str) -> List[str]:
        response = requests.get(tags_url, timeout=5)
        response.raise_for_status()
        models_data = response.json()
        all_models = [model["name"] for model in models_data.get("models", [])]
        if not all_models:
            return []

        vision_models: List[str] = []
        for name in all_models:
            lower = name.lower()
            if any(keyword in lower for keyword in VISION_KEYWORDS):
                vision_models.append(name)

        if vision_models:
            return vision_models

        # fallback: return all if none matched, but mark empty to trigger pull
        return []

    @staticmethod
    def _pull_florence_model() -> bool:
        try:
            completed = subprocess.run(
                ["ollama", "pull", "florence"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print("[ReferenceFusionPromptBuilder] Ollama executable not found; cannot pull florence model")
            return False

        if completed.returncode != 0:
            print(
                "[ReferenceFusionPromptBuilder] Failed to pull florence model: "
                f"{completed.stderr.strip() or 'unknown error'}"
            )
            return False

        print("[ReferenceFusionPromptBuilder] Successfully pulled florence model")
        return True


NODE_CLASS_MAPPINGS = {
    "WizdroidPoseExtraction": PoseExtractionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidPoseExtraction": "Pose Extraction (Wizdroid)",
}
