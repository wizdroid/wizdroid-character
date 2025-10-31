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


class PhotoAspectExtractorNode:
    """
    ComfyUI node to extract various aspects from images using Ollama vision models.
    Supports extracting clothes, pose, or style descriptions.
    """

    CATEGORY = "Wizdroid/character"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_prompt",)
    FUNCTION = "extract_aspect"

    # Extraction modes and their configurations
    EXTRACTION_MODES = {
        "clothes": {
            "analysis_prompt": "Describe the clothing, outfit, garments, colors, fabrics, accessories, and styling worn by the person in this image. Be specific and detailed.",
            "synthesis_start": "wearing",
            "synthesis_focus": "Focus on the clothing, outfit, colors, fabrics, and accessories",
            "system_prompt": "You are a text-to-image prompt engineer specializing in clothing and outfit descriptions. Create concise, detailed prompts focusing on garments, colors, fabrics, and styling. Output only the prompt, no explanations."
        },
        "pose": {
            "analysis_prompt": "Describe the body pose, stance, posture, limb placement, hand gestures, body position, camera angle, and framing in this image. Be specific and detailed.",
            "synthesis_start": "in",
            "synthesis_focus": "Focus on the body pose, stance, posture, limb placement, hand gestures, and camera angle",
            "system_prompt": "You are a text-to-image prompt engineer specializing in body pose and positioning descriptions. Create concise, detailed prompts focusing on stance, posture, limb placement, and camera angles. Output only the prompt, no explanations."
        },
        "style": {
            "analysis_prompt": "Describe the overall artistic style, visual aesthetic, lighting quality, color grading, mood, atmosphere, and photographic/artistic techniques in this image. Be specific and detailed.",
            "synthesis_start": "with",
            "synthesis_focus": "Focus on the artistic style, visual aesthetic, lighting quality, color grading, and overall mood",
            "system_prompt": "You are a text-to-image prompt engineer specializing in artistic style and visual aesthetics. Create concise, detailed prompts focusing on style, lighting, mood, and artistic techniques. Output only the prompt, no explanations."
        },
        "background": {
            "analysis_prompt": "Describe the background, environment, setting, location, scenery, architectural elements, natural features, and spatial context in this image. Be specific and detailed.",
            "synthesis_start": "with background of",
            "synthesis_focus": "Focus on the environment, setting, scenery, and spatial context",
            "system_prompt": "You are a text-to-image prompt engineer specializing in background and environment descriptions. Create concise, detailed prompts focusing on settings, locations, scenery, and spatial elements. Output only the prompt, no explanations."
        },
        "expression": {
            "analysis_prompt": "Describe the facial expression, emotions, gaze direction, eye contact, mood, facial features, and emotional state conveyed in this image. Be specific and detailed.",
            "synthesis_start": "with",
            "synthesis_focus": "Focus on the facial expression, emotions, gaze, and mood",
            "system_prompt": "You are a text-to-image prompt engineer specializing in facial expressions and emotions. Create concise, detailed prompts focusing on expressions, gaze, mood, and emotional states. Output only the prompt, no explanations."
        },
        "lighting": {
            "analysis_prompt": "Describe the lighting setup, light direction, quality (hard/soft), shadows, highlights, color temperature, time of day indicators, and overall illumination in this image. Be specific and detailed.",
            "synthesis_start": "lit with",
            "synthesis_focus": "Focus on the lighting direction, quality, shadows, highlights, and color temperature",
            "system_prompt": "You are a text-to-image prompt engineer specializing in lighting descriptions. Create concise, detailed prompts focusing on light sources, direction, quality, and illumination effects. Output only the prompt, no explanations."
        },
        "hair": {
            "analysis_prompt": "Describe the hairstyle, hair color, length, texture, styling, hair accessories, and grooming details in this image. Be specific and detailed.",
            "synthesis_start": "with hair",
            "synthesis_focus": "Focus on the hairstyle, color, length, texture, and styling",
            "system_prompt": "You are a text-to-image prompt engineer specializing in hair and hairstyle descriptions. Create concise, detailed prompts focusing on hair characteristics, styling, and accessories. Output only the prompt, no explanations."
        },
        "makeup": {
            "analysis_prompt": "Describe the makeup application, cosmetics style, colors used, techniques (eyes, lips, face, etc.), intensity, and overall beauty styling in this image. Be specific and detailed.",
            "synthesis_start": "with makeup",
            "synthesis_focus": "Focus on the makeup style, application, colors, and techniques",
            "system_prompt": "You are a text-to-image prompt engineer specializing in makeup and beauty descriptions. Create concise, detailed prompts focusing on cosmetics, application style, and beauty techniques. Output only the prompt, no explanations."
        },
        "accessories": {
            "analysis_prompt": "Describe the accessories, jewelry, props, items held or worn, bags, glasses, watches, and decorative elements in this image. Be specific and detailed.",
            "synthesis_start": "with",
            "synthesis_focus": "Focus on the accessories, jewelry, props, and decorative items",
            "system_prompt": "You are a text-to-image prompt engineer specializing in accessories and props descriptions. Create concise, detailed prompts focusing on jewelry, items, and decorative elements. Output only the prompt, no explanations."
        },
        "camera": {
            "analysis_prompt": "Describe the camera angle, lens type, focal length, depth of field, bokeh, perspective, framing, and photographic technique used in this image. Be specific and detailed.",
            "synthesis_start": "shot with",
            "synthesis_focus": "Focus on the camera angle, lens characteristics, depth of field, and photographic technique",
            "system_prompt": "You are a text-to-image prompt engineer specializing in camera and photography technical descriptions. Create concise, detailed prompts focusing on camera angles, lens properties, and photographic techniques. Output only the prompt, no explanations."
        },
        "composition": {
            "analysis_prompt": "Describe the composition, framing, rule of thirds application, depth layers, focal point, visual balance, negative space, and overall image structure in this image. Be specific and detailed.",
            "synthesis_start": "composed with",
            "synthesis_focus": "Focus on the framing, composition rules, depth, focal point, and visual balance",
            "system_prompt": "You are a text-to-image prompt engineer specializing in composition and framing descriptions. Create concise, detailed prompts focusing on compositional elements, balance, and visual structure. Output only the prompt, no explanations."
        },
        "color_palette": {
            "analysis_prompt": "Describe the color palette, dominant colors, color harmony, saturation levels, contrast, tonal range, and overall color scheme in this image. Be specific and detailed.",
            "synthesis_start": "with colors",
            "synthesis_focus": "Focus on the color palette, dominant hues, harmony, saturation, and contrast",
            "system_prompt": "You are a text-to-image prompt engineer specializing in color palette descriptions. Create concise, detailed prompts focusing on colors, harmony, saturation, and tonal relationships. Output only the prompt, no explanations."
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = cls._collect_ollama_models()

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "extraction_mode": (["clothes", "pose", "style", "background", "expression", "lighting", "hair", "makeup", "accessories", "camera", "composition", "color_palette"], {"default": "clothes"}),
                "character_image": ("IMAGE",),
                "custom_prompt_1": ("STRING", {"default": "Transfer the clothing from the person in the second image onto the person from the first image, while matching the pose of the person in the third image. Preserve the identity and facial features of the person in the first image.", "multiline": True}),
                "custom_prompt_2": ("STRING", {"default": "", "multiline": True}),
                "custom_prompt_3": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def extract_aspect(
        self,
        ollama_url: str,
        ollama_model: str,
        extraction_mode: str,
        character_image,
        custom_prompt_1: str = "",
        custom_prompt_2: str = "",
        custom_prompt_3: str = "",
    ) -> Tuple[str]:
        # Get mode configuration
        mode_config = self.EXTRACTION_MODES.get(extraction_mode, self.EXTRACTION_MODES["clothes"])

        character_b64 = _image_to_base64(character_image)

        print(f"[PhotoAspectExtractor] Image conversion results:")
        print(f"  - Character: {len(character_b64) if character_b64 else 0} chars")
        print(f"  - Extraction mode: {extraction_mode}")

        if not character_b64:
            raise ValueError("Character image must be provided and valid")

        # Analyze character image
        print(f"[PhotoAspectExtractor] Analyzing character image...")
        
        character_desc = self._analyze_single_image(
            ollama_url, ollama_model, character_b64,
            mode_config["analysis_prompt"]
        )
        
        if character_desc.startswith("[ERROR"):
            return (f"Failed to analyze image: {character_desc}",)
        
        print(f"[PhotoAspectExtractor] Analysis complete")
        print(f"  - Extracted {extraction_mode}: {character_desc[:80]}...")
        
        # Generate final description prompt
        synthesis_prompt = f"""Based on the following analysis, create a concise text-to-image prompt.

Analysis: {character_desc}

Requirements:
- Start with '{mode_config['synthesis_start']}'
- {mode_config['synthesis_focus']}
- Be specific and detailed
- Keep within 100 tokens
- No meta commentary, just the description"""

        system_prompt = mode_config["system_prompt"]
        
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

        print(f"[PhotoAspectExtractor] Calling Ollama vision model: {ollama_model}")
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
            print(f"[PhotoAspectExtractor] Added custom prompts at beginning")
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
            print(f"[PhotoAspectExtractor] Sending request to {ollama_url}")
            print(f"[PhotoAspectExtractor] Model: {payload.get('model')}")
            print(f"[PhotoAspectExtractor] Images count: {len(payload.get('images', []))}")
            
            response = requests.post(ollama_url, json=payload, timeout=120)
            
            print(f"[PhotoAspectExtractor] Response status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
            result = (data.get("response") or "").strip()
            print(f"[PhotoAspectExtractor] Received response ({len(result)} chars): {result[:100]}...")
            
            if not result:
                print("[PhotoAspectExtractor] WARNING: Empty response from Ollama")
                return "[Empty response from vision model]"
            
            return result
        except requests.exceptions.HTTPError as exc:
            error_msg = f"[PhotoAspectExtractor] HTTP error: {exc}"
            print(error_msg)
            if hasattr(exc.response, 'text'):
                print(f"[PhotoAspectExtractor] Response body: {exc.response.text[:500]}")
            return f"[ERROR: {exc}]"
        except requests.exceptions.ConnectionError as exc:
            error_msg = f"[PhotoAspectExtractor] Connection error: {exc}"
            print(error_msg)
            return f"[ERROR: Cannot connect to Ollama at {ollama_url}]"
        except requests.exceptions.Timeout as exc:
            error_msg = f"[PhotoAspectExtractor] Timeout error: {exc}"
            print(error_msg)
            return "[ERROR: Request timed out]"
        except Exception as exc:
            error_msg = f"[PhotoAspectExtractor] Error invoking Ollama: {exc}"
            print(error_msg)
            return f"[ERROR: {str(exc)}]"

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        if requests is None:
            return ["install_requests_library"]
        try:
            tags_url = f"{ollama_url}/api/tags"
            models = PhotoAspectExtractorNode._fetch_filter_models(tags_url)
            if models:
                return models

            if PhotoAspectExtractorNode._pull_florence_model():
                models = PhotoAspectExtractorNode._fetch_filter_models(tags_url)
                if models:
                    return models

            return ["no_vision_models"]
        except requests.exceptions.ConnectionError:
            return ["ollama_not_running"]
        except requests.exceptions.Timeout:
            return ["ollama_timeout"]
        except Exception as exc:
            print(f"[PhotoAspectExtractor] Error fetching Ollama models: {exc}")
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
            print("[PhotoAspectExtractor] Ollama executable not found; cannot pull florence model")
            return False

        if completed.returncode != 0:
            print(
                "[PhotoAspectExtractor] Failed to pull florence model: "
                f"{completed.stderr.strip() or 'unknown error'}"
            )
            return False

        print("[PhotoAspectExtractor] Successfully pulled florence model")
        return True


NODE_CLASS_MAPPINGS = {
    "WizdroidPhotoAspectExtractor": PhotoAspectExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidPhotoAspectExtractor": "Photo Aspect Extractor (Wizdroid)",
}
