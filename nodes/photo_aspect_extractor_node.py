import base64
import io
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

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

from lib.content_safety import CONTENT_RATING_CHOICES, enforce_sfw
from lib.data_files import load_json
from lib.system_prompts import apply_content_policy, load_system_prompt_text

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


def _sanitize_clothes_prompt(text: str) -> str:
    """Best-effort cleanup for clothes-mode outputs.

    Some vision models (incl. llava variants) tend to add identity/background/event details even
    when instructed not to. This keeps the output focused on clothing/accessories.
    """

    raw = (text or "").strip()
    if not raw:
        return ""

    # Trim wrapping quotes some models add.
    if (raw.startswith("\"") and raw.endswith("\"")) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    # If the model included identity/context before the clothes clause, keep only from 'wearing' onward.
    lower_raw = raw.lower()
    if lower_raw.startswith("prompt:"):
        raw = raw[len("prompt:") :].strip()
        lower_raw = raw.lower()

    wearing_idx = lower_raw.find("wearing")
    if wearing_idx > 0:
        prefix = lower_raw[:wearing_idx]
        # Keep the prefix if it already contains clothing info (e.g., "in a black suit...").
        if not any(hint in prefix for hint in (
            "suit",
            "jacket",
            "coat",
            "shirt",
            "tie",
            "pants",
            "trousers",
            "jeans",
            "dress",
            "skirt",
            "shoes",
            "boots",
            "watch",
        )):
            raw = raw[wearing_idx:].strip()
            lower_raw = raw.lower()

    # If there's no 'wearing', anchor on the first clothing keyword to drop identity/context prefixes.
    if "wearing" not in lower_raw:
        earliest = None
        for hint in (
            "suit",
            "jacket",
            "coat",
            "shirt",
            "tie",
            "pants",
            "trousers",
            "jeans",
            "dress",
            "skirt",
            "shoes",
            "boots",
            "watch",
        ):
            idx = lower_raw.find(hint)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx
        if earliest is not None and earliest > 0:
            raw = raw[earliest:].strip()
            lower_raw = raw.lower()

    disallowed = {
        "prompt:",
        "background",
        "flags",
        "flag",
        "event",
        "ceremony",
        "president",
        "obama",
        "biden",
        "trump",
        "standing",
        "smiling",
        "smile",
        "camera",
        "gaze",
        "looking",
        "portrait",
        "patriotic",
        "in front of",
        "behind",
        "setting",
        "location",
        "scene",
        "curtain",
        "arms crossed",
        "formal setting",
    }

    clothing_hints = {
        "wearing",
        "suit",
        "jacket",
        "coat",
        "shirt",
        "t-shirt",
        "tee",
        "tie",
        "pants",
        "trousers",
        "jeans",
        "skirt",
        "dress",
        "shoes",
        "boots",
        "sneakers",
        "watch",
        "belt",
        "gloves",
        "hat",
        "cap",
        "scarf",
        "sunglasses",
        "fabric",
        "leather",
        "denim",
    }

    # Split into clauses and keep clauses that look clothing-related and not disallowed.
    # Prefer simple, deterministic parsing over another LLM pass.
    parts: List[str] = []
    for chunk in raw.replace(";", ".").split("."):
        clause = chunk.strip().strip(",")
        if not clause:
            continue
        lower = clause.lower()
        if any(bad in lower for bad in disallowed):
            continue
        if any(hint in lower for hint in clothing_hints):
            parts.append(clause)

    cleaned = ", ".join(parts).strip()
    if not cleaned:
        cleaned = raw

    # Normalize leading token to match the expected prompt style.
    lower_cleaned = cleaned.lower().lstrip()
    if lower_cleaned.startswith("wearing "):
        cleaned = "wearing " + cleaned[len("wearing ") :]
    elif lower_cleaned.startswith("wearing,"):
        cleaned = "wearing" + cleaned[len("wearing") :]
    elif cleaned.startswith("Wearing "):
        cleaned = "wearing " + cleaned[len("Wearing ") :]
    else:
        first = lower_cleaned.split(" ", 1)[0] if lower_cleaned else ""
        if first in {"suit", "jacket", "coat", "shirt", "dress", "skirt"}:
            cleaned = "wearing a " + cleaned.lstrip()
        elif first in {"tie", "pants", "trousers", "jeans", "shoes", "boots", "watch"}:
            cleaned = "wearing " + cleaned.lstrip()

    return cleaned.strip().strip(",")


def _load_json(name: str) -> Dict:
    return load_json(name)


def _get_extraction_modes() -> Dict[str, Dict[str, str]]:
    try:
        payload = load_json("photo_aspect_modes.json")
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return {}
    # Shallow validation
    out: Dict[str, Dict[str, str]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = {str(k): str(v) for k, v in value.items()}
    return out
def _with_random(options: List[str]) -> Tuple[str, ...]:
    return tuple([RANDOM_LABEL, NONE_LABEL] + options)


def _choose(value: Optional[str], options: List[str]) -> Optional[str]:
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt != NONE_LABEL]
        if not pool:
            pool = options[:]
        selection = random.choice(pool)
    else:
        selection = value

    if selection == NONE_LABEL or selection is None:
        return None
    return selection


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
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("extracted_prompt", "preview")
    FUNCTION = "extract_aspect"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = cls._collect_ollama_models()
        modes = _get_extraction_modes()
        mode_keys = sorted(modes.keys()) or [
            "clothes",
            "pose",
            "style",
            "background",
            "expression",
            "lighting",
            "hair",
            "makeup",
            "accessories",
            "camera",
            "composition",
            "color_palette",
            "full_description",
        ]

        return {
            "required": {
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW only"}),
                "extraction_mode": (mode_keys, {"default": mode_keys[0]}),
                "retain_face": ("BOOLEAN", {"default": False}),
                "retain_pose_and_camera": ("BOOLEAN", {"default": True}),
                "character_image": ("IMAGE",),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def extract_aspect(
        self,
        ollama_url: str,
        ollama_model: str,
        content_rating: str,
        extraction_mode: str,
        retain_face: bool,
        retain_pose_and_camera: bool,
        character_image,
        custom_prompt: str = "",
    ) -> Tuple[str]:
        modes = _get_extraction_modes()
        mode_config = modes.get(extraction_mode) or modes.get("clothes") or {}
        if not mode_config:
            return ("[ERROR: photo_aspect_modes.json missing or invalid]", "[ERROR: photo_aspect_modes.json missing or invalid]")

        # Modify prompts for face preservation if enabled
        analysis_prompt = mode_config.get("analysis_prompt", "Describe the image.")
        system_prompt = mode_config.get("system_prompt", "Output only the prompt.")

        # Keep extraction tightly scoped to the selected mode.
        # Many vision models will otherwise leak background/identity details into a focused aspect prompt.
        if extraction_mode != "full_description":
            if extraction_mode == "clothes":
                analysis_prompt = (
                    f"{analysis_prompt} IMPORTANT: Only describe clothing/outfit/accessories. "
                    "Do NOT mention background, setting, flags, location, or the person's identity."
                )
            elif extraction_mode != "background":
                analysis_prompt = (
                    f"{analysis_prompt} IMPORTANT: Only describe the requested aspect. "
                    "Do NOT mention unrelated details such as background/setting, identity, or other attributes."
                )

        if retain_face:
            # Always discourage facial-feature descriptions when face retention is requested.
            # Even in non-face modes, some vision models will mention face/eyes/gaze unless explicitly told not to.
            if extraction_mode == "full_description":
                analysis_prompt = "Provide a comprehensive description of everything visible in this image, including subjects, setting, lighting, colors, composition, mood, and any notable details. Be thorough and specific. IMPORTANT: Do not describe specific facial features (eyes, nose, mouth, face shape) as these should be preserved from the original."
                system_prompt = "You are a text-to-image prompt engineer specializing in comprehensive scene descriptions for face-preserving image editing. Create detailed, vivid prompts that capture the complete essence of an image including all visual elements, mood, and context, while explicitly avoiding description of facial features. Output only the prompt, no explanations."
            else:
                analysis_prompt = (
                    f"{analysis_prompt} IMPORTANT: Avoid describing specific facial features (eyes, nose, mouth, face shape) "
                    "as these should be preserved from the original."
                )
                system_prompt = (system_prompt or "").replace(
                    "Output only the prompt, no explanations.",
                    "IMPORTANT: Do not describe facial features (eyes, nose, mouth, face shape) as these should be preserved. Output only the prompt, no explanations.",
                )

        character_b64 = _image_to_base64(character_image)

        logging.getLogger(__name__).debug("[PhotoAspectExtractor] Image conversion results:")
        logging.getLogger(__name__).debug(f"  - Character: {len(character_b64) if character_b64 else 0} chars")
        logging.getLogger(__name__).debug(f"  - Extraction mode: {extraction_mode}")
        print(f"  - Retain face: {retain_face}")

        if not character_b64:
            raise ValueError("Character image must be provided and valid")

        # Analyze character image
        print(f"[PhotoAspectExtractor] Analyzing character image...")

        analysis_system = load_system_prompt_text("system_prompts/photo_aspect_analysis_system.txt", content_rating)
        
        character_desc = self._analyze_single_image(
            ollama_url,
            ollama_model,
            character_b64,
            analysis_prompt,
            analysis_system,
        )
        
        if character_desc.startswith("[ERROR"):
            error_msg = f"Failed to analyze image: {character_desc}"
            return (error_msg, error_msg)
        
        print(f"[PhotoAspectExtractor] Analysis complete")
        print(f"  - Extracted {extraction_mode}: {character_desc[:80]}...")
        
        # Generate final description prompt
        if extraction_mode == "full_description":
            if retain_face:
                synthesis_prompt = f"""Based on the following comprehensive analysis, create a detailed text-to-image prompt that captures the entire scene for face-preserving image editing.

Analysis: {character_desc}

Requirements:
- Start with: 'Retain the facial features from the original image.'
- Then create a complete, detailed description of the entire image
- Include all subjects, setting, lighting, colors, mood, and atmosphere
- Be vivid and specific with rich descriptive language
- Do NOT describe facial features (eyes, nose, mouth, face shape) - these are preserved
- Keep within 150 tokens for comprehensive coverage
- No meta commentary, just the descriptive prompt"""
            else:
                synthesis_prompt = f"""Based on the following comprehensive analysis, create a detailed text-to-image prompt that captures the entire scene.

Analysis: {character_desc}

Requirements:
- Create a complete, detailed description of the entire image
- Include all subjects, setting, lighting, colors, mood, and atmosphere
- Be vivid and specific with rich descriptive language
- Keep within 150 tokens for comprehensive coverage
- No meta commentary, just the descriptive prompt"""
        else:
            if retain_face and extraction_mode in ["expression", "hair", "makeup"]:
                synthesis_prompt = f"""Based on the following analysis, create a concise text-to-image prompt for face-preserving image editing.

Analysis: {character_desc}

Requirements:
- Start with: 'Retain the facial features from the original image.'
- Then describe: '{mode_config['synthesis_start']}'
- {mode_config['synthesis_focus']}
- Be specific and detailed
- Do NOT describe facial features (eyes, nose, mouth, face shape) - these are preserved
- Keep within 100 tokens
- No meta commentary, just the description"""
            else:
                pose_camera_line = (
                    "- Preserve the original pose and camera angle; do not introduce new pose/camera descriptions unless the mode is pose or camera\n"
                    if retain_pose_and_camera
                    else ""
                )
                synthesis_prompt = f"""Based on the following analysis, create a concise text-to-image prompt.

Analysis: {character_desc}

Requirements:
- Start with the exact word '{mode_config['synthesis_start']}'
- {mode_config['synthesis_focus']}
- Only describe the requested aspect; do not add background/identity details unless the mode is background
{pose_camera_line}- Be specific and detailed
- Keep within 100 tokens
- No meta commentary, just the description"""

        system_prompt = apply_content_policy(system_prompt, content_rating)
        
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
                "num_predict": 250 if extraction_mode == "full_description" else 200,
                "temperature": 0.7,
            }
        }

        print(f"[PhotoAspectExtractor] Calling Ollama vision model: {ollama_model}")
        response = self._invoke_ollama(synthesis_url, payload)

        if extraction_mode == "clothes" and response and not response.startswith("[ERROR"):
            response = _sanitize_clothes_prompt(response)
        
        # Build final prompt prefix in a predictable order:
        # 1) face-retention directive (if requested)
        # 2) pose/camera retention directive (if requested)
        # 2) any user-provided custom prompt
        # 3) model-generated aspect prompt
        prefix_parts: List[str] = []
        if retain_face:
            prefix_parts.append("Retain the facial features from the original image.")
        if retain_pose_and_camera:
            prefix_parts.append("Retain the original pose and camera angle from the original image.")
        if custom_prompt.strip():
            prefix_parts.append(custom_prompt.strip())

        out_parts: List[str] = []
        out_parts.extend(prefix_parts)
        if response:
            out_parts.append(response)

        out = ", ".join([p for p in out_parts if p])
        if prefix_parts:
            print("[PhotoAspectExtractor] Added prompt prefixes")

        if content_rating != "NSFW allowed":
            err = enforce_sfw(out)
            if err:
                blocked = "[Blocked: potential NSFW content detected. Switch content_rating to 'NSFW allowed'.]"
                return (blocked, blocked)
        return (out, out)
    
    def _analyze_single_image(
        self,
        ollama_url: str,
        ollama_model: str,
        image_b64: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """Analyze a single image with vision model."""
        if system is None:
            # Backward-compatible default for internal callers that predate the
            # explicit 'system' parameter (e.g. LoRA dataset export tooling).
            # Keep it safely scoped to SFW unless the caller explicitly supplies a policy.
            system = load_system_prompt_text("system_prompts/photo_aspect_analysis_system.txt", "SFW only")

        # Ensure URL has the /api/generate endpoint
        if not ollama_url.endswith("/api/generate"):
            ollama_url = ollama_url.rstrip("/") + "/api/generate"
        
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "system": system,
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

        last_data: Optional[Dict] = None
        for attempt in range(2):
            try:
                print(f"[PhotoAspectExtractor] Sending request to {ollama_url}")
                print(f"[PhotoAspectExtractor] Model: {payload.get('model')}")
                print(f"[PhotoAspectExtractor] Images count: {len(payload.get('images', []))}")

                response = requests.post(ollama_url, json=payload, timeout=120)

                print(f"[PhotoAspectExtractor] Response status: {response.status_code}")

                response.raise_for_status()
                data = response.json()
                last_data = data if isinstance(data, dict) else None

                # Ollama may return output as either:
                # - /api/generate: { response: "...", thinking: "..." }
                # - /api/chat: { message: { content: "..." } }
                msg = data.get("message") if isinstance(data, dict) else None
                if isinstance(msg, dict):
                    result = (msg.get("content") or "").strip()
                else:
                    result = ""

                if not result and isinstance(data, dict):
                    result = (data.get("response") or "").strip()

                print(f"[PhotoAspectExtractor] Received response ({len(result)} chars): {result[:100]}...")

                if result:
                    return result

                # Some vision models (notably qwen3-vl variants) can spend tokens in internal
                # 'thinking' and end up with an empty 'response' when num_predict is too low.
                # If Ollama indicates truncation, retry once with a higher token budget.
                done_reason = (data.get("done_reason") or "") if isinstance(data, dict) else ""
                if attempt == 0 and str(done_reason).lower() == "length":
                    opts = payload.get("options") if isinstance(payload, dict) else None
                    if not isinstance(opts, dict):
                        opts = {}

                    num_predict = opts.get("num_predict")
                    if isinstance(num_predict, int) and num_predict > 0:
                        bumped = min(max(num_predict * 2, num_predict + 200), 1024)
                    else:
                        bumped = 400

                    payload = json.loads(json.dumps(payload))
                    payload.setdefault("options", {})
                    payload["options"]["num_predict"] = bumped
                    print(
                        "[PhotoAspectExtractor] Empty response with done_reason=length; "
                        f"retrying once with num_predict={bumped}"
                    )
                    continue

                print("[PhotoAspectExtractor] WARNING: Empty response from Ollama")
                return "[Empty response from vision model]"
            except requests.exceptions.HTTPError as exc:
                error_msg = f"[PhotoAspectExtractor] HTTP error: {exc}"
                print(error_msg)
                if hasattr(exc.response, "text"):
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

        # Should be unreachable due to returns, but keep a safe fallback.
        if last_data and isinstance(last_data, dict) and (last_data.get("thinking") or ""):
            print("[PhotoAspectExtractor] WARNING: Model returned only 'thinking' and no response")
        return "[Empty response from vision model]"

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
            logging.getLogger(__name__).warning("[PhotoAspectExtractor] Ollama executable not found; cannot pull florence model")
            return False

        if completed.returncode != 0:
            logging.getLogger(__name__).warning(
                "[PhotoAspectExtractor] Failed to pull florence model: %s",
                completed.stderr.strip() or "unknown error",
            )
            return False

        logging.getLogger(__name__).info("[PhotoAspectExtractor] Successfully pulled florence model")
        return True


NODE_CLASS_MAPPINGS = {
    "WizdroidPhotoAspectExtractor": PhotoAspectExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidPhotoAspectExtractor": "Photo Aspect Extractor",
}
