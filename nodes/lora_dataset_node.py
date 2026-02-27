"""🧙 Wizdroid LoRA Dataset Export Node - Export and validate image datasets for LoRA training."""

import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "training/kohya/datasets"

# Validation thresholds
MIN_IMAGE_SIZE = 256
MAX_IMAGE_SIZE = 4096
MIN_IMAGES = 3
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _save_image_with_resize(source: Path, target: Path, resize_mode: str) -> Tuple[int, int]:
    """Copy or resize an image and return final dimensions."""
    if resize_mode == "none":
        target.write_bytes(source.read_bytes())
        with Image.open(source) as img:
            return img.size

    max_side = 1024 if resize_mode == "longest_1024" else 512
    with Image.open(source) as img:
        resample_attr = getattr(Image, "Resampling", None)
        resample_filter = resample_attr.LANCZOS if resample_attr else Image.LANCZOS
        img.thumbnail((max_side, max_side), resample=resample_filter)

        save_kwargs = {}
        format_hint = None
        suffix = target.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            format_hint = "JPEG"
            save_kwargs["quality"] = 95
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
        elif suffix == ".png":
            format_hint = "PNG"

        img.save(target, format=format_hint, **save_kwargs)
        return img.size


def _validate_dataset(images_dir: Path, captions: list) -> List[str]:
    """Validate the exported dataset for quality and completeness.
    
    Returns a list of warning/error messages. Empty list = all good.
    """
    issues = []
    
    # Check minimum image count
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if len(image_files) < MIN_IMAGES:
        issues.append(f"⚠️ Only {len(image_files)} images found. Recommend at least {MIN_IMAGES} for LoRA training.")
    
    # Check each image
    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                w, h = img.size
                if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                    issues.append(f"⚠️ {img_file.name}: too small ({w}x{h}), min {MIN_IMAGE_SIZE}px recommended.")
                if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
                    issues.append(f"⚠️ {img_file.name}: very large ({w}x{h}), may slow training.")
        except Exception as e:
            issues.append(f"❌ {img_file.name}: cannot open image - {e}")
    
    # Check captions
    empty_captions = [c for c in captions if not c.get("prompt", "").strip()]
    if empty_captions:
        issues.append(f"⚠️ {len(empty_captions)} images have empty captions.")
    
    # Check for caption file consistency
    for caption in captions:
        caption_file = images_dir / f"{Path(caption['file_name']).stem}.txt"
        image_file = images_dir / caption["file_name"]
        if not image_file.exists():
            issues.append(f"❌ {caption['file_name']}: referenced in captions but file missing.")
    
    return issues


class WizdroidLoRADatasetNode:
    """🧙 Export image datasets for LoRA training with auto-captioning and built-in validation."""
    
    CATEGORY = "🧙 Wizdroid/Training"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("dataset_path", "captions_jsonl", "validation_report")
    FUNCTION = "export_dataset"

    @classmethod
    def INPUT_TYPES(cls):
        from wizdroid_lib.constants import DEFAULT_OLLAMA_URL
        from wizdroid_lib.ollama_client import collect_models
        
        ollama_models = collect_models(DEFAULT_OLLAMA_URL)
        
        return {
            "required": {
                "dataset_slug": ("STRING", {"default": "char_demo"}),
                "character_tag": ("STRING", {"default": "<char_demo>"}),
                "images": ("STRING", {"default": ""}),
                "prompt_source": (["manual", "ollama_vision"], {"default": "ollama_vision"}),
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "resize_mode": (["none", "longest_1024", "longest_512"], {"default": "none"}),
                "validate_dataset": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "manual_prompt": ("STRING", {"default": ""}),
                "write_caption_files": ("BOOLEAN", {"default": True}),
            },
        }

    def export_dataset(
        self,
        dataset_slug: str,
        character_tag: str,
        images: str,
        prompt_source: str,
        ollama_url: str,
        ollama_model: str,
        resize_mode: str,
        validate_dataset: bool = True,
        manual_prompt: str = "",
        write_caption_files: bool = True,
    ) -> Tuple[str, str, str]:
        """
        Create dataset directory with `images/`, `captions.jsonl`, and `metadata.yaml`.
        Optionally validates the dataset after export.

        `images` is the path to a directory containing files, or a comma-separated list of image paths.
        """

        root = Path(DEFAULT_DATA_ROOT) / dataset_slug
        images_dir = root / "images"
        root.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Normalize images input
        if "," in images:
            provided = [p.strip() for p in images.split(",") if p.strip()]
        else:
            provided = [images] if images else []
            if images and not any(images.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                # assume single folder
                provided = [str(p) for p in Path(images).glob("**/*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

        captions = []

        for idx, p in enumerate(provided):
            p_path = Path(p)
            if not p_path.is_file():
                continue

            target_name = f"{idx:04d}{p_path.suffix}"
            target_path = images_dir / target_name
            should_overwrite = resize_mode != "none"
            if not target_path.exists() or should_overwrite:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                _save_image_with_resize(p_path, target_path, resize_mode)

            prompt = None
            if prompt_source == "manual":
                prompt = manual_prompt.strip()
            elif prompt_source == "ollama_vision":
                # Use Ollama vision model to describe the image
                try:
                    import base64
                    from wizdroid_lib.ollama_client import generate_text

                    with open(p_path, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode("utf-8")

                    ok, desc = generate_text(
                        ollama_url=ollama_url,
                        model=ollama_model,
                        prompt=(
                            "Describe this image in detail for use as an AI image generation caption. "
                            "Include: subject appearance, clothing, pose, expression, background, lighting, "
                            "and overall style. Be specific and concise. Output only the description."
                        ),
                        system="You are a precise image captioning assistant. Output only descriptive text, no explanations.",
                        images=[b64],
                        timeout=60,
                    )
                    prompt = desc if ok else ""
                except Exception as e:
                    logger.warning(f"Vision captioning failed for {p_path.name}: {e}")
                    prompt = ""

            if prompt is None:
                prompt = ""

            # attach character tag at front if not already present
            if character_tag and character_tag not in prompt:
                prompt = f"{character_tag} {prompt}".strip()

            captions.append({"file_name": target_name, "prompt": prompt})

            if write_caption_files:
                caption_path = images_dir / f"{Path(target_name).stem}.txt"
                caption_path.write_text(prompt, encoding="utf-8")

        # Write captions.jsonl
        captions_jsonl = root / "captions.jsonl"
        with captions_jsonl.open("w", encoding="utf-8") as fh:
            for e in captions:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")

        # Write metadata
        meta = {
            "lora_tag": character_tag,
            "num_images": len(captions),
            "prompt_source": prompt_source,
            "resize_mode": resize_mode,
        }
        try:
            import yaml

            with (root / "metadata.yaml").open("w", encoding="utf-8") as fh:
                yaml.safe_dump(meta, fh)
        except Exception:
            with (root / "metadata.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)

        # Built-in validation
        validation_report = "✅ Dataset exported successfully."
        if validate_dataset:
            issues = _validate_dataset(images_dir, captions)
            if issues:
                validation_report = f"Dataset exported with {len(issues)} issue(s):\n" + "\n".join(issues)
            else:
                validation_report = f"✅ Dataset exported and validated: {len(captions)} images, all checks passed."

        return str(root), str(captions_jsonl), validation_report


NODE_CLASS_MAPPINGS = {"WizdroidLoRADataset": WizdroidLoRADatasetNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidLoRADataset": "🧙 Wizdroid: LoRA Dataset Export"}
