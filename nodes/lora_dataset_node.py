import json
import os
from pathlib import Path
from typing import Tuple

from PIL import Image

from .character_prompt_node import WizdroidCharacterPromptNode
from .photo_aspect_extractor_node import WizdroidPhotoAspectNode


DEFAULT_DATA_ROOT = "training/kohya/datasets"


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


class WizdroidLoRADatasetNode:
    """🧙 Export image datasets for LoRA training with auto-captioning."""
    
    CATEGORY = "🧙 Wizdroid/Training"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_path", "captions_jsonl")
    FUNCTION = "export_dataset"

    @classmethod
    def INPUT_TYPES(cls):
        ollama_models = WizdroidPhotoAspectNode._collect_ollama_models()
        return {
            "required": {
                "dataset_slug": ("STRING", {"default": "char_demo"}),
                "character_tag": ("STRING", {"default": "<char_demo>"}),
                "images": ("STRING", {"default": ""}),
                "prompt_source": (["manual", "character_builder", "ollama_vision"], {"default": "character_builder"}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434"}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "resize_mode": (["none", "longest_1024", "longest_512"], {"default": "none"}),
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
        manual_prompt: str = "",
        write_caption_files: bool = True,
    ) -> Tuple[str]:
        """
        Create dataset directory with `images/`, `captions.jsonl`, and `metadata.yaml`.

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
            if images and not any(p.endswith(tuple([".png", ".jpg", ".jpeg"])) for p in provided):
                # assume single folder
                provided = [p for p in Path(images).glob("**/*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]

        captions = []

        # Helper for Ollama vision model
        extractor = WizdroidPhotoAspectNode()
        builder = WizdroidCharacterPromptNode()

        for idx, p in enumerate(provided):
            p_path = Path(p)
            if p_path.is_file():
                target_name = f"{idx:04d}{p_path.suffix}"
                target_path = images_dir / target_name
                should_overwrite = resize_mode != "none"
                if target_path.exists() and not should_overwrite:
                    pass
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_image_with_resize(p_path, target_path, resize_mode)
            else:
                continue

            prompt = None
            if prompt_source == "manual":
                prompt = manual_prompt.strip()
            elif prompt_source == "character_builder":
                # attempt to generate using minimal defaults
                prompt, _, _ = builder.build_prompt(
                    character_name="",
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    prompt_style="SDXL",
                    retain_face=False,
                    image_category="none",
                    image_content_rating="SFW only",
                    gender="none",
                    race="none",
                    age_group="none",
                    body_type="none",
                    hair_color="none",
                    hair_style="none",
                    eye_color="none",
                    facial_expression="none",
                    face_angle="none",
                    camera_angle="none",
                    pose_content_rating="SFW only",
                    pose_style="none",
                    makeup_style="none",
                    fashion_outfit="none",
                    fashion_style="none",
                    footwear_style="none",
                    background_stage_style="none",
                    background_location_style="none",
                    background_imaginative_style="none",
                    lighting_style="none",
                    camera_lens="none",
                    color_palette="none",
                    upcycled_fashion="none",
                    region="none",
                    country="none",
                    culture="none",
                    custom_text_llm="",
                    custom_text_append="",
                )
            elif prompt_source == "ollama_vision":
                # Use PhotoAspectExtractor to describe clothing and style
                import base64

                with open(p_path, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("utf-8")

                # call the internal analyzer with the base64-encoded image to get a description
                desc = extractor._analyze_single_image(
                    ollama_url,
                    ollama_model,
                    b64,
                    "Describe the clothing, outfit, garments, colors, fabrics, accessories, and styling worn by the person in this image. Be specific and detailed.",
                )
                prompt = desc

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

        return str(root), str(captions_jsonl)


NODE_CLASS_MAPPINGS = {"WizdroidLoRADataset": WizdroidLoRADatasetNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidLoRADataset": "🧙 Wizdroid: LoRA Dataset Export"}
