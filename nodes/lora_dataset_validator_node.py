import json
from pathlib import Path
from typing import Tuple
from PIL import Image


class LoRADatasetValidatorNode:
    CATEGORY = "Wizdroid/train"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "report")
    FUNCTION = "validate_dataset"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_root": ("STRING", {"default": "training/kohya/datasets/char_demo"}),
                "min_short_side": ("INT", {"default": 768}),
                "min_images": ("INT", {"default": 20}),
            }
        }

    def validate_dataset(self, dataset_root: str, min_short_side: int, min_images: int) -> Tuple[str]:
        root = Path(dataset_root)
        captions = root / "captions.jsonl"
        images_dir = root / "images"

        report = []
        if not root.exists():
            return ("ERROR", "dataset root not found")

        if not captions.exists():
            report.append("captions.jsonl missing")

        entries = []
        try:
            with open(captions, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
        except Exception as e:
            report.append(f"captions parse error: {e}")

        if len(entries) < min_images:
            report.append(f"too few images: {len(entries)} < {min_images}")

        for e in entries:
            fname = e.get("file_name")
            if not fname:
                report.append(f"entry missing file_name: {e}")
                continue
            p = images_dir / fname
            if not p.exists():
                report.append(f"image missing: {fname}")
                continue
            try:
                im = Image.open(p)
                w, h = im.size
                s = min(w, h)
                if s < min_short_side:
                    report.append(f"image too small: {fname}, short side {s} < {min_short_side}")
            except Exception as ex:
                report.append(f"image open error: {fname} -> {ex}")

        if not report:
            return ("OK", "Dataset looks good")
        return ("WARN", "\n".join(report))


NODE_CLASS_MAPPINGS = {"WizdroidLoRADatasetValidator": LoRADatasetValidatorNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidLoRADatasetValidator": "LoRA Dataset Validator"}
