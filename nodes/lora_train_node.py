import json
import os
import subprocess
from pathlib import Path
from typing import Tuple


class SDXLLoRATrainerNode:
    CATEGORY = "Wizdroid/train"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "trained_lora")
    FUNCTION = "train_lora"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_root": ("STRING", {"default": "training/kohya/datasets/char_demo"}),
                "python_exec": ("STRING", {"default": "/usr/bin/python3"}),
                "pretrained_model_name_or_path": ("STRING", {"default": "/path/to/sdxl"}),
                "max_train_steps": ("INT", {"default": 3000}),
                "network_dim": ("INT", {"default": 64}),
                "network_alpha": ("INT", {"default": 64}),
                "output_dir": ("STRING", {"default": "training/kohya/results/char_demo"}),
            },
            "optional": {
                "use_safetensors": ("BOOLEAN", {"default": True}),
            },
        }

    def train_lora(
        self,
        dataset_root: str,
        python_exec: str,
        pretrained_model_name_or_path: str,
        max_train_steps: int,
        network_dim: int,
        network_alpha: int,
        output_dir: str,
        use_safetensors: bool = True,
    ) -> Tuple[str]:
        data = Path(dataset_root)
        captions = data / "captions.jsonl"
        images = data / "images"
        if not data.exists() or not captions.exists():
            return ("ERROR: Dataset or captions missing", "")

        cmd = [
            python_exec,
            "thirdparty/sd-scripts/sdxl_train.py",
            "--pretrained_model_name_or_path",
            pretrained_model_name_or_path,
            "--train_data_dir",
            str(images),
            "--in_json",
            str(captions),
            "--sdxl",
            "--network_module",
            "networks.lora",
            "--network_dim",
            str(network_dim),
            "--network_alpha",
            str(network_alpha),
            "--max_train_steps",
            str(max_train_steps),
            "--output_dir",
            str(output_dir),
        ]

        if use_safetensors:
            cmd.extend(["--use_safetensors", "--save_model_as", "safetensors"])

        # Use a small log file in output to see final weights
        out_log = Path(output_dir) / "train.log"
        out_log.parent.mkdir(parents=True, exist_ok=True)

        try:
            with out_log.open("w", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
                proc.wait()

            if proc.returncode != 0:
                return (f"ERROR: Trainer exited with code {proc.returncode}", str(out_log))

            # Find safetensors in output
            candidates = list(Path(output_dir).glob("**/*.safetensors"))
            if not candidates:
                return ("OK", str(out_log))
            # return last modified
            best = max(candidates, key=lambda p: p.stat().st_mtime)
            return ("OK", str(best))
        except FileNotFoundError as e:
            return (f"ERROR: {e}", "")


NODE_CLASS_MAPPINGS = {"WizdroidSDXLLoRATrainer": SDXLLoRATrainerNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidSDXLLoRATrainer": "SDXL LoRA Trainer"}
