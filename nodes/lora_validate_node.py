import subprocess
from pathlib import Path
from typing import Tuple


class WizdroidLoRAValidateNode:
    """🧙 Validate LoRA training by generating preview images."""
    
    CATEGORY = "🧙 Wizdroid/Training"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "out_dir")
    FUNCTION = "validate_lora"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "<char_demo> portrait, cinematic lighting"}),
                "python_exec": ("STRING", {"default": "/usr/bin/python3"}),
                "pretrained_model_ckpt": ("STRING", {"default": "/path/to/sdxl_ckpt.safetensors"}),
                "out_dir": ("STRING", {"default": "training/kohya/validate"}),
            }
        }

    def validate_lora(self, lora_path: str, prompt: str, python_exec: str, pretrained_model_ckpt: str, out_dir: str) -> Tuple[str]:
        if not Path(lora_path).exists():
            return ("ERROR: Missing LoRA weights", "")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_exec,
            "thirdparty/sd-scripts/sdxl_gen_img.py",
            "--ckpt",
            pretrained_model_ckpt,
            "--outdir",
            str(out),
            "--prompt",
            prompt,
            "--network_module",
            "networks.lora",
            "--network_weights",
            lora_path,
            "--n_iter",
            "1",
            "--scale",
            "8.0",
            "--steps",
            "28",
            "--W",
            "1024",
            "--H",
            "1024",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return (f"ERROR: validate failed: {proc.stderr[:200]}", "")

        return ("OK", str(out))


NODE_CLASS_MAPPINGS = {"WizdroidLoRAValidate": WizdroidLoRAValidateNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WizdroidLoRAValidate": "🧙 Wizdroid: LoRA Validate"}
