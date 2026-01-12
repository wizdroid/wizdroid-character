# 🧙 Wizdroid Character Nodes for ComfyUI

Custom nodes for ComfyUI that generate and edit character-focused prompts using Ollama LLM.

## ✨ Features

### Prompt Generation (🧙 Wizdroid/Prompts)
- **🧙 Wizdroid: Character Prompt** - Comprehensive character prompt builder with structured options (gender, age, body, hair, makeup, fashion, poses, backgrounds, etc.)
- **🧙 Wizdroid: Scene Generator** - Generate vivid scene prompts for any imaginable scenario
- **🧙 Wizdroid: Background** - Create surreal background prompts without human figures
- **🧙 Wizdroid: Meta Prompt** - Expand loose keywords into detailed image prompts
- **🧙 Wizdroid: Prompt Combiner** - Merge multiple prompts into one coherent description
- **🧙 Wizdroid: Image Edit** - Generate multi-image editing instructions (face swap, style transfer)
- **🧙 Wizdroid: Multi-Angle** - Camera position prompts for Qwen multi-angle LoRA
- **🧙 Wizdroid: Contest Prompt** - Data-driven contest prompt generator

### Analysis (🧙 Wizdroid/Analysis)
- **🧙 Wizdroid: Photo Aspect Extractor** - Extract clothes, pose, style from images using vision models

### Training (🧙 Wizdroid/Training)
- **🧙 Wizdroid: LoRA Dataset Export** - Export image datasets for LoRA training
- **🧙 Wizdroid: LoRA Trainer** - Train SDXL LoRA models using Kohya sd-scripts
- **🧙 Wizdroid: LoRA Validate** - Generate preview images to validate trained LoRAs
- **🧙 Wizdroid: LoRA Dataset Validator** - Check dataset quality and completeness

## 🎨 Fashion Styles Gallery

Browse our [Fashion Styles AI Gallery](https://wizdroid.github.io/wizdroid-character/gallery.html) to explore all 57 fashion aesthetics across different AI image generation models (Flux.1 Dev, SDXL 1.0, Z-Image-Turbo, Qwen Image 202512).

## 📦 Install

1. Clone into your ComfyUI custom nodes folder:

   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/wizdroid/wizdroid-character.git
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running with at least one text model (and optional vision models for analysis nodes).

## ⚙️ Configure

Key JSON files in `data/`:

- `character_options.json` – character attributes, poses, fashion, backgrounds, etc.
- `countries.json` / `regions.json` – geography for style prompts.
- `prompt_styles.json` – prompt templates per model family (SDXL, Flux, SD3, etc.).

Content policy:
- `content_policies.json` – content rating policies (SFW, NSFW, Mixed).

System prompts:
- `system_prompts/*.txt` – structured system prompts for each node type.

Contest config (optional):
- `contest.json` – replaceable contest definition for the Contest Prompt node.

Restart ComfyUI after editing these files.

## 🚀 Use

Once installed, look for nodes under these categories in ComfyUI:
- `🧙 Wizdroid/Prompts` - All prompt generation nodes
- `🧙 Wizdroid/Analysis` - Image analysis nodes
- `🧙 Wizdroid/Training` - LoRA training nodes

Wire them into your normal image-generation workflows.

## 📄 License

Licensed under the Apache License, Version 2.0.
