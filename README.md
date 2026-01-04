# Wizdroid Character Nodes for ComfyUI

Custom nodes for ComfyUI that generate and edit character-focused prompts using Ollama.

## What you get

- Character prompt builder with structured options (gender, age, body, hair, makeup, fashion, poses, backgrounds, etc.).
- Character edit node to tweak existing prompts.
- Prompt combiner for merging prompt fragments.
- Photo aspect extractor using Ollama vision models.
- LoRA dataset export helpers for Kohya-style training.
- Data-driven contest prompt generator (loads contest rules from JSON in `data/`).

## Fashion Styles Gallery

Browse our [Fashion Styles AI Gallery](https://wizdroid.github.io/wizdroid-character/docs/www/gallery.html) to explore all 57 fashion aesthetics across different AI image generation models (Flux.1 Dev, SDXL 1.0, Z-Image-Turbo, Qwen Image 202512).

## Install

1. Clone into your ComfyUI custom nodes folder:

   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/wizdroid-character.git
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running with at least one text model (and optional vision models if you use the analysis nodes).

## Configure

Key JSON files in `data/`:

- `character_options.json` – character attributes, poses, fashion, backgrounds, etc.
- `countries.json` / `regions.json` – geography for style prompts.
- `prompt_styles.json` – prompt templates per model family.

Contest config (optional):

- `contest.json` – single replaceable contest definition for the `Contest Prompt Generator (Ollama)` node.
   - Replace this file per contest theme (then restart ComfyUI).

System prompts & content policy:

- `system_prompts/*.txt` – system prompts used by the Ollama-backed nodes.
- `content_policies.json` – text blocks appended to system prompts based on `content_rating`.

Restart ComfyUI after editing these files.

## Use

Once installed, look for nodes under the `Wizdroid/character` category in ComfyUI and wire them into your normal image-generation workflows.

## License

Licensed under the Apache License, Version 2.0.