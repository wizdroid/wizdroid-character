# Wizdroid Character Nodes for ComfyUI

_Version: 2025.11.01_

A streamlined collection of custom nodes for ComfyUI that provide AI-powered prompt generation tools using Ollama. These nodes focus on detailed character creation, cultural storytelling, image analysis, and remix workflows.

## Features

### Character Prompt Builder
- **Purpose**: Generates detailed character prompts for image generation with extensive customization options
- **Attributes**: Character name, gender, age group, body type, hair color/style, eye color, facial expressions, poses, makeup, fashion styles, upcycled fashion materials, and backgrounds
- **Integration**: Ollama LLM integration for intelligent prompt crafting
- **Controls**: LLM token-limit dropdown (128–4096 tokens in 128-token steps; applies only to the Ollama model) plus workflow seed control
- **Camera lens**: `camera_lens` uses concise focal-length or lens-type names (24mm, 50mm, 85mm, 70-200mm, fisheye, anamorphic, medium format, etc.), without attached descriptions.
- **Color palettes**: `color_palette` contains practical photoshoot palettes (black and white, warm neutrals, muted earth tones, teal & orange, high-contrast, studio white) rather than creative/abstract names.
- **Pose controls**: `pose_content_rating` toggle (SFW / NSFW / Mixed) paired with SFW and NSFW pose catalogs for safer randomization
- **Output**: Positive prompt, negative prompt, and a preview string (matches the positive prompt)

### Character Edit Node
- **Purpose**: Takes an existing prompt and nudges it toward refined directions (fashion, lighting, mood, etc.)
- **Tools**: Section-by-section emphasis controls with Ollama-powered rewrites
- **Controls**: Shares the same LLM token dropdown (128–4096 tokens) to keep Ollama generations on budget
- **Pose controls**: Select whether edits should draw from the SFW or NSFW pose pool before randomization
- **Output**: Revamped prompt plus diff-style summary for reference

### Prompt Combiner Node
- **Purpose**: Merge multiple prompt fragments into a cohesive description
- **Features**: Adjustable weights, connective phrasing, and automated deduplication
- **Controls**: Uses the same Ollama token dropdown to enforce consistent LLM verbosity across merged prompts
- **Output**: Blended prompt ready for downstream nodes

### Photo Aspect Extractor
- **Purpose**: Analyzes images using vision models to extract specific aspects
- **Modes**: Clothes, pose, style, background, expression, lighting, hair, makeup, accessories, camera settings, composition, color palette
- **Integration**: Ollama vision models (LLaVA, Florence, etc.) for image analysis
- **Output**: Extracted aspect descriptions for prompt engineering

> Tip: Every LLM-driving node now exposes a `token_limit_override` dropdown (128–4096 tokens, default 128). It only governs Ollama’s response length, not your downstream image generator, so you can standardize prompt verbosity without touching `prompt_styles.json`.

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/wizdroid-character.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is running with appropriate models:
   - For text generation: Install models like `llama2`, `mistral`, or `codellama`
   - For vision analysis: Install vision models like `llava`, `bakllava`, or `florence`

## Configuration

The nodes use JSON configuration files in the `data/` directory:
- `character_options.json`: Character attributes, upcycled materials, and styling options
- `countries.json`: Specific countries for fashion inspiration
- `regions.json`: Broader regional areas for fashion inspiration
- `prompt_styles.json`: Output format styles for different AI models
   - Includes presets for Flux, SD/SDXL, SDXL-Turbo, Juggernaut, RealVis, HiDream, and Qwen editing formats
Note: `fantasy_options.json` and the Fantasy Scene node have been removed from this package. If you previously relied on fantasy scene generation, please migrate any workflows to the Character Prompt Builder or add a custom node. Former `glamour_options` live inside the `pose_style.sfw` / `pose_style.nsfw` blocks in `character_options.json`, and the unused `followup_questions.json` catalog has been retired.

## Usage

After installation, the nodes will appear in ComfyUI under the "Wizdroid/character" category.

### Common Workflow
1. Use **Character Prompt Builder** to generate richly detailed prompts
2. Use **Photo Aspect Extractor** (or your preferred external pose tool) to analyze reference images
3. Combine extracted elements with generated prompts for refined results via **Prompt Combiner**
4. Iterate with **Character Edit Node** to explore alternate directions

## Requirements

- ComfyUI
- Ollama server running locally
- Python packages: requests, torch, PIL (Pillow)
- For vision features: Ollama vision models

## Contributing

Feel free to contribute by:
- Adding more cultural outfits and countries
- Expanding attribute options in JSON files
- Improving prompt generation logic
- Adding new node types
- Enhancing vision model integrations

## License

[Add your license here]