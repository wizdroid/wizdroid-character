# 🧙 Wizdroid Character Nodes for ComfyUI

Custom nodes for ComfyUI that generate and edit character-focused prompts using Ollama LLM.
Optimized for **consistent character generation across genders** with gender-aware filtering.

## ✨ Features

### Prompt Generation (🧙 Wizdroid/Prompts)
- **🧙 Wizdroid: Character Prompt** - Comprehensive character prompt builder with gender-aware options (body, hair, makeup, fashion, poses, backgrounds, etc.)
- **🧙 Wizdroid: Character Edit** - Multi-reference image editing (face, clothing, pose, background, style transfers) with gender-appropriate beauty defaults
- **🧙 Wizdroid: Scene Generator** - Generate vivid scene prompts for any imaginable scenario
- **🧙 Wizdroid: Background** - Create surreal background prompts without human figures
- **🧙 Wizdroid: Meta Prompt** - Expand loose keywords into detailed image prompts
- **🧙 Wizdroid: Prompt Combiner** - Merge multiple prompts into one coherent description
- **🧙 Wizdroid: Image Edit** - Generate multi-image editing instructions (face swap, style transfer)

### Analysis (🧙 Wizdroid/Analysis)
- **🧙 Wizdroid: Photo Aspect Extractor** - Extract clothes, pose, style from images using vision models

### Training (🧙 Wizdroid/Training)
- **🧙 Wizdroid: LoRA Dataset Export** - Export image datasets for LoRA training with built-in validation

## 🔄 v2025.12.01 — Architecture Refactor

### What Changed
- **Removed** Contest Prompt, LoRA Trainer, LoRA Validate, LoRA Dataset Validator nodes (14 → 9 nodes)
- **Merged** Character Edit + Multi-Angle into a single unified Character Edit node
- **Consolidated** data into `data/shared/` — single source of truth for body types, emotions, hair, eyes, makeup, poses, fashion, backgrounds, camera/lighting
- **Added gender-aware filtering** — body types, poses, makeup, fashion outfits and styles are now tagged with gender and filtered based on selected gender
- **Added male-specific options** — 6 male body types, 12 male hairstyles, 15 male poses, 4 male makeup styles, 17 male-specific fashion items
- **Data-driven architecture** — Background node moved from hardcoded tuples to JSON data files
- **Built-in dataset validation** — LoRA Dataset Export now validates automatically (no separate node needed)

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

### Data Architecture

```
data/
├── shared/                      # Single source of truth
│   ├── body_types.json          # Gender-tagged body types
│   ├── emotions.json            # 100 comprehensive emotions
│   ├── eye_colors.json          # 40 eye colors
│   ├── hair.json                # Colors + gender-specific styles
│   ├── skin_tones.json          # 40 skin tones
│   ├── makeup.json              # Gender-tagged makeup styles
│   ├── poses.json               # SFW/NSFW + gender-specific poses
│   ├── backgrounds.json         # Studio/real/imaginative
│   ├── camera_lighting.json     # Lighting, camera, color palettes
│   ├── fashion.json             # Outfits, styles, footwear (gender-tagged)
│   └── background_edit.json     # Background node themes/moods
├── character_options.json       # Character identity (gender, race, age, image category)
├── scene_data.json              # Scene categories and moods
├── meta_prompt_options.json     # Meta prompt worlds and visual styles
├── photo_aspect_modes.json      # Photo aspect extraction modes
├── prompt_styles.json           # Prompt templates per model family
├── content_policies.json        # Content rating policies
└── system_prompts/              # LLM system prompts
```

### Gender-Aware Filtering

When a gender is selected, all gender-tagged options are automatically filtered:
- **Male selected** → Shows male + any-gender items (e.g., "V-taper athletic", "crew cut", "groomed masculine")
- **Female selected** → Shows female + any-gender items (e.g., "hourglass", "hime cut", "coquette feminine")
- **None/Random** → Shows all items from all genders

Restart ComfyUI after editing data files.

## 🚀 Use

Once installed, look for nodes under these categories in ComfyUI:
- `🧙 Wizdroid/Prompts` - All prompt generation nodes
- `🧙 Wizdroid/Analysis` - Image analysis nodes
- `🧙 Wizdroid/Training` - LoRA training nodes

Wire them into your normal image-generation workflows.

##  License

Licensed under the Apache License, Version 2.0.
