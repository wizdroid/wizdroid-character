# Wizdroid Character Nodes for ComfyUI

A comprehensive collection of custom nodes for ComfyUI that provide AI-powered prompt generation tools using Ollama. These nodes help create detailed, professional prompts for text-to-image generation across various domains including character creation, fashion, fantasy scenes, and image analysis.

## Features

### Character Prompt Builder
- **Purpose**: Generates detailed character prompts for image generation with extensive customization options
- **Attributes**: Gender, age group, body type, hair color/style, eye color, facial expressions, poses, makeup, fashion styles, and backgrounds
- **Integration**: Ollama LLM integration for intelligent prompt crafting
- **Output**: Positive prompt, negative prompt, and follow-up questions for refinement

### Fantasy Scene Builder
- **Purpose**: Creates atmospheric fantasy and horror scene prompts
- **Elements**: Fantasy themes, subjects, environments, lighting, visual styles, textures, compositions, and special effects
- **Integration**: Ollama LLM for generating vivid, immersive scene descriptions
- **Output**: Scene prompt and negative prompt

### Upcycled Fashion Node
- **Purpose**: Generates professional prompts for sustainable fashion featuring everyday objects transformed into glamorous designer wear
- **Materials**: 50+ upcycled materials including plastic bags, bubble wrap, tarpaulin, garbage bags, latex, and more
- **Integration**: Ollama LLM for creative upcycling prompt generation
- **Output**: High-fashion photography prompts showcasing sustainable design innovation

### Fashion Supermodel Node
- **Purpose**: Generates professional fashion photography prompts featuring contemporary regional fashion styles
- **Features**: 28 glamour enhancement options (subtle, sensual, erotic, royal, religious themes), regional fashion from around the world
- **Integration**: Ollama LLM for modern regional fashion prompt creation
- **Output**: Detailed fashion photography prompt with regional style focus

### Photo Aspect Extractor
- **Purpose**: Analyzes images using vision models to extract specific aspects
- **Modes**: Clothes, pose, style, background, expression, lighting, hair, makeup, accessories, camera settings, composition, color palette
- **Integration**: Ollama vision models (LLaVA, Florence, etc.) for image analysis
- **Output**: Extracted aspect descriptions for prompt engineering

### Pose Extraction Node
- **Purpose**: Extracts detailed pose descriptions from character images
- **Features**: Analyzes body position, stance, gestures, camera angles, and framing
- **Integration**: Ollama vision models for precise pose analysis
- **Output**: Pose-focused prompts for consistent character positioning

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
- `character_options.json`: Character attributes and options
- `regions.json`: Regional fashion areas
- `fantasy_options.json`: Fantasy scene elements
- `prompt_styles.json`: Output format styles for different AI models
- `glamour_options.json`: Glamour enhancement styles
- `upcycled_materials.json`: Everyday materials for upcycled fashion
- `followup_questions.json`: Refinement questions for prompts

## Usage

After installation, the nodes will appear in ComfyUI under the "Wizdroid/character" and "Wizdroid/fantasy" categories.

### Common Workflow
1. Use **Character Prompt Builder** or **Fashion Supermodel Node** to generate base prompts
2. Use **Photo Aspect Extractor** or **Pose Extraction Node** to analyze reference images
3. Combine extracted elements with generated prompts for refined results
4. Use **Fantasy Scene Builder** for atmospheric scene creation

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