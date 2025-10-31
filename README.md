# Wizdroid Character Nodes for ComfyUI

A collection of custom nodes for ComfyUI that generate professional fashion supermodel prompts featuring traditional cultural outfits from various countries with glamorous styling.

## Features

- **Fashion Supermodel Node**: Generates detailed prompts for fashion photography featuring traditional cultural attire enhanced with various glamour styles
- **Cultural Integration**: Supports traditional outfits from multiple countries
- **Glamour Enhancements**: 28 different glamour enhancement options including sultry, sensual, erotic, royal, and religious themes
- **Ollama Integration**: Uses local Ollama models for prompt generation
- **Multiple Prompt Styles**: Supports different AI model formats (SDXL, etc.)

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

3. Ensure Ollama is running with a compatible model (e.g., llama2, mistral)

## Usage

After installation, the nodes will appear in ComfyUI under the "Wizdroid/character" category.

### Fashion Supermodel Node

- **Inputs**:
  - `ollama_url`: URL of your Ollama server (default: http://localhost:11434)
  - `ollama_model`: Available Ollama models
  - `prompt_style`: Output format style (SDXL, etc.)
  - `country`: Traditional outfit country (or Random)
  - `glamour_enhancement`: Glamour styling approach (or Random)
  - `gender`: Model gender identity (or Random)

- **Output**: Generated fashion photography prompt

## Configuration

The node uses JSON configuration files in the `data/` directory:
- `character_options.json`: Gender and other character options
- `countries.json`: Available countries for traditional outfits
- `prompt_styles.json`: Prompt formatting styles for different AI models
- `glamour_options.json`: Glamour enhancement options

## Contributing

Feel free to contribute by:
- Adding more countries and traditional outfits
- Expanding glamour enhancement options
- Improving prompt generation logic
- Adding new node types

## License

[Add your license here]