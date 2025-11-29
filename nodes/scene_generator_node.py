import json
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

try:
    import requests
except ImportError:
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RANDOM_LABEL = "Random"
NONE_LABEL = "none"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

logger = logging.getLogger(__name__)

# Scene prompt cache
_SCENE_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 50

# Scene data cache (JSON-driven)
_SCENE_DATA_CACHE: Optional[Tuple[int, Dict[str, Any]]] = None
_DEFAULT_SCENE_DATA = {
    "scene_categories": {
        "Open Scene": ["dynamic cinematic scene"],
    },
    "scene_moods": ["mysterious"],
    "time_of_day": ["night"],
    "weather_conditions": ["clear sky"],
    "population_density": ["single figure"],
}

def _load_scene_data() -> Dict[str, Any]:
    """Load scene configuration from JSON (with mtime caching)."""
    global _SCENE_DATA_CACHE
    path = DATA_DIR / "scene_data.json"
    try:
        mtime = int(path.stat().st_mtime_ns)
    except FileNotFoundError:
        logger.warning("[SceneGenerator] scene_data.json not found; using defaults")
        return _DEFAULT_SCENE_DATA

    cached = _SCENE_DATA_CACHE
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.exception("[SceneGenerator] Failed to load scene_data.json: %s", exc)
        payload = _DEFAULT_SCENE_DATA

    _SCENE_DATA_CACHE = (mtime, payload)
    return payload


def _with_random(options: List[str]) -> Tuple[str, ...]:
    """Prepend Random and none options to a list."""
    values = [RANDOM_LABEL, NONE_LABEL]
    for opt in options:
        if opt != NONE_LABEL:
            values.append(opt)
    return tuple(values)


def _choose(value: Optional[str], options: List[str], rng: random.Random) -> Optional[str]:
    """Select a value, handling Random and none cases."""
    if value == RANDOM_LABEL:
        pool = [opt for opt in options if opt != NONE_LABEL]
        return rng.choice(pool) if pool else None
    return None if value == NONE_LABEL or value is None else value


def _cache_key(data: Dict) -> str:
    """Generate cache key for scene caching."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class SceneGeneratorNode:
    """
    Generates vivid scene prompts for any imaginable scenario.
    From crime scenes to fantasy battles, horror to celebrations.
    """
    
    CATEGORY = "Wizdroid/scene"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("scene_prompt", "negative_prompt", "preview")
    FUNCTION = "generate_scene"

    @classmethod
    def INPUT_TYPES(cls):
        scene_data = _load_scene_data()
        scene_categories = scene_data.get("scene_categories") or _DEFAULT_SCENE_DATA["scene_categories"]
        scene_moods = scene_data.get("scene_moods") or _DEFAULT_SCENE_DATA["scene_moods"]
        time_of_day = scene_data.get("time_of_day") or _DEFAULT_SCENE_DATA["time_of_day"]
        weather_conditions = scene_data.get("weather_conditions") or _DEFAULT_SCENE_DATA["weather_conditions"]
        population_density = scene_data.get("population_density") or _DEFAULT_SCENE_DATA["population_density"]

        # Flatten all scene types for the dropdown
        all_scenes: List[str] = []
        for scenes in scene_categories.values():
            all_scenes.extend(scenes)
        if not all_scenes:
            all_scenes = _DEFAULT_SCENE_DATA["scene_categories"]["Open Scene"]
        
        # Load prompt styles
        try:
            prompt_styles = json.loads((DATA_DIR / "prompt_styles.json").read_text())
        except:
            prompt_styles = {"SDXL": {}, "Flux": {}, "SD3": {}}
        
        ollama_models = cls._collect_ollama_models()
        category_names = list(scene_categories.keys()) or list(_DEFAULT_SCENE_DATA["scene_categories"].keys())
        
        return {
            "required": {
                # === LLM Settings ===
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0]}),
                "prompt_style": (tuple(prompt_styles.keys()), {"default": "SDXL"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 200, "min": 50, "max": 500, "step": 10}),
                
                # === Scene Selection ===
                "scene_category": (_with_random(category_names), {"default": RANDOM_LABEL}),
                "specific_scene": (_with_random(all_scenes), {"default": RANDOM_LABEL}),
                "custom_scene": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe any scene you can imagine..."}),
                
                # === Scene Atmosphere ===
                "mood": (_with_random(scene_moods), {"default": RANDOM_LABEL}),
                "time_of_day": (_with_random(time_of_day), {"default": RANDOM_LABEL}),
                "weather": (_with_random(weather_conditions), {"default": NONE_LABEL}),
                "population": (_with_random(population_density), {"default": NONE_LABEL}),
                
                # === Visual Style ===
                "chaos_level": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "detail_level": ("INT", {"default": 7, "min": 1, "max": 10, "step": 1}),
                "cinematic": ("BOOLEAN", {"default": True}),
                
                # === Additional Elements ===
                "include_characters": ("BOOLEAN", {"default": True}),
                "include_creatures": ("BOOLEAN", {"default": False}),
                "include_vehicles": ("BOOLEAN", {"default": False}),
                "include_effects": ("BOOLEAN", {"default": True}),
                
                # === Custom Additions ===
                "additional_elements": ("STRING", {"multiline": True, "default": "", "placeholder": "Add specific elements: 'giant tentacles, floating debris, red emergency lights'"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
            }
        }

    def generate_scene(
        self,
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        temperature: float,
        max_tokens: int,
        scene_category: str,
        specific_scene: str,
        custom_scene: str,
        mood: str,
        time_of_day: str,
        weather: str,
        population: str,
        chaos_level: int,
        detail_level: int,
        cinematic: bool,
        include_characters: bool,
        include_creatures: bool,
        include_vehicles: bool,
        include_effects: bool,
        additional_elements: str,
        seed: int,
    ):
        rng = random.Random(seed)
        
        # Load prompt styles
        try:
            prompt_styles = json.loads((DATA_DIR / "prompt_styles.json").read_text())
        except:
            prompt_styles = {"SDXL": {"guidance": "comma-separated descriptors", "negative_prompt": ""}}
        
        style_meta = prompt_styles.get(prompt_style, prompt_styles.get("SDXL", {}))
        scene_data = _load_scene_data()
        scene_categories = scene_data.get("scene_categories") or _DEFAULT_SCENE_DATA["scene_categories"]
        scene_moods = scene_data.get("scene_moods") or _DEFAULT_SCENE_DATA["scene_moods"]
        time_of_day_list = scene_data.get("time_of_day") or _DEFAULT_SCENE_DATA["time_of_day"]
        weather_conditions = scene_data.get("weather_conditions") or _DEFAULT_SCENE_DATA["weather_conditions"]
        population_density = scene_data.get("population_density") or _DEFAULT_SCENE_DATA["population_density"]
        category_names = list(scene_categories.keys()) or list(_DEFAULT_SCENE_DATA["scene_categories"].keys())
        
        # Resolve scene selection
        resolved_category = _choose(scene_category, category_names, rng)
        
        # If specific_scene is Random, pick from the resolved category
        if specific_scene == RANDOM_LABEL:
            if resolved_category:
                scene_pool = scene_categories.get(resolved_category, [])
            else:
                # Pick from all scenes
                scene_pool = [s for scenes in scene_categories.values() for s in scenes]
            resolved_scene = rng.choice(scene_pool) if scene_pool else "mysterious scene"
        else:
            resolved_scene = _choose(specific_scene, [], rng)
        
        # Custom scene overrides selection
        custom = custom_scene.strip()
        if custom:
            resolved_scene = custom
        
        # Resolve atmosphere
        resolved_mood = _choose(mood, scene_moods, rng)
        resolved_time = _choose(time_of_day, time_of_day_list, rng)
        resolved_weather = _choose(weather, weather_conditions, rng)
        resolved_population = _choose(population, population_density, rng)
        
        # Build selections dict for caching
        selections = {
            "scene": resolved_scene,
            "category": resolved_category,
            "mood": resolved_mood,
            "time": resolved_time,
            "weather": resolved_weather,
            "population": resolved_population,
            "chaos": chaos_level,
            "detail": detail_level,
            "cinematic": cinematic,
            "characters": include_characters,
            "creatures": include_creatures,
            "vehicles": include_vehicles,
            "effects": include_effects,
            "additions": additional_elements.strip(),
            "style": prompt_style,
            "temp": temperature,
        }
        
        # Check cache
        cache_key = _cache_key(selections)
        if cache_key in _SCENE_CACHE:
            scene_prompt = _SCENE_CACHE[cache_key]
        else:
            scene_prompt = self._invoke_llm(
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                prompt_style=prompt_style,
                style_meta=style_meta,
                selections=selections,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Cache with size limit
            if len(_SCENE_CACHE) >= _MAX_CACHE_SIZE:
                _SCENE_CACHE.pop(next(iter(_SCENE_CACHE)))
            _SCENE_CACHE[cache_key] = scene_prompt
        
        negative_prompt = style_meta.get("negative_prompt", "")
        
        return scene_prompt, negative_prompt, scene_prompt

    @staticmethod
    def _invoke_llm(
        ollama_url: str,
        ollama_model: str,
        prompt_style: str,
        style_meta: Dict,
        selections: Dict,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate scene prompt via Ollama."""
        
        # Build dynamic system prompt based on scene type
        scene = selections.get("scene", "")
        category = selections.get("category", "")
        
        system_prompt = (
            "You are an expert scene description generator for AI image generation. "
            "Create vivid, detailed scene prompts that capture atmosphere, action, and visual elements. "
            "Your prompts should be cinematically composed and visually striking. "
            "Output ONLY the prompt text - no explanations, no markdown, no meta commentary. "
            "Never start with 'Here', 'This', 'Prompt', or 'A scene of'."
        )
        
        # Build scene description
        scene_parts = [f"Scene: {scene}"]
        
        if selections.get("mood"):
            scene_parts.append(f"Mood: {selections['mood']}")
        if selections.get("time"):
            scene_parts.append(f"Time: {selections['time']}")
        if selections.get("weather"):
            scene_parts.append(f"Weather: {selections['weather']}")
        if selections.get("population"):
            scene_parts.append(f"Population: {selections['population']}")
        
        scene_parts.append(f"Chaos level: {selections.get('chaos', 5)}/10")
        scene_parts.append(f"Detail level: {selections.get('detail', 7)}/10")
        
        if selections.get("cinematic"):
            scene_parts.append("Style: Cinematic, dramatic composition")
        
        # Element inclusions
        elements = []
        if selections.get("characters"):
            elements.append("include relevant characters/people")
        if selections.get("creatures"):
            elements.append("include monsters/creatures/animals")
        if selections.get("vehicles"):
            elements.append("include vehicles if relevant")
        if selections.get("effects"):
            elements.append("include visual effects (smoke, sparks, magic, etc.)")
        
        if elements:
            scene_parts.append(f"Elements: {', '.join(elements)}")
        
        if selections.get("additions"):
            scene_parts.append(f"Additional elements: {selections['additions']}")
        
        guidance = style_meta.get('guidance', 'Single paragraph with comma-separated descriptors')
        token_limit = min(max_tokens, style_meta.get('token_limit', 200))
        
        user_prompt = (
            f"Create a {prompt_style} scene prompt.\n"
            f"{chr(10).join(scene_parts)}\n"
            f"Format: {guidance}\n"
            f"Keep under {token_limit} tokens. Be vivid and specific. Output only the prompt:"
        )

        try:
            if requests is None:
                return "[Please install 'requests': pip install requests]"
            
            payload = {
                "model": ollama_model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": token_limit * 4,
                }
            }
            
            response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            
            # Clean up common LLM prefixes
            prefixes = ["Here is", "Here's", "This prompt", "Prompt:", "A scene of", "Scene:"]
            for prefix in prefixes:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].lstrip(": ")
            
            return result or "[Empty response from Ollama]"
            
        except requests.exceptions.ConnectionError:
            return "[Ollama server not running. Please start Ollama.]"
        except requests.exceptions.Timeout:
            return "[Ollama request timed out]"
        except Exception as e:
            logger.exception(f"[SceneGenerator] Error invoking LLM: {e}")
            return f"[Error: {str(e)}]"

    @staticmethod
    def _collect_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
        """Fetch available Ollama models."""
        try:
            if requests is None:
                return ["install_requests_library"]
            
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return models if models else ["no_models_found"]
        except:
            return ["ollama_not_running"]


NODE_CLASS_MAPPINGS = {
    "SceneGeneratorNode": SceneGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeneratorNode": "Scene Generator",
}
