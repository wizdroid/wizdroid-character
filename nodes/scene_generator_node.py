import json
import random
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
import logging

from wizdroid_lib.constants import CONTENT_RATING_CHOICES, DEFAULT_OLLAMA_URL, NONE_LABEL, RANDOM_LABEL
from wizdroid_lib.content_safety import enforce_sfw
from wizdroid_lib.data_files import load_json
from wizdroid_lib.helpers import choose, with_random
from wizdroid_lib.ollama_client import collect_models, generate_text
from wizdroid_lib.system_prompts import load_system_prompt_text
from wizdroid_lib.paths import DATA_DIR

logger = logging.getLogger(__name__)

# === Caching Globals ===
_SCENE_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 50

# Data Caches (mtime based)
_SCENE_DATA_CACHE: Optional[Tuple[int, Dict[str, Any]]] = None
_PROMPT_STYLES_CACHE: Optional[Tuple[int, Dict[str, Any]]] = None

# Model Cache (Time-to-live based)
_MODELS_CACHE: List[str] = []
_MODELS_LAST_UPDATE: float = 0
_MODELS_TTL = 60.0  # Refresh models at most every 60 seconds

_DEFAULT_SCENE_DATA = {
    "scene_categories": {"Open Scene": ["dynamic cinematic scene"]},
    "scene_moods": ["mysterious"],
    "time_of_day": ["night"],
    "weather_conditions": ["clear sky"],
    "population_density": ["single figure"],
}

def _load_json_cached(filename: str, default: Dict, cache_var: Optional[Tuple[int, Dict]]) -> Tuple[Tuple[int, Dict], Dict]:
    """Generic mtime-based JSON loader."""
    path = DATA_DIR / filename
    try:
        # If file doesn't exist, return default and no cache update
        if not path.exists():
            return (0, default), default
            
        mtime = int(path.stat().st_mtime_ns)
    except OSError:
        return (0, default), default

    if cache_var and cache_var[0] == mtime:
        return cache_var, cache_var[1]

    try:
        payload = load_json(filename)
        return (mtime, payload), payload
    except Exception as exc:
        logger.error(f"[SceneGenerator] Failed to load {filename}: {exc}")
        return (0, default), default

def _get_cached_models(url: str) -> List[str]:
    """Get Ollama models with TTL caching to prevent UI blocking."""
    global _MODELS_CACHE, _MODELS_LAST_UPDATE
    
    now = time.time()
    if not _MODELS_CACHE or (now - _MODELS_LAST_UPDATE > _MODELS_TTL):
        try:
            models = collect_models(url)
            if models:
                _MODELS_CACHE = models
                _MODELS_LAST_UPDATE = now
        except Exception as e:
            logger.warning(f"[SceneGenerator] Could not fetch models: {e}")
            if not _MODELS_CACHE:
                return ["llama3:latest"] # Fallback
    return _MODELS_CACHE

def _cache_key(data: Dict) -> str:
    """Generate cache key for scene caching."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class SceneGeneratorNode:
    """
    Generates vivid scene prompts for any imaginable scenario.
    Optimized with caching and robust error handling.
    """
    
    CATEGORY = "Wizdroid/scene"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("scene_prompt", "negative_prompt", "preview")
    FUNCTION = "generate_scene"

    @classmethod
    def INPUT_TYPES(cls):
        global _SCENE_DATA_CACHE, _PROMPT_STYLES_CACHE
        
        # Load data with caching
        _SCENE_DATA_CACHE, scene_data = _load_json_cached("scene_data.json", _DEFAULT_SCENE_DATA, _SCENE_DATA_CACHE)
        _PROMPT_STYLES_CACHE, prompt_styles = _load_json_cached("prompt_styles.json", {"SDXL": {}, "Flux": {}, "SD3": {}}, _PROMPT_STYLES_CACHE)

        # Extract lists safely
        scene_categories = scene_data.get("scene_categories", _DEFAULT_SCENE_DATA["scene_categories"])
        scene_moods = scene_data.get("scene_moods", _DEFAULT_SCENE_DATA["scene_moods"])
        time_of_day = scene_data.get("time_of_day", _DEFAULT_SCENE_DATA["time_of_day"])
        weather_conditions = scene_data.get("weather_conditions", _DEFAULT_SCENE_DATA["weather_conditions"])
        population_density = scene_data.get("population_density", _DEFAULT_SCENE_DATA["population_density"])

        # Flatten all scene types for the dropdown
        all_scenes: List[str] = []
        for scenes in scene_categories.values():
            all_scenes.extend(scenes)
        if not all_scenes:
            all_scenes = ["dynamic scene"]
        
        # Get models (cached)
        ollama_models = _get_cached_models(DEFAULT_OLLAMA_URL)
        category_names = list(scene_categories.keys())
        
        return {
            "required": {
                # === LLM Settings ===
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": (tuple(ollama_models), {"default": ollama_models[0] if ollama_models else ""}),
                "content_rating": (CONTENT_RATING_CHOICES, {"default": "SFW"}),
                "prompt_style": (tuple(prompt_styles.keys()), {"default": "SDXL"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 200, "min": 50, "max": 500, "step": 10}),
                
                # === Scene Selection ===
                "scene_category": (with_random(category_names), {"default": RANDOM_LABEL}),
                "specific_scene": (with_random(all_scenes), {"default": RANDOM_LABEL}),
                "custom_scene": ("STRING", {"multiline": True, "default": "", "placeholder": "Describe any scene you can imagine..."}),
                
                # === Scene Atmosphere ===
                "mood": (with_random(scene_moods), {"default": RANDOM_LABEL}),
                "time_of_day": (with_random(time_of_day), {"default": RANDOM_LABEL}),
                "weather": (with_random(weather_conditions), {"default": NONE_LABEL}),
                "population": (with_random(population_density), {"default": NONE_LABEL}),
                
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
        content_rating: str,
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
        global _SCENE_DATA_CACHE, _PROMPT_STYLES_CACHE
        rng = random.Random(seed)
        
        # Ensure data is loaded (using cache)
        _SCENE_DATA_CACHE, scene_data = _load_json_cached("scene_data.json", _DEFAULT_SCENE_DATA, _SCENE_DATA_CACHE)
        _PROMPT_STYLES_CACHE, prompt_styles = _load_json_cached("prompt_styles.json", {"SDXL": {}}, _PROMPT_STYLES_CACHE)
        
        style_meta = prompt_styles.get(prompt_style, prompt_styles.get("SDXL", {}))
        
        # Extract lists
        scene_categories = scene_data.get("scene_categories", {})
        
        # Resolve scene selection
        resolved_category = choose(scene_category, list(scene_categories.keys()), rng)
        
        # If specific_scene is Random, pick from the resolved category
        if specific_scene == RANDOM_LABEL:
            if resolved_category:
                scene_pool = scene_categories.get(resolved_category, [])
            else:
                # Pick from all scenes
                scene_pool = [s for scenes in scene_categories.values() for s in scenes]
            resolved_scene = rng.choice(scene_pool) if scene_pool else "mysterious scene"
        else:
            resolved_scene = choose(specific_scene, [], rng)
        
        # Custom scene overrides selection
        if custom_scene.strip():
            resolved_scene = custom_scene.strip()
        
        # Resolve atmosphere
        resolved_mood = choose(mood, scene_data.get("scene_moods", []), rng)
        resolved_time = choose(time_of_day, scene_data.get("time_of_day", []), rng)
        resolved_weather = choose(weather, scene_data.get("weather_conditions", []), rng)
        resolved_population = choose(population, scene_data.get("population_density", []), rng)
        
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
            "content_rating": content_rating,
        }
        
        # Check cache
        cache_key = _cache_key(selections)
        if cache_key in _SCENE_CACHE:
            scene_prompt = _SCENE_CACHE[cache_key]
        else:
            scene_prompt = self._invoke_llm(
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                content_rating=content_rating,
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
        content_rating: str,
        prompt_style: str,
        style_meta: Dict,
        selections: Dict,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate scene prompt via Ollama."""
        
        system_prompt = load_system_prompt_text("system_prompts/scene_generator_system.txt", content_rating)
        
        # Build scene description parts
        parts = [f"Scene: {selections.get('scene', 'unknown')}"]
        
        # Optional attributes
        for key, label in [
            ("mood", "Mood"), ("time", "Time"), ("weather", "Weather"), ("population", "Population")
        ]:
            if val := selections.get(key):
                parts.append(f"{label}: {val}")
        
        parts.append(f"Chaos level: {selections.get('chaos', 5)}/10")
        parts.append(f"Detail level: {selections.get('detail', 7)}/10")
        
        if selections.get("cinematic"):
            parts.append("Style: Cinematic, dramatic composition")
        
        # Element inclusions
        elements = []
        if selections.get("characters"): elements.append("include relevant characters/people")
        if selections.get("creatures"): elements.append("include monsters/creatures/animals")
        if selections.get("vehicles"): elements.append("include vehicles if relevant")
        if selections.get("effects"): elements.append("include visual effects (smoke, sparks, magic)")
        
        if elements:
            parts.append(f"Elements: {', '.join(elements)}")
        
        if add := selections.get("additions"):
            parts.append(f"Additional elements: {add}")
        
        guidance = style_meta.get('guidance', 'Single paragraph with comma-separated descriptors')
        token_limit = min(max_tokens, style_meta.get('token_limit', 200))
        
        user_prompt = (
            f"Create a {prompt_style} scene prompt.\n"
            f"{chr(10).join(parts)}\n"
            f"Format: {guidance}\n"
            f"Keep under {token_limit} tokens. Be vivid and specific. Output only the prompt:"
        )

        ok, result = generate_text(
            ollama_url=ollama_url,
            model=ollama_model,
            prompt=user_prompt,
            system=system_prompt,
            options={
                "temperature": float(temperature),
                "num_predict": int(token_limit) * 4,
            },
            timeout=120,
        )

        if not ok:
            return f"[Error: {result}]"

        # Clean up common LLM prefixes
        result = result.strip()
        prefixes = ["Here is", "Here's", "This prompt", "Prompt:", "A scene of", "Scene:", "Sure, here"]
        for prefix in prefixes:
            if result.lower().startswith(prefix.lower()):
                # Find the first colon or newline after the prefix to strip
                split_idx = len(prefix)
                # If there's a colon right after, skip it
                if split_idx < len(result) and result[split_idx] == ':':
                    split_idx += 1
                result = result[split_idx:].strip()

        if content_rating == "SFW":
            err = enforce_sfw(result)
            if err:
                return "[Blocked: potential NSFW content detected. Switch content_rating to 'Mixed' or 'NSFW'.]"

        return result or "[Empty response from Ollama]"


NODE_CLASS_MAPPINGS = {
    "SceneGeneratorNode": SceneGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneGeneratorNode": "Scene Generator",
}
