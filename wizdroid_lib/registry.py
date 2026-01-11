"""Centralized data registry for wizdroid-character nodes.

Provides a single point of access for all JSON data with efficient caching
to avoid redundant file loads across nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

from .data_files import load_json
from .paths import DATA_DIR


class DataRegistry:
    """Singleton registry for cached data access across all nodes."""
    
    _instance: Optional["DataRegistry"] = None
    _data: Dict[str, Any] = {}
    _mtimes: Dict[str, int] = {}
    
    def __new__(cls) -> "DataRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get cached data by key (filename without .json extension)."""
        instance = cls()
        filename = f"{key}.json" if not key.endswith(".json") else key
        
        # Check if file has changed
        try:
            path = DATA_DIR / filename
            if not path.exists():
                return default
            current_mtime = int(path.stat().st_mtime_ns)
            
            if key in instance._data and instance._mtimes.get(key) == current_mtime:
                return instance._data[key]
            
            # Load fresh data
            data = load_json(filename)
            instance._data[key] = data
            instance._mtimes[key] = current_mtime
            return data
            
        except Exception:
            return default
    
    @classmethod
    def get_character_options(cls) -> Dict[str, Any]:
        """Get character_options.json data."""
        return cls.get("character_options", {})
    
    @classmethod
    def get_prompt_styles(cls) -> Dict[str, Any]:
        """Get prompt_styles.json data."""
        return cls.get("prompt_styles", {})
    
    @classmethod
    def get_regions(cls) -> Dict[str, Any]:
        """Get regions.json data."""
        return cls.get("regions", {})
    
    @classmethod
    def get_countries(cls) -> Dict[str, Any]:
        """Get countries.json data."""
        return cls.get("countries", {})
    
    @classmethod
    def get_cultures(cls) -> Dict[str, Any]:
        """Get cultures.json data."""
        return cls.get("cultures", {})
    
    @classmethod
    def get_scene_data(cls) -> Dict[str, Any]:
        """Get scene_data.json data."""
        return cls.get("scene_data", {})
    
    @classmethod
    def get_meta_prompt_options(cls) -> Dict[str, Any]:
        """Get meta_prompt_options.json data."""
        return cls.get("meta_prompt_options", {})
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached data (useful for testing)."""
        instance = cls()
        instance._data.clear()
        instance._mtimes.clear()


# Convenience functions for common access patterns
def get_character_options() -> Dict[str, Any]:
    """Get character options with caching."""
    return DataRegistry.get_character_options()


def get_prompt_styles() -> Dict[str, Any]:
    """Get prompt styles with caching."""
    return DataRegistry.get_prompt_styles()
