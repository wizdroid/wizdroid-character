import re
import hashlib
from typing import Tuple


class WizdroidGenerateFilenameNode:
    """🧙 Generate a filename from text by replacing non-alphanumeric characters with underscores or using a hash."""

    CATEGORY = "🧙 Wizdroid/Utilities"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "generate_filename"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["text", "hash"], {"default": "text"}),
                "max_length": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "case": (["none", "lower", "upper"], {"default": "none"}),
            }
        }

    def generate_filename(self, text: str, mode: str, max_length: int, case: str) -> Tuple[str]:
        """
        Generate a filename from text using one of two modes:
        
        Mode 'text':
        - Replace non-alphabetic characters with underscores
        - Limit to max_length characters
        - Remove leading/trailing underscores
        - Apply case conversion (none, lower, upper)
        
        Mode 'hash':
        - Generate MD5 hash of the text
        - No character limit constraints
        - Always produces consistent, compact unique identifier
        - Useful for long prompts without truncation loss
        """
        if not text.strip():
            return ("filename",)

        if mode == "hash":
            # Generate hash-based filename
            hash_digest = hashlib.md5(text.encode()).hexdigest()
            filename = f"hash_{hash_digest}"
        else:
            # Original text-based mode
            # Replace non-alphanumeric characters (except underscore) with underscore
            filename = re.sub(r'[^a-zA-Z0-9_]', '_', text)

            # Replace multiple consecutive underscores with a single underscore
            filename = re.sub(r'_+', '_', filename)

            # Remove leading and trailing underscores
            filename = filename.strip('_')

            # Truncate to max_length
            if len(filename) > max_length:
                filename = filename[:max_length].rstrip('_')

        # Apply case conversion (only to text mode, hash is already lowercase)
        if case == "lower":
            filename = filename.lower()
        elif case == "upper":
            filename = filename.upper()

        # Ensure we have something
        if not filename:
            filename = "filename"

        return (filename,)


class WizdroidSearchReplaceNode:
    """🧙 Search and replace text in a string."""

    CATEGORY = "🧙 Wizdroid/Utilities"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "search_replace"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "search": ("STRING", {"multiline": False, "default": ""}),
                "replace": ("STRING", {"multiline": False, "default": ""}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
                "use_regex": ("BOOLEAN", {"default": False}),
            }
        }

    def search_replace(
        self,
        text: str,
        search: str,
        replace: str,
        case_sensitive: bool,
        use_regex: bool,
    ) -> Tuple[str]:
        """
        Search and replace text:
        - Supports regex patterns (when use_regex is True)
        - Case-sensitive or insensitive search
        """
        if not search:
            return (text,)

        if use_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                result = re.sub(search, replace, text, flags=flags)
            except re.error as e:
                return (f"[ERROR: Invalid regex - {e}]",)
        else:
            if case_sensitive:
                result = text.replace(search, replace)
            else:
                # Case-insensitive replace
                pattern = re.compile(re.escape(search), re.IGNORECASE)
                result = pattern.sub(replace, text)

        return (result,)


class WizdroidShortenTextNode:
    """🧙 Shorten text to a specified length with optional suffix."""

    CATEGORY = "🧙 Wizdroid/Utilities"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shortened_text",)
    FUNCTION = "shorten_text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "max_length": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1}),
                "suffix": ("STRING", {"multiline": False, "default": "..."}),
                "preserve_words": ("BOOLEAN", {"default": False}),
            }
        }

    def shorten_text(
        self,
        text: str,
        max_length: int,
        suffix: str,
        preserve_words: bool,
    ) -> Tuple[str]:
        """
        Shorten text to max_length:
        - If preserve_words is True, tries to break at word boundaries
        - Adds suffix if text is truncated
        """
        if len(text) <= max_length:
            return (text,)

        if not preserve_words:
            # Simple truncation
            truncated = text[:max_length - len(suffix)] + suffix
            return (truncated,)

        # Preserve words - find the last space before max_length
        truncate_at = max_length - len(suffix)
        last_space = text.rfind(' ', 0, truncate_at)

        if last_space == -1:
            # No space found, just truncate
            truncated = text[:truncate_at] + suffix
        else:
            truncated = text[:last_space] + suffix

        return (truncated,)


# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "WizdroidGenerateFilename": WizdroidGenerateFilenameNode,
    "WizdroidSearchReplace": WizdroidSearchReplaceNode,
    "WizdroidShortenText": WizdroidShortenTextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WizdroidGenerateFilename": "🧙 Generate Filename",
    "WizdroidSearchReplace": "🧙 Search & Replace",
    "WizdroidShortenText": "🧙 Shorten Text",
}
