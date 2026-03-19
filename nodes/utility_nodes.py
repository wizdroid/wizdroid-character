import re
import hashlib
import time
from typing import Tuple


class WizdroidGenerateFilenameNode:
    """🧙 Generate flexible filenames with timestamps, hashes, prepend/append, and random suffixes."""

    CATEGORY = "🧙 Wizdroid/Utilities"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "generate_filename"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["text", "hash", "timestamp", "timestamp_hash"], {"default": "text"}),
                "prepend": ("STRING", {"default": ""}),
                "append": ("STRING", {"default": ""}),
                "separator": (["_", "-", ""], {"default": "_"}),
                "include_date_prefix": ("BOOLEAN", {"default": False}),
                "include_random_suffix": ("BOOLEAN", {"default": False}),
                "random_suffix_length": ("INT", {"default": 4, "min": 2, "max": 16, "step": 1}),
                "max_length": ("INT", {"default": 255, "min": 1, "max": 1024, "step": 1}),
                "case": (["none", "lower", "upper"], {"default": "none"}),
            }
        }

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Replace non-alphanumeric characters with underscores."""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', text)
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')

    @staticmethod
    def _get_date_prefix() -> str:
        """Get current date in YYYYMMDD format."""
        return time.strftime("%Y%m%d")

    @staticmethod
    def _get_timestamp_suffix(include_ms: bool = False) -> str:
        """Get current timestamp (epoch). If include_ms, includes milliseconds."""
        if include_ms:
            ts = int(time.time() * 1000)
            return f"ts{ts}"
        else:
            ts = int(time.time())
            return f"ts{ts}"

    @staticmethod
    def _get_random_suffix(length: int = 4) -> str:
        """Generate random hex suffix."""
        import random
        hex_chars = "0123456789abcdef"
        return "".join(random.choice(hex_chars) for _ in range(length))

    def generate_filename(
        self,
        text: str,
        mode: str,
        prepend: str,
        append: str,
        separator: str,
        include_date_prefix: bool,
        include_random_suffix: bool,
        random_suffix_length: int,
        max_length: int,
        case: str,
    ) -> Tuple[str]:
        """
        Generate a filename with flexible options.
        
        Modes:
        - 'text': Sanitize and truncate text, optional char limit
        - 'hash': MD5 hash of text (no truncation)
        - 'timestamp': Unix timestamp (uses text as fallback if empty)
        - 'timestamp_hash': Timestamp + hash of text (best for unique, traceable files)
        
        Features:
        - Prepend/Append: Add custom prefixes/suffixes
        - Date prefix: Automatically prepend YYYYMMDD
        - Random suffix: Add random hex string for extra uniqueness
        - Separator: Control how parts are joined
        - Case conversion: Apply lower/upper casing
        """
        parts = []

        # 1. Date prefix (if enabled)
        if include_date_prefix:
            parts.append(self._get_date_prefix())

        # 2. Prepend text
        if prepend.strip():
            prepend_clean = self._sanitize_text(prepend.strip())
            if prepend_clean:
                parts.append(prepend_clean)

        # 3. Main content based on mode
        if mode == "hash":
            if text.strip():
                hash_digest = hashlib.md5(text.encode()).hexdigest()
                parts.append(f"hash{hash_digest}")
            else:
                # Fallback: if no text, use timestamp
                parts.append(self._get_timestamp_suffix())
        elif mode == "timestamp":
            parts.append(self._get_timestamp_suffix())
        elif mode == "timestamp_hash":
            ts = self._get_timestamp_suffix()
            if text.strip():
                hash_digest = hashlib.md5(text.encode()).hexdigest()[:8]  # Short hash
                parts.append(f"{ts}_{hash_digest}")
            else:
                parts.append(ts)
        else:  # mode == "text"
            if text.strip():
                text_clean = self._sanitize_text(text)
                if len(text_clean) > max_length:
                    text_clean = text_clean[:max_length].rstrip('_')
                parts.append(text_clean)
            else:
                # Fallback: use timestamp if text is empty
                parts.append(self._get_timestamp_suffix())

        # 4. Random suffix (if enabled)
        if include_random_suffix:
            parts.append(self._get_random_suffix(random_suffix_length))

        # 5. Append text
        if append.strip():
            append_clean = self._sanitize_text(append.strip())
            if append_clean:
                parts.append(append_clean)

        # 6. Join with separator
        filename = separator.join(parts)

        # 7. Apply case conversion
        if case == "lower":
            filename = filename.lower()
        elif case == "upper":
            filename = filename.upper()

        # 8. Final truncation to max_length
        if len(filename) > max_length:
            filename = filename[:max_length].rstrip('_').rstrip('-').rstrip()

        # 9. Ensure we have something
        if not filename:
            filename = "file"

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
