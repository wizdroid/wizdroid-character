from __future__ import annotations

import re
from typing import Optional


CONTENT_RATING_CHOICES = ("SFW only", "NSFW allowed")

# Conservative blocklist: goal is to prevent accidental NSFW prompts.
# (This is not a comprehensive safety classifier.)
_NSFW_RE = re.compile(
    r"\b("
    r"nsfw|"
    r"nude|nudity|naked|topless|bottomless|nipples?|areolae?|"
    r"pussy|cunt|cock|dick|penis|vagina|"
    r"blowjob|handjob|masturbat(e|ion)|orgasm|cum|ejaculat(e|ion)|"
    r"porn|porno|hentai|erotic|sex\s*act|anal|deep\s*throat"
    r")\b",
    flags=re.IGNORECASE,
)


def looks_nsfw(text: str) -> bool:
    return bool(_NSFW_RE.search(text or ""))


def enforce_sfw(text: str) -> Optional[str]:
    """Return an error string if the text appears NSFW, else None."""

    if looks_nsfw(text):
        return "Blocked: potential NSFW content detected."
    return None
