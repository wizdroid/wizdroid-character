from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def data_path(relative_name: str) -> Path:
    return DATA_DIR / relative_name
