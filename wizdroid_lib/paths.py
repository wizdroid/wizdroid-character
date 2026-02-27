from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SHARED_DIR = DATA_DIR / "shared"


def data_path(relative_name: str) -> Path:
    return DATA_DIR / relative_name


def shared_path(relative_name: str) -> Path:
    return SHARED_DIR / relative_name
