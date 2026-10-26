"""Config loading, path resolution, and seeding."""

import os
import random
from pathlib import Path

import numpy as np
import yaml
from dotenv import find_dotenv, load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(relative: str) -> Path:
    """Resolve a config path relative to the repo root."""
    return REPO_ROOT / relative


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def groq_api_key() -> str:
    # Looks for .env from the current directory upwards.
    load_dotenv(find_dotenv(usecwd=True))
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key
