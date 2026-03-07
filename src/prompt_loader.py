from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.config import ACTIVE_FILE,PROMPTS_DIR

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_active_prompt_version(prompt_name: str) -> str:
    active_config = load_json(ACTIVE_FILE)
    if prompt_name not in active_config:
        raise KeyError(f"No active version configured for prompt: {prompt_name}")
    return active_config[prompt_name]


def load_prompt(prompt_name: str, version: str | None = None) -> Dict[str, Any]:
    version = version or get_active_prompt_version(prompt_name)
    prompt_path = PROMPTS_DIR / version / f"{prompt_name}.json"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return load_json(prompt_path)