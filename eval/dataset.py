"""Loads eval prompt sets (JSONL, one {"prompt": ...} object per line)."""

from __future__ import annotations

import json
from pathlib import Path


def load_prompts(path: str | Path) -> list[str]:
    prompts = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
    return prompts
