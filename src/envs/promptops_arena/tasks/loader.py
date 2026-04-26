"""Task loader: reads JSONL files in this directory."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional


_TASK_DIR = Path(__file__).parent
_FILES = ["math.jsonl", "code.jsonl", "json_extract.jsonl"]


def load_tasks(
    split: Optional[str] = None,
    types: Optional[Iterable[str]] = None,
) -> List[dict]:
    """
    Load all tasks, optionally filtered by split ('train'/'test') and types.

    Returns: list of task dicts.
    """
    out: List[dict] = []
    for fname in _FILES:
        path = _TASK_DIR / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                if split is not None and t.get("split") != split:
                    continue
                if types is not None and t.get("type") not in set(types):
                    continue
                out.append(t)
    return out


def sample_task(rng: random.Random, split: str = "train") -> dict:
    tasks = load_tasks(split=split)
    return rng.choice(tasks)
