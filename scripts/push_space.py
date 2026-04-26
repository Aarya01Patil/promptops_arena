"""Create + push the HF Space for PromptOps Arena."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = os.environ.get("SPACE_ID", "Dar3devil/promptops-arena")


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        token=token,
    )

    ignore = [
        "outputs/*",
        "outputs/**",
        ".venv/*",
        ".venv/**",
        ".git/*",
        ".git/**",
        "__pycache__/*",
        "**/__pycache__/**",
        "*.pyc",
        "node_modules/**",
        ".pytest_cache/**",
        ".mypy_cache/**",
        ".ruff_cache/**",
        ".cursor/**",
        "logs/**",
        "results/.cache/**",
    ]

    print(f"[push] uploading {ROOT} -> space {SPACE_ID}")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=SPACE_ID,
        repo_type="space",
        ignore_patterns=ignore,
        commit_message="PromptOps Arena demo",
    )
    print(f"[push] done. https://huggingface.co/spaces/{SPACE_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
