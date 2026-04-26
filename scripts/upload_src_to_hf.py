"""Upload the project source to a HF dataset, mirrored at /code in the Job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="Dar3devil/promptops-arena-src")
    p.add_argument("--root", default=".")
    args = p.parse_args()

    root = Path(args.root).resolve()
    api = HfApi()
    create_repo(args.repo, repo_type="dataset", exist_ok=True, private=True)

    ignore_patterns = [
        "**/__pycache__/**",
        "**/.pytest_cache/**",
        ".pytest_cache/**",
        "**/.benchmarks/**",
        ".benchmarks/**",
        ".git/**",
        ".git",
        "outputs/**",
        "**/*.pyc",
        "**/*.pyo",
        ".venv/**",
        "venv/**",
        ".env",
        ".env.local",
        ".vscode/**",
        ".idea/**",
        ".cursor/**",
        ".codex/**",
        ".claude/**",
        ".agents/**",
        "AGENTS.md",
        "wandb/**",
        "*.log",
        ".DS_Store",
        "BLOG.md",
        "notebooks/**",
    ]

    print(f"[upload] uploading {root} -> dataset {args.repo}")
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo,
        repo_type="dataset",
        ignore_patterns=ignore_patterns,
        commit_message="sync source for HF Jobs training run",
    )
    print(f"[upload] done. https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    sys.exit(main() or 0)
