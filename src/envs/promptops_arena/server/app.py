"""
FastAPI server for PromptOps Arena env.

Run:
    uvicorn src.envs.promptops_arena.server.app:app --host 0.0.0.0 --port 8000

Or via:
    python -m src.envs.promptops_arena.server.app
"""

from __future__ import annotations

import os

from openenv.core.env_server import create_app

# Try in-repo first (when running scripts from project root); fall back
# to fully-qualified package import (when installed).
try:
    from src.envs.promptops_arena.server.environment import PromptOpsArenaEnvironment
    from src.envs.promptops_arena.models import PromptOpsAction, PromptOpsObservation
except ImportError:  # pragma: no cover
    from envs.promptops_arena.server.environment import PromptOpsArenaEnvironment
    from envs.promptops_arena.models import PromptOpsAction, PromptOpsObservation


max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))

app = create_app(
    PromptOpsArenaEnvironment,
    PromptOpsAction,
    PromptOpsObservation,
    env_name="promptops_arena",
    max_concurrent_envs=max_concurrent,
)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
