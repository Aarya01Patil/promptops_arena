"""
Type-safe data contracts for the PromptOps Arena environment.

Action: agent emits a full new system prompt to give to the LLM-under-test.
Observation: task text, last completion, last reward, edit-turn counter.
State: full episode history for logging / demo replay.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server import Action, Observation, State  # type: ignore
except Exception:  # pragma: no cover - shim for envs where openenv-core layout differs
    from pydantic import BaseModel

    class Action(BaseModel):  # type: ignore[no-redef]
        pass

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0

    class State(BaseModel):  # type: ignore[no-redef]
        pass

from pydantic import Field


class PromptOpsAction(Action):
    """Agent's action: write/replace the system prompt for the LLM-under-test."""

    new_system_prompt: str = Field(
        ...,
        description="Full system prompt the agent wants to give the frozen LLM-under-test",
    )


class PromptOpsObservation(Observation):
    """What the agent sees after each step."""

    task_text: str = ""
    task_type: str = ""  # "math" | "code" | "json"
    current_prompt: str = ""
    last_completion: str = ""
    last_reward: float = 0.0
    last_correctness: float = 0.0
    edit_turn: int = 0
    max_turns: int = 3
    reward_components: Dict[str, float] = Field(default_factory=dict)


class PromptOpsState(State):
    """Episode state for logging and replay."""

    task_id: str = ""
    task_type: str = ""
    task_text: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    best_reward: float = 0.0
    solved: bool = False
