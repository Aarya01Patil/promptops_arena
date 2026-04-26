"""
HTTP/WebSocket client for the PromptOps Arena env.

Used by the demo Space, manual exploration, and any out-of-process consumer.
GRPO training uses the in-process `PromptOpsArenaEnvironment` directly.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import PromptOpsAction, PromptOpsObservation, PromptOpsState


class PromptOpsArenaEnv(EnvClient[PromptOpsAction, PromptOpsObservation, PromptOpsState]):
    """Client; subclass of openenv.core.env_client.EnvClient."""

    def _step_payload(self, action: PromptOpsAction) -> Dict[str, Any]:
        return {"new_system_prompt": action.new_system_prompt}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PromptOpsObservation]:
        obs_data = payload.get("observation", payload)
        observation = PromptOpsObservation(
            task_text=obs_data.get("task_text", ""),
            task_type=obs_data.get("task_type", ""),
            current_prompt=obs_data.get("current_prompt", ""),
            last_completion=obs_data.get("last_completion", ""),
            last_reward=obs_data.get("last_reward", 0.0),
            last_correctness=obs_data.get("last_correctness", 0.0),
            edit_turn=obs_data.get("edit_turn", 0),
            max_turns=obs_data.get("max_turns", 3),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            reward_components=obs_data.get("reward_components", {}),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=observation.reward or 0.0,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PromptOpsState:
        return PromptOpsState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_type=payload.get("task_type", ""),
            task_text=payload.get("task_text", ""),
            history=payload.get("history", []),
            best_reward=payload.get("best_reward", 0.0),
            solved=payload.get("solved", False),
        )
