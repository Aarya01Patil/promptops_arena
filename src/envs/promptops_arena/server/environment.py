"""
PromptOps Arena environment.

reset()      : sample a task; return initial observation with empty prompt
step(action) : run LLM-under-test with action.new_system_prompt + task,
               verify, compute reward, return observation
state        : full episode state with history (used for logging/replay)
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from ..models import PromptOpsAction, PromptOpsObservation, PromptOpsState
from ..tasks import load_tasks
from ..verifiers import verify
from .. import llm_under_test
from .rewards import compute_reward


class PromptOpsArenaEnvironment(Environment):
    """
    The agent's action is a full system prompt. We run the frozen LLM-under-test
    with [system=action, user=task_text], verify, and reward.

    Episode terminates when correctness == 1.0 OR edit_turn >= max_turns.
    """

    def __init__(
        self,
        max_turns: int = 3,
        split: str = "train",
        seed: Optional[int] = None,
        task_types: Optional[list[str]] = None,
    ):
        super().__init__()
        self._max_turns = max_turns
        self._split = split
        self._task_types = task_types
        self._rng = random.Random(seed)
        self._tasks = load_tasks(split=split, types=task_types)
        if not self._tasks:
            raise RuntimeError(
                f"No tasks loaded for split={split!r} types={task_types!r}"
            )
        self._state: PromptOpsState = PromptOpsState(episode_id=str(uuid.uuid4()))
        self._task: dict = {}
        self._edit_turn: int = 0

    # ---- OpenEnv API ----

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PromptOpsObservation:
        if seed is not None:
            self._rng = random.Random(seed)

        if task_id is not None:
            matches = [t for t in self._tasks if t.get("id") == task_id]
            self._task = matches[0] if matches else self._rng.choice(self._tasks)
        else:
            self._task = self._rng.choice(self._tasks)

        self._edit_turn = 0
        self._state = PromptOpsState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self._task.get("id", ""),
            task_type=self._task.get("type", ""),
            task_text=self._task.get("question", ""),
            history=[],
            best_reward=0.0,
            solved=False,
        )

        return PromptOpsObservation(
            task_text=self._state.task_text,
            task_type=self._state.task_type,
            current_prompt="",
            last_completion="",
            last_reward=0.0,
            last_correctness=0.0,
            edit_turn=0,
            max_turns=self._max_turns,
            done=False,
            reward=0.0,
            metadata={
                "task_id": self._state.task_id,
                "episode_id": self._state.episode_id,
            },
            reward_components={},
        )

    def step(
        self,
        action: PromptOpsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PromptOpsObservation:
        if not self._task:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        prompt = action.new_system_prompt or ""
        completion = llm_under_test.generate(prompt, self._state.task_text)
        verifier_result = verify(self._task, completion)
        reward_dict = compute_reward(self._task, prompt, completion, verifier_result)
        total = reward_dict["total"]

        self._edit_turn += 1
        self._state.step_count += 1
        if total > self._state.best_reward:
            self._state.best_reward = total
        if reward_dict["correctness"] >= 1.0:
            self._state.solved = True

        self._state.history.append(
            {
                "edit_turn": self._edit_turn,
                "system_prompt": prompt,
                "completion": completion,
                "reward": reward_dict,
                "verifier": verifier_result,
            }
        )

        done = self._state.solved or self._edit_turn >= self._max_turns

        return PromptOpsObservation(
            task_text=self._state.task_text,
            task_type=self._state.task_type,
            current_prompt=prompt,
            last_completion=completion,
            last_reward=total,
            last_correctness=reward_dict["correctness"],
            edit_turn=self._edit_turn,
            max_turns=self._max_turns,
            done=done,
            reward=total,
            reward_components=reward_dict,
            metadata={
                "task_id": self._state.task_id,
                "episode_id": self._state.episode_id,
                "verifier_details": verifier_result.get("details", ""),
                "solved": self._state.solved,
            },
        )

    @property
    def state(self) -> PromptOpsState:
        return self._state

    # ---- in-process helper used by training (skip HTTP) ----

    def execute_prompt(
        self,
        task: dict,
        system_prompt: str,
    ) -> dict:
        """
        Single-shot evaluation: given a task and a candidate system prompt,
        run the LLM-under-test and return reward components + completion.

        Used by the GRPO reward function during training to avoid HTTP latency.
        """
        completion = llm_under_test.generate(system_prompt, task.get("question", ""))
        verifier_result = verify(task, completion)
        reward_dict = compute_reward(task, system_prompt, completion, verifier_result)
        return {
            "reward": reward_dict,
            "completion": completion,
            "verifier": verifier_result,
        }
