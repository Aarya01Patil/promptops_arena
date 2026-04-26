"""
Phase 2 smoke test: in-process exercise of reset() / step() / state.

Uses the stub LLM backend (set via env var) so this runs in <1s and proves
the env plumbing works without downloading any model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from project root: add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PROMPTOPS_LLM_BACKEND", "stub")

from src.envs.promptops_arena.server.environment import PromptOpsArenaEnvironment
from src.envs.promptops_arena.models import PromptOpsAction
from src.envs.promptops_arena import llm_under_test


GOOD_PROMPTS = {
    "math": (
        "You are a careful math solver. Read the problem, think step by step, "
        "then put ONLY the final numeric answer inside <answer>...</answer> tags. "
        "Do not include units."
    ),
    "code": (
        "You are a Python coder. Output ONLY a single ```python code block``` "
        "containing the requested function. No prose, no examples, no print statements."
    ),
    "json": (
        "You are a JSON extractor. Output ONLY a single ```json code block``` "
        "containing a valid JSON object that matches the requested schema. "
        "No prose."
    ),
}


def run(task_type: str) -> dict:
    env = PromptOpsArenaEnvironment(max_turns=3, split="train", seed=42, task_types=[task_type])
    obs = env.reset()
    print(f"\n=== {task_type.upper()} | task_id={env.state.task_id} ===")
    print(f"task: {obs.task_text}")

    action = PromptOpsAction(new_system_prompt=GOOD_PROMPTS[task_type])
    obs2 = env.step(action)
    print(f"completion: {obs2.last_completion[:120]!r}")
    print(f"reward components: {obs2.reward_components}")
    print(f"done: {obs2.done}, edit_turn: {obs2.edit_turn}, solved: {env.state.solved}")
    return {
        "task_type": task_type,
        "reward": obs2.last_reward,
        "components": obs2.reward_components,
        "solved": env.state.solved,
        "step_count": env.state.step_count,
    }


def main() -> int:
    print(f"LLM backend: {llm_under_test.backend_name()}")
    results = []
    for tt in ("math", "code", "json"):
        results.append(run(tt))

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['task_type']:5s}: reward={r['reward']:+.3f} solved={r['solved']} "
              f"step_count={r['step_count']} components={r['components']}")

    # Exit-criterion check: every type produced a structured reward dict
    ok = all(r["components"].get("total") is not None for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
