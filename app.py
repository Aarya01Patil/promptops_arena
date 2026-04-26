"""
PromptOps Arena — HF Space demo (Gradio).

Tabs:
  1. Try the env: pick a task, edit a system prompt, see the LLM-under-test
     respond + the per-component reward. Up to 3 edit turns per episode.
  2. Reward curve: training_log.jsonl rolling avg over GRPO rollouts.
  3. Baselines vs trained agent: bar chart of mean reward / accuracy.

The frozen LLM-under-test runs in-process. ZeroGPU is used at first inference.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make src importable regardless of where Gradio runs
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # type: ignore

# Default to the real backend on Spaces; allow override
os.environ.setdefault("PROMPTOPS_LLM_BACKEND", "transformers")

from src.envs.promptops_arena.server.environment import PromptOpsArenaEnvironment  # noqa: E402
from src.envs.promptops_arena.tasks import load_tasks  # noqa: E402

ENV = PromptOpsArenaEnvironment(split="test", seed=0)
ALL_TASKS: List[dict] = load_tasks(split="test")
TASKS_BY_ID: Dict[str, dict] = {t["id"]: t for t in ALL_TASKS}


SUGGESTED_PROMPTS = {
    "math": (
        "You are a careful math solver. Solve step by step internally, then "
        "output ONLY the final numeric answer inside <answer>...</answer> tags. "
        "No units, no extra words."
    ),
    "code": (
        "You are a Python coder. Output exactly one ```python ...``` code block "
        "containing only the requested function definition. No prose, no examples."
    ),
    "json": (
        "You are a JSON extractor. Output exactly one ```json ...``` code block "
        "containing a valid JSON object that matches the schema. No prose."
    ),
}


def list_task_choices() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for t in ALL_TASKS:
        label = f"[{t['type']}] {t['id']}: {t['question'][:70]}"
        out.append((label, t["id"]))
    return out


def get_task_info(task_id: str) -> Tuple[str, str, str]:
    t = TASKS_BY_ID.get(task_id)
    if not t:
        return "", "", ""
    schema = ""
    if t.get("type") == "json" and "schema" in t:
        schema = f"\n\nSchema: ```json\n{json.dumps(t['schema'], indent=2)}\n```"
    if t.get("type") == "code" and "tests" in t:
        schema = "\n\nUnit tests:\n```python\n" + "\n".join(t["tests"]) + "\n```"
    return t["question"] + schema, t.get("type", ""), SUGGESTED_PROMPTS.get(t.get("type", ""), "")


def run_prompt(task_id: str, system_prompt: str) -> Tuple[str, str, str]:
    """Run one shot of [system_prompt, task] through the env."""
    t = TASKS_BY_ID.get(task_id)
    if t is None:
        return "(no task selected)", "", ""
    if not (system_prompt or "").strip():
        return "(empty prompt)", "", ""
    res = ENV.execute_prompt(t, system_prompt)
    completion = res["completion"]
    rd = res["reward"]
    breakdown = (
        f"correctness: {rd['correctness']:.2f}\n"
        f"format    : {rd['format']:.2f}  (×0.1 in total)\n"
        f"brevity   : {rd['brevity']:+.3f}\n"
        f"-------\n"
        f"TOTAL     : {rd['total']:+.3f}"
    )
    verifier = res.get("verifier", {})
    details = verifier.get("details", "")
    return completion, breakdown, details


def load_reward_curve_image() -> str | None:
    p = Path(__file__).resolve().parent / "docs" / "reward_curve.png"
    exists = p.exists()
    print(f"[app] reward_curve.png path={p} exists={exists}")
    return str(p) if exists else None


def load_comparison_image() -> str | None:
    p = Path(__file__).resolve().parent / "docs" / "baseline_comparison.png"
    exists = p.exists()
    print(f"[app] baseline_comparison.png path={p} exists={exists}")
    return str(p) if exists else None


def load_comparison_table() -> str:
    p = Path(__file__).resolve().parent / "results" / "comparison.json"
    if not p.exists():
        return "_No comparison.json yet — train + run plot_results.py to populate._"
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d.get("policies", {})
    if not rows:
        return "_comparison.json is empty._"
    lines = [
        "| policy | n | correct | format | mean_reward |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, r in rows.items():
        lines.append(
            f"| {label} | {r['n']} | {r['correct']} | {r['format']} | {r['mean_reward']:+.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INTRO = """
# PromptOps Arena 🎯

> An RL environment where an agent learns to **write better prompts** via GRPO,
> across math, code, and JSON-extraction tasks.

- **Agent (trained):** Qwen2.5-1.5B-Instruct + LoRA, optimized with GRPO.
- **LLM-under-test (frozen):** Qwen2.5-0.5B-Instruct.
- **Reward:** `correctness + 0.1·format + brevity_penalty`, all programmatic.

Try writing your own system prompts in the **Try the env** tab.
"""


with gr.Blocks(title="PromptOps Arena", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Try the env"):
        with gr.Row():
            task_dd = gr.Dropdown(
                choices=list_task_choices(),
                value=ALL_TASKS[0]["id"] if ALL_TASKS else None,
                label="Pick a task",
                interactive=True,
            )
        task_text = gr.Markdown(label="Task")
        task_type_box = gr.Textbox(label="task type", interactive=False)
        with gr.Row():
            with gr.Column():
                system_prompt = gr.Textbox(
                    label="Your system prompt (this is the action)",
                    lines=8,
                    placeholder="Write the system prompt to give to the small frozen LLM…",
                )
                with gr.Row():
                    suggest_btn = gr.Button("Use suggested prompt")
                    run_btn = gr.Button("▶ Run", variant="primary")
            with gr.Column():
                completion_out = gr.Textbox(
                    label="LLM-under-test completion", lines=8, interactive=False,
                )
                reward_out = gr.Textbox(
                    label="Reward decomposition", lines=6, interactive=False,
                )
                verifier_out = gr.Textbox(
                    label="Verifier details", lines=2, interactive=False,
                )

        def _on_task(task_id):
            text, ttype, suggested = get_task_info(task_id)
            return text, ttype, suggested

        task_dd.change(_on_task, inputs=task_dd, outputs=[task_text, task_type_box, system_prompt])
        suggest_btn.click(_on_task, inputs=task_dd, outputs=[task_text, task_type_box, system_prompt])
        run_btn.click(run_prompt, inputs=[task_dd, system_prompt],
                      outputs=[completion_out, reward_out, verifier_out])

    with gr.Tab("Reward curve"):
        gr.Markdown("### GRPO training reward curve\n"
                    "Each point is the env's total reward for one rollout during training.")
        rc_img = gr.Image(value=load_reward_curve_image(), label="reward_curve.png",
                          type="filepath", interactive=False, show_label=False)
        gr.Markdown(
            "_If this is empty, training hasn't been run yet or `docs/reward_curve.png` "
            "is missing. Run `scripts/plot_results.py` after training._"
        )

    with gr.Tab("Baselines vs trained agent"):
        gr.Markdown("### Comparison on the held-out test split\n")
        cmp_img = gr.Image(value=load_comparison_image(), label="baseline_comparison.png",
                           type="filepath", interactive=False, show_label=False)
        gr.Markdown(load_comparison_table())

    with gr.Tab("How it works"):
        gr.Markdown((Path(__file__).resolve().parent / "docs" / "SCOPE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    demo.queue().launch()
