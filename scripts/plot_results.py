"""
Phase 7: Build reward-curve plot + comparison artifact.

Inputs (any subset that exists):
  results/training_log.jsonl                 # per-step rewards from GRPO
  results/baseline_zero_shot_real_subset.json
  results/baseline_cot_real_subset.json
  results/baseline_zero_shot_stub.json
  results/baseline_cot_stub.json
  results/trained_agent.json                 # eval of the trained agent (Phase 6)

Outputs:
  results/comparison.json
  docs/reward_curve.png
  docs/baseline_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[plot] WARN failed to load {p}: {e}")
        return None


def load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    rows: List[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def smooth(values: List[float], window: int = 10) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / max(1, i + 1 - lo))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--docs-dir", default="docs")
    args = p.parse_args()

    res = Path(args.results_dir)
    docs = Path(args.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1. reward curve ----
    log_rows = load_jsonl(res / "training_log.jsonl")
    if log_rows:
        rewards = [r["reward"]["total"] for r in log_rows if "reward" in r]
        smoothed = smooth(rewards, window=20)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(rewards, alpha=0.25, label="raw reward")
        ax.plot(smoothed, linewidth=2, label="rolling avg (20)")
        ax.set_xlabel("training rollout #")
        ax.set_ylabel("reward")
        ax.set_title("GRPO training reward curve  ·  PromptOps Arena")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = docs / "reward_curve.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[plot] wrote {out}  ({len(rewards)} points)")
    else:
        print("[plot] no training_log.jsonl yet -> skip reward curve")

    # ---- 2. baseline comparison ----
    files = {
        "zero_shot (stub)": res / "baseline_zero_shot_stub.json",
        "cot (stub)": res / "baseline_cot_stub.json",
        "zero_shot (real LLM)": res / "baseline_zero_shot_real.json",
        "cot (real LLM)": res / "baseline_cot_real.json",
        "trained agent (real LLM)": res / "trained_agent.json",
    }
    # fall back to the smaller subset files if the wider-n versions don't exist
    fallback = {
        "zero_shot (real LLM)": res / "baseline_zero_shot_real_subset.json",
        "cot (real LLM)": res / "baseline_cot_real_subset.json",
    }
    for k, p in fallback.items():
        if not files[k].exists() and p.exists():
            files[k] = p

    rows: Dict[str, Dict[str, Any]] = {}
    for label, path in files.items():
        d = load_json(path)
        if d is None:
            continue
        ov = d.get("overall", {})
        rows[label] = {
            "n": ov.get("n", 0),
            "correct": ov.get("correct", 0),
            "format": ov.get("format", 0),
            "mean_reward": ov.get("mean_reward", 0.0),
            "by_type": d.get("by_type", {}),
            "backend": d.get("llm_backend", "unknown"),
        }

    comparison = {
        "policies": rows,
        "ranking_by_mean_reward": sorted(
            rows.items(),
            key=lambda kv: kv[1]["mean_reward"],
            reverse=True,
        ),
    }
    (res / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(f"[plot] wrote {res/'comparison.json'}")

    # ---- 3. comparison bar chart ----
    if rows:
        labels = list(rows.keys())
        means = [rows[l]["mean_reward"] for l in labels]
        accs = [rows[l]["correct"] / max(1, rows[l]["n"]) for l in labels]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].barh(labels, means, color="#4c72b0")
        axes[0].set_xlabel("mean reward")
        axes[0].set_title("Mean reward by policy")
        axes[0].grid(axis="x", alpha=0.3)
        axes[0].invert_yaxis()

        axes[1].barh(labels, accs, color="#55a868")
        axes[1].set_xlabel("fraction correct")
        axes[1].set_title("Correctness by policy")
        axes[1].set_xlim(0, 1)
        axes[1].grid(axis="x", alpha=0.3)
        axes[1].invert_yaxis()

        fig.tight_layout()
        out = docs / "baseline_comparison.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
