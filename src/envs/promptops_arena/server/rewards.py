"""
Reward function: decomposed, bounded, gated by correctness.

total = correctness + 0.1 * format_bonus + brevity_penalty

Where:
  correctness  in {0.0, 1.0}            -- programmatic verifier
  format_bonus in {0.0, 1.0}            -- multiplied by 0.1 in total
  brevity_penalty in [-0.1, 0.0]        -- only if prompt > 800 chars
                                           = -0.05 * max(0, (len-800))/200
                                           clipped to -0.1

If correctness == 0, the format_bonus is still added (small) but the agent
cannot exceed 0.1 reward without correctness.
"""

from __future__ import annotations

from typing import Any, Dict


def compute_reward(
    task: Dict[str, Any],
    system_prompt: str,
    completion: str,
    verifier_result: Dict[str, Any],
) -> Dict[str, float]:
    correctness = float(verifier_result.get("correctness", 0.0))
    format_ok = bool(verifier_result.get("format_ok", False))
    format_bonus = 1.0 if format_ok else 0.0

    p_len = len(system_prompt or "")
    excess = max(0, p_len - 800)
    brevity_penalty = -0.05 * (excess / 200.0)
    if brevity_penalty < -0.1:
        brevity_penalty = -0.1

    total = correctness + 0.1 * format_bonus + brevity_penalty

    return {
        "correctness": correctness,
        "format": format_bonus,
        "brevity": brevity_penalty,
        "total": total,
    }
