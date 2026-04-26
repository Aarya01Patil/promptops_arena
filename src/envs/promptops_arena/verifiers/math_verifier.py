"""
Math verifier: extract final numeric answer and exact-match against ground truth.

Format-bonus is awarded if completion contains <answer>...</answer> tags
or a \\boxed{...} expression.
"""

from __future__ import annotations

import re
from typing import Any, Dict


_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_LINE = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_number(s: str) -> str | None:
    """Pull the first number out of s, return its canonical string form."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = _NUMBER.search(s)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if v.is_integer():
        return str(int(v))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _extract(completion: str) -> tuple[str | None, bool]:
    """Return (extracted_answer, format_ok)."""
    if not completion:
        return None, False

    m = _ANSWER_TAG.search(completion)
    if m:
        return _normalize_number(m.group(1)), True

    m = _BOXED.search(completion)
    if m:
        return _normalize_number(m.group(1)), True

    m = _FINAL_LINE.search(completion)
    if m:
        return _normalize_number(m.group(1)), False

    nums = _NUMBER.findall(completion)
    if nums:
        return _normalize_number(nums[-1]), False

    return None, False


def verify_math(task: Dict[str, Any], completion: str) -> Dict[str, Any]:
    expected = _normalize_number(str(task.get("answer", "")))
    extracted, format_ok = _extract(completion or "")

    if expected is None:
        return {"correctness": 0.0, "format_ok": format_ok, "details": "bad ground truth"}

    correct = extracted is not None and extracted == expected
    return {
        "correctness": 1.0 if correct else 0.0,
        "format_ok": format_ok,
        "details": f"expected={expected} extracted={extracted}",
    }
