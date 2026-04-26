"""
Code verifier: extract a python code block from completion, run it in a
subprocess with the task's test cases appended, with a hard timeout.

NEVER use in-process exec(). Subprocess only.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(completion: str) -> tuple[str | None, bool]:
    if not completion:
        return None, False
    m = _CODE_BLOCK.search(completion)
    if m:
        return m.group(1).strip(), True
    if "def " in completion:
        return completion.strip(), False
    return None, False


def verify_code(task: Dict[str, Any], completion: str) -> Dict[str, Any]:
    code, format_ok = _extract_code(completion or "")
    if code is None:
        return {"correctness": 0.0, "format_ok": False, "details": "no code"}

    tests = task.get("tests", [])
    if not tests:
        return {"correctness": 0.0, "format_ok": format_ok, "details": "no tests in task"}

    test_block = "\n".join(tests)
    program = f"{code}\n\n# --- tests ---\n{test_block}\nprint('__OK__')\n"

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return {"correctness": 0.0, "format_ok": format_ok, "details": "timeout"}
        except Exception as e:
            return {"correctness": 0.0, "format_ok": format_ok, "details": f"runner err: {e}"}

    ok = proc.returncode == 0 and "__OK__" in (proc.stdout or "")
    detail = (proc.stderr or proc.stdout or "")[:200].replace("\n", " ")
    return {
        "correctness": 1.0 if ok else 0.0,
        "format_ok": format_ok,
        "details": detail,
    }
