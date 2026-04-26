"""
JSON verifier: extract a JSON object from completion, validate against a
jsonschema, and check value equality on required fields.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from jsonschema import validate, ValidationError


_JSON_BLOCK = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(completion: str) -> tuple[Any, bool]:
    if not completion:
        return None, False
    m = _JSON_BLOCK.search(completion)
    candidate = m.group(1).strip() if m else None
    format_ok = m is not None
    if candidate is None:
        m2 = _OBJ.search(completion)
        if m2:
            candidate = m2.group(0)
    if candidate is None:
        return None, False
    try:
        return json.loads(candidate), format_ok
    except json.JSONDecodeError:
        return None, format_ok


def _strip_nones(x: Any) -> Any:
    """HuggingFace `datasets` unifies nested dict schemas across rows by
    null-padding missing keys. That turns a clean schema like
    {"properties": {"name": {...}, "age": {...}}} into one with
    {"properties": {"name": ..., "age": ..., "email": None, ...}} if other rows
    in the dataset had those keys. jsonschema rejects the Nones. Recursively
    drop them so verification is robust to that mangling.
    """
    if isinstance(x, dict):
        return {k: _strip_nones(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_strip_nones(v) for v in x if v is not None]
    return x


def verify_json(task: Dict[str, Any], completion: str) -> Dict[str, Any]:
    obj, format_ok = _extract_json(completion or "")
    if obj is None:
        return {"correctness": 0.0, "format_ok": format_ok, "details": "parse fail"}

    schema = _strip_nones(task.get("schema", {}))
    if schema:
        try:
            validate(instance=obj, schema=schema)
        except ValidationError as e:
            return {
                "correctness": 0.0,
                "format_ok": format_ok,
                "details": f"schema: {str(e.message)[:120]}",
            }
        except Exception as e:
            # Schema itself malformed (e.g., still has Nones somewhere).
            return {
                "correctness": 0.0,
                "format_ok": format_ok,
                "details": f"schema-error: {type(e).__name__}: {str(e)[:120]}",
            }

    expected = _strip_nones(task.get("expected", {}))
    if expected:
        for k, v in expected.items():
            if obj.get(k) != v:
                return {
                    "correctness": 0.0,
                    "format_ok": format_ok,
                    "details": f"mismatch {k}: got {obj.get(k)!r} expected {v!r}",
                }

    return {"correctness": 1.0, "format_ok": format_ok, "details": "ok"}
