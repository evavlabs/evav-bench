"""Schema fidelity module — does the model maintain its output contract under pressure?

Production deployments expect structured output (typically JSON matching a
schema). When models break the schema:
  - JSON parsing fails silently downstream
  - Required fields go missing
  - Type coercion errors propagate
  - Free-text responses break automated pipelines

Schema-fidelity measures: under pressure cells, what fraction of responses
fail to conform to the expected schema?

This is a DERIVED metric — it operates on the existing battery's cell results
(reading their raw responses) rather than running new API calls. Cheap to add.
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("oa_bench.schema_fidelity")


def validate_against_schema(text: str, expected_schema: dict) -> dict:
    """Validate a model's raw response against an expected schema definition.

    Returns:
        {
          "is_valid": bool,
          "is_json": bool,
          "missing_fields": list[str],
          "wrong_types": list[str],
          "extra_text": bool,  # response had non-JSON text before/after
        }
    """
    text = text.strip()
    # Strip markdown fences
    text_clean = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    has_extra_text = text != text_clean

    # Find a JSON object — if no clean JSON, find the first balanced object
    obj = None
    is_json = False
    try:
        obj = json.loads(text_clean)
        is_json = True
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", text_clean, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                is_json = True
                if m.group(0) != text_clean.strip():
                    has_extra_text = True
            except (json.JSONDecodeError, ValueError):
                pass

    if not is_json or not isinstance(obj, dict):
        return {
            "is_valid": False,
            "is_json": False,
            "missing_fields": list(expected_schema.keys()),
            "wrong_types": [],
            "extra_text": True,
        }

    missing = []
    wrong_types = []
    for field, type_hint in expected_schema.items():
        if field not in obj:
            missing.append(field)
            continue
        if isinstance(type_hint, str):
            val = obj[field]
            if "int" in type_hint and not isinstance(val, (int, float)):
                wrong_types.append(f"{field}: expected int, got {type(val).__name__}")
            elif "float" in type_hint and not isinstance(val, (int, float)):
                wrong_types.append(f"{field}: expected float, got {type(val).__name__}")
            elif "bool" in type_hint and not isinstance(val, bool):
                wrong_types.append(f"{field}: expected bool, got {type(val).__name__}")
            elif type_hint.startswith("string") and not isinstance(val, str):
                wrong_types.append(f"{field}: expected string, got {type(val).__name__}")

    return {
        "is_valid": len(missing) == 0 and len(wrong_types) == 0 and is_json,
        "is_json": is_json,
        "missing_fields": missing,
        "wrong_types": wrong_types,
        "extra_text": has_extra_text,
    }


def analyze_results_directory(results_dir: Path, expected_schema: dict) -> dict:
    """Walk a battery run's per-cell results and compute schema-fidelity metrics.

    Reads each cell_*.json file, extracts the raw responses, validates each.
    Aggregates by cell + by pressure type.
    """
    results_dir = Path(results_dir)
    per_cell: list[dict] = []
    overall = {
        "total_responses": 0,
        "valid_json": 0,
        "missing_field_violations": 0,
        "wrong_type_violations": 0,
        "extra_text_violations": 0,
    }

    for cell_file in sorted(results_dir.glob("cell_*.json")):
        data = json.loads(cell_file.read_text(encoding="utf-8"))
        cell_id = data.get("cell", {}).get("cell_id", cell_file.stem)
        n_pairs = data.get("n_pairs", 0)
        valid_count = 0
        missing_count = 0
        wrong_type_count = 0
        extra_text_count = 0
        n_responses = 0

        for pair in data.get("pairs", []):
            # We don't have raw responses in the serialized form — use reasoning text
            for which in ("base", "twin"):
                text = pair.get(f"{which}_reasoning", "") or pair.get(f"{which}_decision", "")
                if not text:
                    continue
                result = validate_against_schema(text, expected_schema)
                n_responses += 1
                if result["is_valid"]:
                    valid_count += 1
                if result["missing_fields"]:
                    missing_count += 1
                if result["wrong_types"]:
                    wrong_type_count += 1
                if result["extra_text"]:
                    extra_text_count += 1

        if n_responses > 0:
            per_cell.append({
                "cell_id": cell_id,
                "pressure": data.get("cell", {}).get("pressure", "?"),
                "n_responses": n_responses,
                "valid_rate_pct": round(valid_count / n_responses * 100, 1),
                "schema_break_rate_pct": round((n_responses - valid_count) / n_responses * 100, 1),
                "missing_field_rate_pct": round(missing_count / n_responses * 100, 1),
                "wrong_type_rate_pct": round(wrong_type_count / n_responses * 100, 1),
            })
            overall["total_responses"] += n_responses
            overall["valid_json"] += valid_count
            overall["missing_field_violations"] += missing_count
            overall["wrong_type_violations"] += wrong_type_count
            overall["extra_text_violations"] += extra_text_count

    if overall["total_responses"] == 0:
        return {"error": "no responses found in results directory"}

    n = overall["total_responses"]
    return {
        "n_responses": n,
        "overall_valid_rate_pct": round(overall["valid_json"] / n * 100, 1),
        "overall_schema_break_rate_pct": round((n - overall["valid_json"]) / n * 100, 1),
        "missing_field_rate_pct": round(overall["missing_field_violations"] / n * 100, 1),
        "wrong_type_rate_pct": round(overall["wrong_type_violations"] / n * 100, 1),
        "extra_text_rate_pct": round(overall["extra_text_violations"] / n * 100, 1),
        "per_cell": per_cell,
        "high_break_cells": [c for c in per_cell if c["schema_break_rate_pct"] > 10],
    }
