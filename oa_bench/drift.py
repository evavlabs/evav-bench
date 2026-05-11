"""Drift baseline + diff commands.

The drift baseline is a content-addressed snapshot of a battery run. It is the
regression standard used to detect behavioral drift when the customer updates
their model, changes their system prompt, or runs the same configuration on
a different date.

Operations:
  - `save`  — snapshot a completed run as a named baseline
  - `diff`  — compare a new run against a baseline; report per-cell deltas + significance

The baseline file is JSON, content-addressed by sha256, versioned per engagement.
"""

from __future__ import annotations
import hashlib
import json
import statistics
from datetime import date
from pathlib import Path

from .battery import BatteryConfig


def save_baseline(output_dir: Path, baseline_name: str | None = None) -> dict:
    """Snapshot a completed battery run as a drift baseline.

    Writes `<output_dir>/drift_baseline.json` containing per-cell summary,
    a content hash, and metadata.
    """
    cfg_path = output_dir / "battery.config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"battery.config.json missing in {output_dir}")
    cfg = BatteryConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))

    cells: dict[str, dict] = {}
    for cell_file in sorted(output_dir.glob("cell_*.json")):
        data = json.loads(cell_file.read_text(encoding="utf-8"))
        cid = data["cell"]["cell_id"]
        cells[cid] = {
            "violation_rate_pct": data.get("violation_rate_pct"),
            "masking_rate_pct": data.get("masking_rate_pct"),
            "n_pairs": data.get("n_pairs"),
            "group": data["cell"]["group"],
            "name": data["cell"]["name"],
            "pressure": data["cell"]["pressure"],
            "doc_tier": data["cell"]["doc_tier"],
            "anchor": data["cell"]["anchor"],
            "intervention": data["cell"]["intervention"],
            "seed": data["cell"]["seed"],
            "temperature": data["cell"]["temperature"],
            "errors_count": len(data.get("errors") or []),
        }

    blob = json.dumps(cells, sort_keys=True).encode("utf-8")
    content_hash = hashlib.sha256(blob).hexdigest()[:16]
    name = baseline_name or f"{cfg.engagement_id}-{date.today().isoformat()}"

    baseline = {
        "schema_version": "v1.0",
        "name": name,
        "engagement_id": cfg.engagement_id,
        "customer": cfg.customer,
        "domain": cfg.domain,
        "model_provider": cfg.model.provider,
        "model_name": cfg.model.name,
        "model_version": cfg.model.version or "",
        "battery_version": cfg.battery_version,
        "saved_date": str(date.today()),
        "content_hash": content_hash,
        "n_cells": len(cells),
        "cells": cells,
    }

    out_path = output_dir / "drift_baseline.json"
    out_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def diff_against_baseline(baseline_path: Path, new_run_dir: Path,
                          significance_pp: float = 5.0) -> dict:
    """Compare a new battery run against a saved baseline.

    Reports per-cell deltas, flags cells with violation-rate movement >= significance_pp,
    and summarizes overall drift.
    """
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    new_baseline = save_baseline(new_run_dir, baseline_name="__diff_target__")

    cells_a: dict[str, dict] = baseline["cells"]
    cells_b: dict[str, dict] = new_baseline["cells"]
    all_keys = sorted(set(cells_a) | set(cells_b))

    cell_diffs = []
    drifted_cells = []
    for k in all_keys:
        a = cells_a.get(k) or {}
        b = cells_b.get(k) or {}
        rate_a = a.get("violation_rate_pct")
        rate_b = b.get("violation_rate_pct")
        mask_a = a.get("masking_rate_pct")
        mask_b = b.get("masking_rate_pct")
        if rate_a is None or rate_b is None:
            continue
        delta = rate_b - rate_a
        mask_delta = (mask_b or 0) - (mask_a or 0)
        is_drifted = abs(delta) >= significance_pp
        if is_drifted:
            drifted_cells.append({
                "cell_id": k,
                "rate_baseline": rate_a,
                "rate_new": rate_b,
                "delta_pp": round(delta, 2),
                "masking_delta_pp": round(mask_delta, 2),
            })
        cell_diffs.append({
            "cell_id": k,
            "rate_baseline": rate_a,
            "rate_new": rate_b,
            "delta_pp": round(delta, 2),
            "masking_delta_pp": round(mask_delta, 2),
            "is_drifted": is_drifted,
        })

    deltas = [d["delta_pp"] for d in cell_diffs]
    summary = {
        "schema_version": "v1.0",
        "baseline_name": baseline["name"],
        "baseline_date": baseline["saved_date"],
        "baseline_hash": baseline["content_hash"],
        "new_run_date": new_baseline["saved_date"],
        "new_run_hash": new_baseline["content_hash"],
        "significance_threshold_pp": significance_pp,
        "n_cells_compared": len(cell_diffs),
        "n_cells_drifted": len(drifted_cells),
        "max_drift_pp": max(deltas, key=abs) if deltas else 0,
        "mean_drift_pp": round(statistics.mean(deltas), 2) if deltas else 0,
        "median_drift_pp": round(statistics.median(deltas), 2) if deltas else 0,
        "stddev_drift_pp": round(statistics.pstdev(deltas), 2) if len(deltas) >= 2 else 0,
        "drifted_cells": drifted_cells,
        "all_cell_diffs": cell_diffs,
    }
    return summary
