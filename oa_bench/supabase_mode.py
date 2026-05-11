"""Supabase mode — bridge between oa-bench and the existing EVAV_Engine.

Uploads a local battery run's config + per-cell results to Supabase so the
Engine + frontend + Tier 2 monitor can consume them. Reuses the data model
defined in `EVAV_Engine/engine/config.py`.

Only loaded when the user passes --supabase to `oa-bench run` or invokes
`oa-bench supabase-upload`. Requires the `[supabase]` extras.
"""

from __future__ import annotations
import json
import logging
import os
import uuid
from pathlib import Path

log = logging.getLogger("oa_bench.supabase")


def _client():
    """Return a Supabase client. Reads SUPABASE_URL / SUPABASE_KEY from env."""
    try:
        from supabase import create_client
    except ImportError as e:
        raise ImportError(
            "supabase SDK not installed. Run: pip install -e \".[supabase]\""
        ) from e
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY required for Supabase mode.")
    return create_client(url, key)


def upload_run(output_dir: Path) -> dict:
    """Upload a completed battery run to Supabase.

    Creates an `evaluations` row + per-cell test results + the matched-pair
    decisions. Returns the eval_id and a summary.
    """
    output_dir = Path(output_dir)
    cfg_path = output_dir / "battery.config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"battery.config.json missing in {output_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cells = sorted(output_dir.glob("cell_*.json"))

    sb = _client()
    eval_id = str(uuid.uuid4())

    # 1. Insert evaluation row
    sb.table("evaluations").insert({
        "id": eval_id,
        "name": f"oa-bench: {cfg['engagement_id']}",
        "domain": cfg["domain"],
        "model_provider": cfg["model"]["provider"],
        "model_name": cfg["model"]["name"],
        "temperature": cfg["model"].get("temperature", 0.3),
        "max_tokens": cfg["model"].get("max_tokens", 1024),
        "seed": cfg["scope"]["seeds"][0],
        "n_pairs": cfg["scope"]["n_per_cell"],
        "status": "completed",
        "violation_rule": cfg.get("scoring", {}),
        "test_types": ["matched_pair"],
    }).execute()
    log.info("created evaluation row id=%s", eval_id)

    # 2. Insert per-cell test results
    total_cells = 0
    total_pairs = 0
    total_violations = 0
    for cell_file in cells:
        data = json.loads(cell_file.read_text(encoding="utf-8"))
        cell = data["cell"]
        sb.table("test_results").insert({
            "evaluation_id": eval_id,
            "test_type": "matched_pair",
            "cell_id": cell["cell_id"],
            "cell_group": cell["group"],
            "cell_name": cell["name"],
            "violation_rate_pct": data["violation_rate_pct"],
            "masking_rate_pct": data["masking_rate_pct"],
            "masking_forms": data.get("masking_forms", {}),
            "precursor_signals": data.get("precursor_signals", {}),
            "n_pairs": data["n_pairs"],
            "elapsed_seconds": data["elapsed_seconds"],
            "errors_count": len(data.get("errors") or []),
        }).execute()
        total_cells += 1
        total_pairs += data["n_pairs"]
        total_violations += int(data["n_pairs"] * data["violation_rate_pct"] / 100)
    log.info("inserted %d test_results rows", total_cells)

    # 3. Update evaluation summary
    sb.table("evaluations").update({
        "total_decisions": total_pairs * 2,
        "completed_decisions": total_pairs * 2,
        "violations_found": total_violations,
        "results_summary": json.dumps({
            "engagement_id": cfg["engagement_id"],
            "n_cells": total_cells,
            "n_pairs": total_pairs,
            "n_violations": total_violations,
        }),
    }).eq("id", eval_id).execute()

    return {
        "eval_id": eval_id,
        "n_cells": total_cells,
        "n_pairs": total_pairs,
        "n_violations": total_violations,
    }
