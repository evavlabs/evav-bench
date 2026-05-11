"""Audit Report + ancillary deliverable renderers.

Renders full Audit Report, Failure Cell Map, Precursor Profile, Intervention
Recommendations, and run comparisons from aggregated cell results.

STATUS: skeleton — produces structured markdown/JSON with the right shape;
deeper-narrative sections (interpretation, exemplars) are stubbed and
populated by analysts during the actual audit-report-writing phase.
"""

from __future__ import annotations
import json
from pathlib import Path

from .card import _load_results, _aggregate_card


def render_report_from_dir(output_dir: Path) -> str:
    cfg, cells = _load_results(Path(output_dir))
    card = _aggregate_card(cfg, cells)

    parts = [
        f"# {cfg.customer} — OA Deployment Safety Audit Report",
        f"\n**Engagement:** {cfg.engagement_id}",
        f"\n**Model:** {cfg.model.provider}/{cfg.model.name}",
        f"\n**Battery:** {cfg.battery_version}",
        f"\n**Cells run:** {len(cells)}",
        f"\n**Total decisions:** {sum(c.get('n_pairs', 0) * 2 for c in cells.values())}",
        "\n---\n",
        "## Section 1 — Executive Summary",
        "\n(To be drafted by audit lead based on findings below.)\n",
        "\n## Section 2 — Methodology\n",
        "Matched-pair causal identification. PRNG-seeded deterministic scenarios. "
        "Detailed methodology in `benchmark/BENCHMARK_SPEC.md`.\n",
        "\n## Section 3 — Results By Group\n",
    ]

    by_group: dict[str, list[dict]] = {}
    for cid, data in cells.items():
        g = data["cell"]["group"]
        by_group.setdefault(g, []).append(data)

    for group in sorted(by_group.keys()):
        parts.append(f"\n### Group {group}\n")
        parts.append("\n| Cell | Name | Violation rate | Masking rate | N |")
        parts.append("\n|---|---|---:|---:|---:|")
        for d in sorted(by_group[group], key=lambda x: x["cell"]["cell_id"]):
            c = d["cell"]
            parts.append(
                f"\n| `{c['cell_id']}` | {c['name']} | "
                f"{d.get('violation_rate_pct', '—')}% | "
                f"{d.get('masking_rate_pct', '—')}% | "
                f"{d.get('n_pairs', 0)} |"
            )

    parts.append("\n\n## Section 4 — Failure Cell Map\n")
    parts.append("\nSee `failure_cell_map.json` (use `oa-bench render-failure-map`).\n")
    parts.append("\n## Section 5 — Intervention Recommendations\n")
    parts.append("\nSee `intervention_recommendations.md` (use `oa-bench render-interventions`).\n")
    parts.append("\n## Section 6 — Precursor Profile\n")
    parts.append("\nSee `precursor_profile.json` (use `oa-bench render-precursor-profile`).\n")
    parts.append("\n## Section 7 — Compliance Masking Analysis\n")
    parts.append(f"\nOverall masking rate: {card['forensics']['compliance_masking_rate']}%\n")
    parts.append("\n## Section 8 — Drift Baseline\n")
    parts.append("\nThis report and all per-cell JSON files constitute the drift baseline for "
                 "future re-runs.\n")
    parts.append("\n## Appendix A — Per-Cell Raw Results\n")
    parts.append("\nSee `cell_*.json` in this directory.\n")

    return "".join(parts)


def render_failure_cell_map(output_dir: Path) -> str:
    cfg, cells = _load_results(Path(output_dir))
    mapping = {
        "schema_version": "v1.0",
        "engagement_id": cfg.engagement_id,
        "model": f"{cfg.model.provider}/{cfg.model.name}",
        "audit_date": "",
        "cells": [],
        "summary": {
            "total_cells": len(cells),
            "cells_with_violations_over_5pct": 0,
            "cells_with_violations_over_25pct": 0,
            "cells_with_violations_over_75pct": 0,
            "max_violation_cell": "",
            "max_violation_rate_pct": 0.0,
        },
    }
    max_v = 0.0
    max_cell = ""
    for cid, data in cells.items():
        v = data.get("violation_rate_pct", 0) or 0
        if v > 5:
            mapping["summary"]["cells_with_violations_over_5pct"] += 1
        if v > 25:
            mapping["summary"]["cells_with_violations_over_25pct"] += 1
        if v > 75:
            mapping["summary"]["cells_with_violations_over_75pct"] += 1
        if v > max_v:
            max_v = v
            max_cell = cid
        c = data["cell"]
        mapping["cells"].append({
            "cell_id": cid,
            "group": c["group"],
            "name": c["name"],
            "axes": {
                "pressure_type": c["pressure"],
                "documentation_tier": c["doc_tier"],
                "anchor": c["anchor"],
                "phrasing": c["phrasing"],
                "authority": c["authority"],
                "intervention": c["intervention"],
                "seed": c["seed"],
                "temperature": c["temperature"],
            },
            "violation_rate_pct": data.get("violation_rate_pct"),
            "n_pairs": data.get("n_pairs"),
            "masking_rate_pct": data.get("masking_rate_pct"),
            "masking_structural_forms": data.get("masking_forms", {}),
            "failure_modes_detected": c.get("failure_modes_detected", []),
        })
    mapping["summary"]["max_violation_cell"] = max_cell
    mapping["summary"]["max_violation_rate_pct"] = max_v
    return json.dumps(mapping, indent=2)


def render_precursor_profile(output_dir: Path) -> str:
    cfg, cells = _load_results(Path(output_dir))
    # Aggregate precursor signals across cells, weighting by N
    agg: dict[str, list[float]] = {}
    for data in cells.values():
        sigs = data.get("precursor_signals") or {}
        for sig, val in sigs.items():
            if val is not None:
                agg.setdefault(sig, []).append(val)

    signals_out = []
    for sig, vals in sorted(agg.items()):
        mean = sum(vals) / len(vals)
        signals_out.append({
            "signal_id": sig,
            "mean_across_cells": round(mean, 5),
            "n_cells": len(vals),
        })

    profile = {
        "schema_version": "v1.0",
        "engagement_id": cfg.engagement_id,
        "model": f"{cfg.model.provider}/{cfg.model.name}",
        "signals": signals_out,
        "monitor_configuration": {
            "window_size_decisions": 20,
            "min_window_decisions": 5,
            "default_sample_rate_pct": 5,
        },
    }
    return json.dumps(profile, indent=2)


def render_intervention_recommendations(output_dir: Path) -> str:
    cfg, cells = _load_results(Path(output_dir))
    # For each B-cell, find the best G-* counterpart and emit a recommendation
    b_cells = {cid: c for cid, c in cells.items() if cid.startswith(("B1.", "B2.", "B3.", "B4.", "B5."))}
    g_cells = {cid: c for cid, c in cells.items() if cid.startswith("G")}

    parts = [
        f"# Intervention Recommendations — {cfg.customer}",
        f"\n**Engagement:** {cfg.engagement_id}",
        f"\n**Model:** {cfg.model.provider}/{cfg.model.name}\n",
        "\n## Top Recommendations\n",
    ]

    for bid, bdata in sorted(b_cells.items(), key=lambda x: -(x[1].get("violation_rate_pct") or 0)):
        b_rate = bdata.get("violation_rate_pct") or 0
        if b_rate < 10:
            continue
        b_short = bid.split(".")[0]
        best_g = None
        best_delta = 0.0
        for gid, gdata in g_cells.items():
            if f".{b_short}" in gid:
                g_rate = gdata.get("violation_rate_pct") or 0
                delta = g_rate - b_rate
                if delta < best_delta:
                    best_delta = delta
                    best_g = gid
        if best_g:
            g_intervention = best_g.split("(")[-1].rstrip(")") if "(" in best_g else "—"
            parts.append(
                f"\n### {bid} — {bdata['cell']['name']}\n"
                f"- Current: {b_rate}%\n"
                f"- Recommended intervention: **{g_intervention}**\n"
                f"- Expected post-intervention: {b_rate + best_delta}%\n"
                f"- D: {best_delta}pp\n"
                f"- Evidence: cell `{best_g}`\n"
            )

    if len(parts) == 4:
        parts.append("\n_No high-violation cells; model is safe across all tested pressure surfaces._\n")
    return "".join(parts)


def render_comparison(dir_a: Path, dir_b: Path) -> str:
    cfg_a, cells_a = _load_results(Path(dir_a))
    cfg_b, cells_b = _load_results(Path(dir_b))
    parts = [
        f"# Comparison: {cfg_a.model.name} vs {cfg_b.model.name}",
        f"\nEngagements: {cfg_a.engagement_id} vs {cfg_b.engagement_id}\n",
        "\n| Cell | A rate | B rate | D pp |",
        "\n|---|---:|---:|---:|",
    ]
    all_cells = set(cells_a.keys()) | set(cells_b.keys())
    for cid in sorted(all_cells):
        ra = (cells_a.get(cid) or {}).get("violation_rate_pct")
        rb = (cells_b.get(cid) or {}).get("violation_rate_pct")
        if ra is None or rb is None:
            delta = "—"
        else:
            delta = f"{rb - ra:+.1f}"
        parts.append(f"\n| `{cid}` | {ra if ra is not None else '—'} | {rb if rb is not None else '—'} | {delta} |")
    return "".join(parts)
