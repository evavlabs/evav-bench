"""Render the visual HTML Evaluation Card from a battery run output.

Usage:
    python render_visual_card.py <battery_output_dir> --out card.html
    python render_visual_card.py --leaderboard-entry <leaderboard.json#claude-sonnet-4> --out card.html

The output is a fully self-contained HTML file with embedded CSS — can be
opened in a browser, embedded in an iframe, served as a download, or piped
through pandoc/wkhtmltopdf to produce a PDF/PNG.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path


TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "deployment_card.html"


def _rate_class(v):
    """CSS class for bar severity. Restrained palette: neutral fill by default,
    accent orange for 35%+, deeper accent for 75%+."""
    if v is None:
        return ""
    if v < 35:
        return ""
    if v < 75:
        return "high"
    return "severe"


def _verdict_class(recommendation):
    """Map readiness recommendation to CSS class. Severe verdicts get accent emphasis."""
    r = (recommendation or "").lower()
    if "not recommended" in r:
        return "severe"
    if "guardrails" in r:
        return "severe"
    return ""


def _fmt_rate(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"


def _fmt_delta(v):
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:+.1f}"


def _fmt_bool(v, true_label="yes", false_label="no"):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return true_label if v else false_label
    return str(v)


BAND_LABELS = {
    "exceptional": "Exceptional",
    "strong": "Strong",
    "solid": "Solid",
    "notable_failure_modes": "Notable failure modes",
    "significant_concerns": "Significant concerns",
    "catastrophic": "Catastrophic",
}

BAND_DESCRIPTIONS = {
    "exceptional": "No currently tested frontier model reaches this band.",
    "strong": "Top of the current state of the art. Some failure modes present; configuration-specific testing recommended before deployment.",
    "solid": "Typical for capable frontier models. Notable areas may need attention depending on deployment context.",
    "notable_failure_modes": "Several areas with rates that warrant investigation before deployment.",
    "significant_concerns": "Significant concerns across multiple modules. Substantial pre-deployment work recommended.",
    "catastrophic": "At least one module shows near-total failure. Unsuitable for regulated deployment without major remediation.",
}

MODULE_LABELS = {
    "pressure": "Pressure",
    "compliance_masking": "Compliance masking",
    "hallucination": "Hallucination",
    "adversarial": "Adversarial",
    "refusal": "Refusal",
    "privacy": "Privacy",
    "schema_fidelity": "Schema fidelity",
    "bias": "Bias",
    "tool_safety": "Tool safety",
    "reasoning_trace": "Reasoning trace",
}


def render_card(data: dict, cli_version: str = "1.0.0") -> str:
    """Render the visual card from a normalized data dict.

    Data dict can come from either:
      - oa-bench card.py output (Evaluation Card JSON)
      - leaderboard.json entry
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Normalize input — accept either schema
    if "provenance" in data:
        # Card-schema input
        p = data["provenance"]
        rates = data.get("rates", {})
        iv = data.get("interventions", {})
        rob = data.get("robustness", {})
        forn = data.get("forensics", {})
        rd = data.get("readiness", {})
        transport = data.get("transport", {})
        model_name = p.get("model_name", "")
        provider = p.get("model_provider", "")
        domain = data.get("domain", "—")
        test_date = p.get("test_date", "")
        seed = p.get("seed", "")
        temperature = p.get("temperature", "")
        n_per_cell = p.get("n_per_cell", "")
        total_decisions = p.get("total_decisions", "")
        engagement_id = p.get("engagement_id", "")
        battery_version = data.get("battery_version", "v1.0")
        prohibit_d = iv.get("prohibit", {}).get("delta_pp")
        bind_d = iv.get("bind", {}).get("delta_pp")
        remind_d = iv.get("remind", {}).get("delta_pp")
        seed_swing = rob.get("seed_swing_pp")
        masking = forn.get("compliance_masking_rate")
        scf = forn.get("self_correction_failure")
        mea = forn.get("motivated_evidence_application")
        lc = forn.get("latent_capture")
        max_failure = rd.get("max_failure_rate")
        best_intervention = rd.get("best_intervention")
        best_delta = rd.get("best_intervention_delta_pp")
        residual = rd.get("residual_failure_rate")
        recommended = rd.get("recommended_use", "")
        within = transport.get("within_domain", "—")
        cross = transport.get("cross_domain", "—")
        findings = []
    elif "rank" in data and ("rates" in data or "domain_results" in data):
        # Leaderboard-entry input
        # Support both old (flat `rates`) and new (per-domain `domain_results`) schemas
        if "domain_results" not in data:
            pass
        else:
            hc = data["domain_results"].get("healthcare", {})
            data.setdefault("rates", {})
            data["rates"].setdefault("a1_baseline_mixed", hc.get("baseline_pct"))
            data["rates"].setdefault("a5_baseline_no_anchor", hc.get("baseline_no_anchor_pct"))
            data["rates"].setdefault("a7_equalized_control", hc.get("equalized_control_pct"))
            data["rates"].setdefault("b1_reward", hc.get("reward_pct"))
            data["rates"].setdefault("b2_threat", hc.get("threat_pct"))
            data["rates"].setdefault("b3_optimization", hc.get("optimization_pct"))
            data["rates"].setdefault("b4_social", hc.get("sycophancy_pct"))
            data["rates"].setdefault("b5_authority", hc.get("authority_pct"))
            data["rates"].setdefault("b6_anchor", hc.get("anchor_pct"))
            data["rates"].setdefault("b7_frame", hc.get("frame_pct"))
            data["rates"].setdefault("c1_combined", hc.get("combined_pct"))
            data["rates"].setdefault("d1_doc_strong", hc.get("doc_strong_pct"))
            data["rates"].setdefault("d2_doc_moderate", hc.get("doc_moderate_pct"))
            data["rates"].setdefault("d3_doc_qualified", hc.get("doc_qualified_pct"))
            data.setdefault("interventions", {})
            bi = hc.get("best_intervention")
            if isinstance(bi, dict):
                if bi.get("name") == "PROHIBIT":
                    data["interventions"].setdefault("prohibit", {"delta_pp": bi.get("delta_pp")})
                if bi.get("name") == "BIND":
                    data["interventions"].setdefault("bind", {"delta_pp": bi.get("delta_pp")})
        model_name = data.get("model_name", "")
        provider = data.get("provider", "")
        rates = data.get("rates", {})
        iv = data.get("interventions", {})
        forn = data.get("forensics", {})
        transport = data.get("transport", {})
        domain = "Healthcare (Medicare prior auth)"
        test_date = data.get("tested_date", "")
        seed = 42
        temperature = 0.3
        n_per_cell = data.get("n_per_cell", "")
        total_decisions = "—"
        engagement_id = data.get("model_id", "")
        battery_version = "v1.0"
        prohibit_d = iv.get("prohibit", {}).get("delta_pp") if iv.get("prohibit") else None
        bind_d = iv.get("bind", {}).get("delta_pp") if iv.get("bind") else None
        remind_d = None
        seed_swing = forn.get("cross_seed_swing_pp")
        masking = forn.get("compliance_masking_rate_pct")
        scf = forn.get("self_correction_failure")
        mea = forn.get("motivated_evidence_application")
        lc = forn.get("latent_capture")
        max_failure = data.get("headline_max_failure_pct")
        best_intervention = "PROHIBIT" if prohibit_d and prohibit_d < (bind_d or 0) else "BIND"
        # Compute best delta
        deltas = [d for d in [prohibit_d, bind_d] if isinstance(d, (int, float))]
        best_delta = min(deltas) if deltas else None
        residual = (max_failure + best_delta) if (max_failure is not None and best_delta is not None) else None
        recommended = data.get("recommended_use", "")
        within = transport.get("within_domain", "—")
        cross = transport.get("cross_domain", "—")
        findings = data.get("key_findings", [])
    else:
        raise ValueError("Unrecognized data schema — expected card JSON or leaderboard entry")

    a1 = rates.get("a1_baseline_mixed")
    a5 = rates.get("a5_baseline_no_anchor")
    a7 = rates.get("a7_equalized_control")
    d1 = rates.get("d1_doc_strong")
    d2 = rates.get("d2_doc_moderate")
    d3 = rates.get("d3_doc_qualified")
    doc_cliff = (d3 - d1) if (isinstance(d1, (int, float)) and isinstance(d3, (int, float))) else None

    b1 = rates.get("b1_reward")
    b2 = rates.get("b2_threat")
    b3 = rates.get("b3_optimization")
    b4 = rates.get("b4_social")
    b5 = rates.get("b5_authority")
    b6 = rates.get("b6_anchor")
    b7 = rates.get("b7_frame")
    c1 = rates.get("c1_combined")

    findings_html = "".join(f"<li>{escape(f)}</li>" for f in findings) or "<li>No notable findings.</li>"

    evav_score = data.get("evav_score")
    evav_band = data.get("evav_band", "")
    weakest_module = data.get("weakest_module", "")
    weakest_module_score = data.get("weakest_module_score")
    evav_band_label = BAND_LABELS.get(evav_band, evav_band)
    band_description = BAND_DESCRIPTIONS.get(evav_band, "")
    weakest_module_label = MODULE_LABELS.get(weakest_module, weakest_module or "n/a")

    substitutions = {
        "model_name": escape(model_name),
        "model_provider": escape(provider),
        "domain": escape(domain),
        "test_date": escape(str(test_date)),
        "seed": str(seed),
        "temperature": str(temperature),
        "n_per_cell": str(n_per_cell),
        "total_decisions": str(total_decisions),
        "battery_version": str(battery_version).replace("v", ""),
        "engagement_id": escape(str(engagement_id)),
        "cli_version": cli_version,
        "a1_baseline_mixed": _fmt_rate(a1),
        "a5_baseline_no_anchor": _fmt_rate(a5),
        "a7_equalized_control": _fmt_rate(a7),
        "a7_flag": "" if (a7 is None or a7 < 5) else '<span class="pill warning">check</span>',
        "d1_doc_strong": _fmt_rate(d1), "d1_class": _rate_class(d1),
        "d2_doc_moderate": _fmt_rate(d2), "d2_class": _rate_class(d2),
        "d3_doc_qualified": _fmt_rate(d3), "d3_class": _rate_class(d3),
        "doc_cliff_pp": _fmt_delta(doc_cliff),
        "b1_reward": _fmt_rate(b1), "b1_class": _rate_class(b1),
        "b2_threat": _fmt_rate(b2), "b2_class": _rate_class(b2),
        "b3_optimization": _fmt_rate(b3), "b3_class": _rate_class(b3),
        "b4_social": _fmt_rate(b4), "b4_class": _rate_class(b4),
        "b5_authority": _fmt_rate(b5), "b5_class": _rate_class(b5),
        "b6_anchor": _fmt_rate(b6), "b6_class": _rate_class(b6),
        "b7_frame": _fmt_rate(b7), "b7_class": _rate_class(b7),
        "c1_combined": _fmt_rate(c1), "c1_class": _rate_class(c1),
        "prohibit_delta": _fmt_delta(prohibit_d),
        "bind_delta": _fmt_delta(bind_d),
        "remind_delta": _fmt_delta(remind_d),
        "within_domain_transport": escape(str(within)),
        "cross_domain_transport": escape(str(cross)),
        "masking_rate": _fmt_rate(masking),
        "scf_label": _fmt_bool(scf, "observed", "not observed"),
        "mea_label": _fmt_bool(mea, "observed", "not observed"),
        "lc_label": _fmt_bool(lc, "observed", "not observed"),
        "seed_swing_pp": _fmt_rate(seed_swing) if seed_swing is None else f"{seed_swing}",
        "max_failure_rate": _fmt_rate(max_failure),
        "best_intervention": escape(str(best_intervention or "n/a")),
        "best_intervention_delta": _fmt_delta(best_delta),
        "residual_failure": _fmt_rate(residual),
        "recommended_use": escape(recommended),
        "verdict_class": _verdict_class(recommended),
        "findings_list": findings_html,
        "evav_score": _fmt_rate(evav_score) if evav_score is not None else "—",
        "evav_band": evav_band,
        "evav_band_label": escape(evav_band_label),
        "band_description": escape(band_description),
        "weakest_module_label": escape(weakest_module_label),
        "weakest_module_score": _fmt_rate(weakest_module_score) if weakest_module_score is not None else "—",
    }

    out = template
    for key, val in substitutions.items():
        out = out.replace("{{ " + key + " }}", str(val))
    return out


def escape(s: str) -> str:
    """HTML-escape a string."""
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def main():
    parser = argparse.ArgumentParser(description="Render visual Evaluation Card.")
    parser.add_argument("source", nargs="?",
                        help="battery output dir OR path to card.json OR leaderboard.json")
    parser.add_argument("--leaderboard-entry",
                        help="Render a specific leaderboard entry. Format: <path>#<model_id>")
    parser.add_argument("--out", default="card.html", help="Output HTML file")
    args = parser.parse_args()

    if args.leaderboard_entry:
        path, model_id = args.leaderboard_entry.split("#")
        lb = json.loads(Path(path).read_text(encoding="utf-8"))
        match = next((c for c in lb["cards"] if c.get("model_id") == model_id), None)
        if not match:
            print(f"No model_id={model_id} in {path}", file=sys.stderr)
            sys.exit(2)
        data = match
    elif args.source:
        src = Path(args.source)
        if src.is_dir():
            # battery output dir — invoke card.py aggregator
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cli"))
            from oa_bench.card import _load_results, _aggregate_card
            cfg, cells = _load_results(src)
            data = _aggregate_card(cfg, cells)
        elif src.is_file():
            data = json.loads(src.read_text(encoding="utf-8"))
        else:
            print(f"Source not found: {src}", file=sys.stderr)
            sys.exit(2)
    else:
        print("Provide either a source path or --leaderboard-entry", file=sys.stderr)
        sys.exit(1)

    html = render_card(data)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
