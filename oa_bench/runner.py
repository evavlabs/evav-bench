"""Per-cell test execution with concurrent dispatch.

Given a Cell + BatteryConfig + a model adapter + domain pack, runs N matched
pairs and emits per-cell results. Cells are resumable — each cell's results
land in a content-addressed file under output_dir, and run_cell skips any
cell whose results already exist and are valid.

Concurrency: matched pairs within a cell run concurrently up to `workers`.
Each pair = 2 sequential API calls (base then twin); the pairs themselves
parallelize.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .battery import BatteryConfig, Cell
from .models._base import ModelAdapter
from .domains._base import DomainPack
from .scoring.matched_pair import score_matched_pair
from .scoring.masking import classify_masking
from .scoring.precursor import extract_precursor_signals


log = logging.getLogger("oa_bench.runner")


@dataclass
class PairResult:
    pair_id: str
    base_decision: str
    base_reasoning: str
    twin_decision: str
    twin_reasoning: str
    violated: bool
    masking_form: Optional[str]
    latency_ms_base: int
    latency_ms_twin: int
    tokens_base: int
    tokens_twin: int
    manipulated_variable: str = ""
    pair_meta: dict | None = None


@dataclass
class CellResult:
    cell: Cell
    pairs: list[PairResult]
    violation_rate_pct: float
    masking_rate_pct: float
    masking_forms: dict[str, float]
    precursor_signals: dict[str, float]
    n_pairs: int
    elapsed_seconds: float
    errors: list[str]


def _cell_result_path(output_dir: Path, cell_id: str) -> Path:
    safe = cell_id.replace("(", "_").replace(")", "_").replace(".", "-")
    return output_dir / f"cell_{safe}.json"


def cell_already_complete(output_dir: Path, cell: Cell) -> bool:
    p = _cell_result_path(output_dir, cell.cell_id)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("complete") is True
    except Exception:
        return False


def _execute_one_pair(
    i: int,
    base_input: str,
    twin_input: str,
    pair_meta: dict,
    cell: Cell,
    cfg: BatteryConfig,
    model: ModelAdapter,
    domain: DomainPack,
) -> tuple[PairResult | None, str | None]:
    """Synchronously execute one matched pair. Returns (PairResult, error)."""
    try:
        # Trading pack embeds the system prompt in the user input via [SYSTEM]...
        # markers. Pull it out if present.
        base_sys, base_user = _split_embedded_system(base_input, cfg.system_prompt)
        twin_sys, twin_user = _split_embedded_system(twin_input, cfg.system_prompt)

        base_out = model.invoke(
            system_prompt=base_sys,
            user_input=base_user,
            temperature=cell.temperature,
            seed=cell.seed,
        )
        twin_out = model.invoke(
            system_prompt=twin_sys,
            user_input=twin_user,
            temperature=cell.temperature,
            seed=cell.seed,
        )

        violated = score_matched_pair(
            base_response=base_out,
            twin_response=twin_out,
            scoring_config=cfg.scoring,
            domain=domain,
        )

        masking_form = None
        if violated:
            masking_form = classify_masking(
                response=twin_out,
                scoring_config=cfg.scoring,
                domain=domain,
            )

        return PairResult(
            pair_id=pair_meta.get("pair_id", f"pair-{i:04d}"),
            base_decision=base_out.get("decision", ""),
            base_reasoning=base_out.get("reasoning", "")[:500],
            twin_decision=twin_out.get("decision", ""),
            twin_reasoning=twin_out.get("reasoning", "")[:500],
            violated=violated,
            masking_form=masking_form,
            latency_ms_base=base_out.get("latency_ms", 0),
            latency_ms_twin=twin_out.get("latency_ms", 0),
            tokens_base=base_out.get("tokens", 0),
            tokens_twin=twin_out.get("tokens", 0),
            manipulated_variable=pair_meta.get("manipulated_variable", ""),
            pair_meta=pair_meta,
        ), None
    except Exception as e:
        return None, f"pair {i}: {type(e).__name__}: {e}"


def _split_embedded_system(text: str, fallback: str) -> tuple[str, str]:
    """If text has [SYSTEM]...[/SYSTEM] block at start, extract it; else use fallback."""
    marker_start = "[SYSTEM]"
    marker_end = "[/SYSTEM]"
    if text.startswith(marker_start):
        end = text.find(marker_end)
        if end > 0:
            sys = text[len(marker_start):end].strip()
            user = text[end + len(marker_end):].strip()
            return sys, user
    return fallback, text


def run_cell(
    cell: Cell,
    cfg: BatteryConfig,
    model: ModelAdapter,
    domain: DomainPack,
    output_dir: Path,
    resume: bool = True,
    workers: int = 1,
) -> CellResult:
    """Execute one cell. Returns aggregated CellResult and writes to disk.

    Pairs within a cell run concurrently up to `workers` threads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = _cell_result_path(output_dir, cell.cell_id)

    if resume and cell_already_complete(output_dir, cell):
        log.info("skip %s (already complete)", cell.cell_id)
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return _deserialize_cell_result(data, cell)

    log.info("run %s  n=%d  workers=%d", cell.cell_id, cfg.scope.n_per_cell, workers)
    t_start = time.time()
    errors: list[str] = []
    pairs: list[PairResult] = []

    # Generate N matched pairs deterministically from cell.seed
    scenario_pairs = list(domain.generate_pairs(
        n=cfg.scope.n_per_cell,
        seed=cell.seed,
        cell=cell,
        cfg=cfg,
    ))

    if workers <= 1:
        for i, (base_input, twin_input, pair_meta) in enumerate(scenario_pairs):
            pr, err = _execute_one_pair(i, base_input, twin_input, pair_meta, cell, cfg, model, domain)
            if pr:
                pairs.append(pr)
            if err:
                errors.append(err)
                log.warning(err)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_execute_one_pair, i, b, t, m, cell, cfg, model, domain): i
                for i, (b, t, m) in enumerate(scenario_pairs)
            }
            for fut in as_completed(futs):
                pr, err = fut.result()
                if pr:
                    pairs.append(pr)
                if err:
                    errors.append(err)
                    log.warning(err)

    pairs.sort(key=lambda p: p.pair_id)

    n_pairs = len(pairs)
    n_violations = sum(1 for p in pairs if p.violated)
    violation_rate = (n_violations / n_pairs * 100) if n_pairs > 0 else 0.0
    n_masked = sum(1 for p in pairs if p.violated and p.masking_form)
    masking_rate = (n_masked / max(1, n_violations) * 100)

    forms = {"affirmative_cite_substitute": 0, "selective_omission": 0, "legitimate_risk_construction": 0}
    for p in pairs:
        if p.violated and p.masking_form in forms:
            forms[p.masking_form] += 1
    forms_pct = {k: (v / max(1, n_violations) * 100) for k, v in forms.items()}

    precursor_signals = extract_precursor_signals(pairs)

    elapsed = time.time() - t_start

    cr = CellResult(
        cell=cell,
        pairs=pairs,
        violation_rate_pct=round(violation_rate, 2),
        masking_rate_pct=round(masking_rate, 2),
        masking_forms=forms_pct,
        precursor_signals=precursor_signals,
        n_pairs=n_pairs,
        elapsed_seconds=round(elapsed, 1),
        errors=errors,
    )

    log.info("done %s  vr=%.1f%%  mask=%.1f%%  n=%d  err=%d  %.1fs",
             cell.cell_id, cr.violation_rate_pct, cr.masking_rate_pct,
             cr.n_pairs, len(cr.errors), cr.elapsed_seconds)

    _serialize_cell_result(cr, result_path)
    return cr


def _serialize_cell_result(cr: CellResult, path: Path):
    payload = {
        "complete": True,
        "schema_version": "v1.0",
        "cell": cr.cell.model_dump(),
        "violation_rate_pct": cr.violation_rate_pct,
        "masking_rate_pct": cr.masking_rate_pct,
        "masking_forms": cr.masking_forms,
        "precursor_signals": cr.precursor_signals,
        "n_pairs": cr.n_pairs,
        "elapsed_seconds": cr.elapsed_seconds,
        "errors": cr.errors,
        "pairs": [
            {
                "pair_id": p.pair_id,
                "base_decision": p.base_decision,
                "base_reasoning": p.base_reasoning,
                "twin_decision": p.twin_decision,
                "twin_reasoning": p.twin_reasoning,
                "violated": p.violated,
                "masking_form": p.masking_form,
                "latency_ms_base": p.latency_ms_base,
                "latency_ms_twin": p.latency_ms_twin,
                "tokens_base": p.tokens_base,
                "tokens_twin": p.tokens_twin,
                "manipulated_variable": p.manipulated_variable,
            }
            for p in cr.pairs
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _deserialize_cell_result(data: dict, cell: Cell) -> CellResult:
    fields = set(PairResult.__annotations__.keys())
    pairs = [
        PairResult(**{k: v for k, v in p.items() if k in fields})
        for p in data.get("pairs", [])
    ]
    return CellResult(
        cell=cell,
        pairs=pairs,
        violation_rate_pct=data["violation_rate_pct"],
        masking_rate_pct=data["masking_rate_pct"],
        masking_forms=data.get("masking_forms", {}),
        precursor_signals=data.get("precursor_signals", {}),
        n_pairs=data["n_pairs"],
        elapsed_seconds=data["elapsed_seconds"],
        errors=data.get("errors", []),
    )
