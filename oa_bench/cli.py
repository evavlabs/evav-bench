"""oa-bench CLI entrypoint.

Click-based command surface. Subcommands map to the operations described in
cli/README.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from oa_bench import __version__, __battery_version__
from oa_bench.battery import BatteryConfig, enumerate_cells, estimate_cost
from oa_bench.runner import run_cell, cell_already_complete

# Force ASCII-safe rendering on legacy Windows consoles (cp1252).
# Set PYTHONIOENCODING=utf-8 in the environment to opt back into unicode.
_use_legacy = os.environ.get("PYTHONIOENCODING", "").lower() != "utf-8" and sys.platform == "win32"
console = Console(legacy_windows=_use_legacy, width=140, no_color=False)


def _load_config(path: Path) -> BatteryConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return BatteryConfig(**raw)


@click.group()
@click.version_option(__version__, prog_name="oa-bench")
@click.option("--log-level", default=None,
              help="Override OA_LOG_LEVEL env var (DEBUG / INFO / WARNING / ERROR).")
def cli(log_level: str | None):
    """OA Evaluation Battery CLI.

    Domain-agnostic matched-pair deployment-safety auditing.

    See cli/README.md for full documentation.
    """
    from oa_bench._logging import configure_logging, get_log_level_from_env
    level = log_level or get_log_level_from_env()
    configure_logging(level=level, console=True, json_file=False)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--estimate-cost/--no-estimate-cost", default=True,
              help="Estimate API cost for this config.")
def validate(config_path: Path, estimate_cost: bool):
    """Validate a battery.config.json. Prints the resolved cell list and cost estimate."""
    try:
        cfg = _load_config(config_path)
    except Exception as e:
        console.print(f"[red][fail] Invalid config:[/red] {e}")
        sys.exit(1)

    console.print(f"[green][ok] Config valid[/green]   engagement_id={cfg.engagement_id}")
    console.print(f"  customer:  {cfg.customer}")
    console.print(f"  domain:    {cfg.domain}")
    console.print(f"  model:     {cfg.model.provider}/{cfg.model.name}")
    console.print(f"  battery:   {cfg.battery_version}")

    cells = enumerate_cells(cfg)

    table = Table(title=f"Cells to run ({len(cells)})")
    table.add_column("Cell ID", style="cyan")
    table.add_column("Group")
    table.add_column("Name")
    table.add_column("Pressure")
    table.add_column("Doc")
    table.add_column("Anchor")
    table.add_column("Failure modes")
    for c in cells:
        table.add_row(
            c.cell_id, c.group, c.name, c.pressure, c.doc_tier, c.anchor,
            ", ".join(c.failure_modes_detected),
        )
    console.print(table)

    if estimate_cost:
        est = oa_estimate(cfg, cells)
        console.print(f"\n[bold]Cost estimate[/bold]")
        console.print(f"  Calls:    {est['n_calls']:,}")
        console.print(f"  Tokens:   ~{est['est_tokens']:,}")
        console.print(f"  Cost:     ~${est['est_cost_usd']}")
        console.print(f"  Runtime:  ~{est['est_runtime_min']} minutes")


def oa_estimate(cfg, cells):
    return estimate_cost(cfg, cells)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path),
              help="Output directory for per-cell results.")
@click.option("--scope", type=click.Choice(["minimal", "quick", "standard", "full"]), default=None,
              help="Override config scope with a preset (minimal/quick/standard/full).")
@click.option("--resume/--no-resume", default=True,
              help="Skip cells that have already completed.")
@click.option("--supabase/--local", default=False,
              help="Supabase mode hooks into the existing EVAV Engine (Tier 2/3). "
                   "Local mode (default) calls model APIs directly.")
@click.option("--workers", default=1, type=int,
              help="Concurrent workers. v1 supports 1 only; v1.1 will support N>1.")
@click.option("--dry-run/--no-dry-run", default=False,
              help="Validate, enumerate, and estimate cost without making API calls.")
def run(config_path: Path, output: Path, scope: str | None, resume: bool, supabase: bool, workers: int, dry_run: bool):
    """Run a battery and write per-cell results to OUTPUT."""
    cfg = _load_config(config_path)
    if scope:
        from oa_bench.battery import SCOPE_PRESETS, ScopeConfig
        preset = {k: v for k, v in SCOPE_PRESETS[scope].items() if not k.startswith("_")}
        cfg.scope = ScopeConfig(**preset)
        console.print(f"[bold]Scope override:[/bold] {scope}  -- {SCOPE_PRESETS[scope]['_description']}")
    cells = enumerate_cells(cfg)
    output.mkdir(parents=True, exist_ok=True)

    # Configure per-run JSON log file
    from oa_bench._logging import configure_logging, get_log_level_from_env
    configure_logging(output_dir=output, level=get_log_level_from_env(),
                      console=True, json_file=True)

    # Copy config alongside results for reproducibility
    (output / "battery.config.json").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    est = oa_estimate(cfg, cells)
    console.print(f"[bold]Battery:[/bold] {cfg.battery_version}  cells={len(cells)}  "
                  f"est_cost=${est['est_cost_usd']}  est_runtime~{est['est_runtime_min']}min")

    if dry_run:
        console.print("[yellow]Dry run — no API calls.[/yellow]")
        return

    if supabase:
        console.print("[yellow]Supabase mode is not yet implemented in v1.0. "
                      "Falling back to local mode.[/yellow]")

    # Resolve model + domain
    from oa_bench.models import get_adapter
    from oa_bench.domains import get_domain_pack
    try:
        model = get_adapter(cfg.model)
    except Exception as e:
        console.print(f"[red]Model adapter error:[/red] {e}")
        sys.exit(2)
    try:
        domain = get_domain_pack(cfg.domain)
    except Exception as e:
        console.print(f"[red]Domain pack error:[/red] {e}")
        sys.exit(3)

    completed = 0
    skipped = 0
    failed = 0
    for cell in cells:
        if resume and cell_already_complete(output, cell):
            skipped += 1
            console.print(f"[dim]skip {cell.cell_id} (already complete)[/dim]")
            continue
        try:
            cr = run_cell(cell, cfg, model, domain, output, resume=resume, workers=workers)
            completed += 1
            console.print(f"[green][ok][/green] {cell.cell_id}  "
                          f"violation_rate={cr.violation_rate_pct}%  "
                          f"masking={cr.masking_rate_pct}%  "
                          f"n={cr.n_pairs}  "
                          f"elapsed={cr.elapsed_seconds}s  "
                          f"err={len(cr.errors)}")
        except Exception as e:
            failed += 1
            console.print(f"[red][fail] {cell.cell_id}:[/red] {type(e).__name__}: {e}")

    console.print(f"\n[bold]Done.[/bold]  completed={completed}  skipped={skipped}  failed={failed}")
    console.print(f"Results in: {output}/")
    console.print(f"Render card:   oa-bench render-card {output}/ --format md")
    console.print(f"Render report: oa-bench render-report {output}/")


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["md", "json", "pdf"]), default="md")
def render_card(output_dir: Path, fmt: str):
    """Render the Evaluation Card from a completed battery run."""
    from oa_bench.card import render_card_from_dir
    out = render_card_from_dir(output_dir, fmt=fmt)
    click.echo(out)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def render_report(output_dir: Path):
    """Render the full Audit Report (markdown) from a completed battery run."""
    from oa_bench.report import render_report_from_dir
    out = render_report_from_dir(output_dir)
    click.echo(out)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def render_failure_map(output_dir: Path):
    """Render the Failure Cell Map (JSON) from a completed battery run."""
    from oa_bench.report import render_failure_cell_map
    out = render_failure_cell_map(output_dir)
    click.echo(out)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def render_precursor_profile(output_dir: Path):
    """Render the Precursor Profile (JSON)."""
    from oa_bench.report import render_precursor_profile
    out = render_precursor_profile(output_dir)
    click.echo(out)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def render_interventions(output_dir: Path):
    """Render Intervention Recommendations (markdown)."""
    from oa_bench.report import render_intervention_recommendations
    out = render_intervention_recommendations(output_dir)
    click.echo(out)


@cli.command()
@click.argument("dir_a", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("dir_b", type=click.Path(exists=True, file_okay=False, path_type=Path))
def compare(dir_a: Path, dir_b: Path):
    """Diff two battery runs (model comparison or drift detection)."""
    from oa_bench.report import render_comparison
    out = render_comparison(dir_a, dir_b)
    click.echo(out)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--name", "name", default=None, help="Optional baseline name; defaults to engagement-id + date.")
def baseline(output_dir: Path, name: str | None):
    """Save a completed battery run as a drift baseline."""
    from oa_bench.drift import save_baseline
    b = save_baseline(output_dir, baseline_name=name)
    console.print(f"[green][ok] Baseline saved[/green]  name={b['name']}  "
                  f"hash={b['content_hash']}  cells={b['n_cells']}")
    console.print(f"  -> {output_dir}/drift_baseline.json")


@cli.command()
@click.argument("baseline_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("new_run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--significance", default=5.0, type=float,
              help="Per-cell drift threshold in percentage points (default 5pp).")
@click.option("--output", default=None, type=click.Path(path_type=Path),
              help="Write diff to file (JSON). Defaults to stdout.")
def drift_diff(baseline_path: Path, new_run_dir: Path, significance: float, output: Path | None):
    """Compare a new battery run against a saved drift baseline."""
    from oa_bench.drift import diff_against_baseline
    diff = diff_against_baseline(baseline_path, new_run_dir, significance_pp=significance)
    text = json.dumps(diff, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green][ok] Diff written[/green]  -> {output}")
    else:
        click.echo(text)

    console.print(f"\n[bold]Drift summary[/bold]")
    console.print(f"  Cells compared:  {diff['n_cells_compared']}")
    console.print(f"  Cells drifted:   {diff['n_cells_drifted']} (>= {significance}pp)")
    console.print(f"  Max drift:       {diff['max_drift_pp']:+.1f}pp")
    console.print(f"  Mean drift:      {diff['mean_drift_pp']:+.1f}pp")


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def supabase_upload(output_dir: Path):
    """Push results to Supabase for Tier 2/3 ingestion via EVAV_Engine."""
    try:
        from oa_bench.supabase_mode import upload_run
    except ImportError:
        console.print("[red]Supabase extras not installed.[/red] Run: pip install -e \".[supabase]\"")
        sys.exit(2)
    try:
        result = upload_run(output_dir)
        console.print(f"[green][ok] Uploaded[/green]  eval_id={result.get('eval_id')}")
    except Exception as e:
        console.print(f"[red]Upload failed:[/red] {type(e).__name__}: {e}")
        sys.exit(3)


# ────────────────────────────────────────────────────────────────────────────
# Operational alignment modules — beyond core matched-pair pressure battery
# ────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--provider", required=True, type=click.Choice(["anthropic", "openai", "google", "deepseek", "openrouter"]))
@click.option("--model", "model_name", required=True)
@click.option("--source", default="builtin", help="Dataset source: builtin, truthfulqa, or path to JSON")
@click.option("--pressure-overlay", default=None, help="Optional pressure narrative to test hallucination x pressure")
@click.option("--output", default="hallucination.json", type=click.Path(path_type=Path))
def hallucination(provider, model_name, source, pressure_overlay, output):
    """Test for factual hallucination, with optional pressure overlay."""
    from oa_bench.battery import ModelConfig
    from oa_bench.models import get_adapter
    from oa_bench.modules.hallucination import run_hallucination_test

    cfg = ModelConfig(provider=provider, name=model_name)
    model = get_adapter(cfg)
    console.print(f"Running hallucination test against {provider}/{model_name}...")
    result = run_hallucination_test(model, source=source, pressure_overlay=pressure_overlay)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green][ok][/green] Results: {output}")
    if "hallucination_rate_pct" in result:
        console.print(f"  Hallucination rate: {result['hallucination_rate_pct']}%")
        console.print(f"  Refusal rate:       {result['refusal_rate_pct']}%")
        console.print(f"  Correct rate:       {result['correct_rate_pct']}%")


@cli.command()
@click.option("--provider", required=True, type=click.Choice(["anthropic", "openai", "google", "deepseek", "openrouter"]))
@click.option("--model", "model_name", required=True)
@click.option("--mode", default="auto", type=click.Choice(["auto", "garak", "builtin"]),
              help="auto picks garak if installed, else builtin")
@click.option("--output", default="adversarial.json", type=click.Path(path_type=Path))
def adversarial(provider, model_name, mode, output):
    """Test prompt-injection / jailbreak resistance. Wraps NVIDIA garak if installed."""
    from oa_bench.battery import ModelConfig
    from oa_bench.models import get_adapter
    from oa_bench.modules.adversarial import run_adversarial

    cfg = ModelConfig(provider=provider, name=model_name)
    model = get_adapter(cfg)
    console.print(f"Running adversarial test against {provider}/{model_name} (mode={mode})...")
    result = run_adversarial(model, mode=mode, provider=provider, name=model_name)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green][ok][/green] Results: {output}")
    if "overall_attack_success_rate_pct" in result:
        console.print(f"  Attack success rate: {result['overall_attack_success_rate_pct']}%")


@cli.command()
@click.option("--provider", required=True, type=click.Choice(["anthropic", "openai", "google", "deepseek", "openrouter"]))
@click.option("--model", "model_name", required=True)
@click.option("--pressure-overlay", default=None)
@click.option("--output", default="refusal.json", type=click.Path(path_type=Path))
def refusal(provider, model_name, pressure_overlay, output):
    """Test refusal calibration -- over-refusal of legitimate vs under-refusal of harmful."""
    from oa_bench.battery import ModelConfig
    from oa_bench.models import get_adapter
    from oa_bench.modules.refusal import run_refusal_test

    cfg = ModelConfig(provider=provider, name=model_name)
    model = get_adapter(cfg)
    console.print(f"Running refusal calibration test against {provider}/{model_name}...")
    result = run_refusal_test(model, pressure_overlay=pressure_overlay)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green][ok][/green] Results: {output}")
    if "over_refusal_rate_pct" in result:
        console.print(f"  Over-refusal (safe refused):    {result['over_refusal_rate_pct']}%")
        console.print(f"  Under-refusal (harmful accepted): {result['under_refusal_rate_pct']}%")
        console.print(f"  Overall accuracy:                  {result['overall_accuracy_pct']}%")


@cli.command()
@click.option("--provider", required=True, type=click.Choice(["anthropic", "openai", "google", "deepseek", "openrouter"]))
@click.option("--model", "model_name", required=True)
@click.option("--pressure-overlay", default=None)
@click.option("--output", default="privacy.json", type=click.Path(path_type=Path))
def privacy(provider, model_name, pressure_overlay, output):
    """Test PII / privacy leakage. Wraps Microsoft Presidio if installed."""
    from oa_bench.battery import ModelConfig
    from oa_bench.models import get_adapter
    from oa_bench.modules.privacy import run_privacy_test

    cfg = ModelConfig(provider=provider, name=model_name)
    model = get_adapter(cfg)
    console.print(f"Running privacy test against {provider}/{model_name}...")
    result = run_privacy_test(model, pressure_overlay=pressure_overlay)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green][ok][/green] Results: {output}")
    if "overall_leak_rate_pct" in result:
        console.print(f"  PII leak rate: {result['overall_leak_rate_pct']}%")


@cli.command(name="schema-check")
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--schema", default=None, type=click.Path(path_type=Path),
              help="JSON file defining expected schema. Defaults to healthcare {decision, reasoning}.")
@click.option("--output", default="schema_fidelity.json", type=click.Path(path_type=Path))
def schema_check(results_dir, schema, output):
    """Analyze schema fidelity -- does the model break output format under pressure?"""
    from oa_bench.modules.schema_fidelity import analyze_results_directory

    if schema:
        expected = json.loads(Path(schema).read_text(encoding="utf-8"))
    else:
        expected = {"decision": "string", "reasoning": "string"}

    result = analyze_results_directory(results_dir, expected)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green][ok][/green] Results: {output}")
    if "overall_schema_break_rate_pct" in result:
        console.print(f"  Schema break rate: {result['overall_schema_break_rate_pct']}%")
        if result.get("high_break_cells"):
            console.print(f"  Cells with >10% schema breaks: {len(result['high_break_cells'])}")


@cli.command()
@click.option("--provider", required=True, type=click.Choice(["anthropic", "openai", "google", "deepseek", "openrouter"]))
@click.option("--model", "model_name", required=True)
@click.option("--decisions-csv", default=None, help="Customer's decision log CSV (else uses synthetic fixture)")
@click.option("--output", default="bias.json", type=click.Path(path_type=Path))
def bias(provider, model_name, decisions_csv, output):
    """Bias / disparate-impact audit. Wraps Fairlearn if installed."""
    from oa_bench.battery import ModelConfig
    from oa_bench.models import get_adapter
    from oa_bench.modules.bias import run_bias_audit

    cfg = ModelConfig(provider=provider, name=model_name)
    model = get_adapter(cfg)
    console.print(f"Running bias audit against {provider}/{model_name}...")
    result = run_bias_audit(model, decisions_csv=decisions_csv)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green][ok][/green] Results: {output}")
    if "demographic_parity_diff_pp" in result:
        console.print(f"  Demographic parity diff: {result['demographic_parity_diff_pp']}pp")
        console.print(f"  Four-fifths impact ratio: {result.get('four_fifths_impact_ratio')}")
        if result.get("four_fifths_violated"):
            console.print(f"  [red][!] ADVERSE IMPACT DETECTED[/red]")


@cli.command(name="tool-safety")
@click.option("--provider", required=True, type=click.Choice(["anthropic", "openai", "google", "deepseek", "openrouter"]))
@click.option("--model", "model_name", required=True)
@click.option("--output", default="tool_safety.json", type=click.Path(path_type=Path))
def tool_safety(provider, model_name, output):
    """Tool / function call safety for agentic deployments."""
    from oa_bench.battery import ModelConfig
    from oa_bench.models import get_adapter
    from oa_bench.modules.tool_safety import run_tool_safety_test

    cfg = ModelConfig(provider=provider, name=model_name)
    model = get_adapter(cfg)
    console.print(f"Running tool-safety probes against {provider}/{model_name}...")
    result = run_tool_safety_test(model)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green][ok][/green] Results: {output}")
    if "overall_accuracy_pct" in result:
        console.print(f"  Tool-safety accuracy: {result['overall_accuracy_pct']}%")


@cli.command(name="reasoning-trace")
@click.option("--provider", required=True, type=click.Choice(["anthropic", "openai", "google", "deepseek", "openrouter"]))
@click.option("--model", "model_name", required=True)
@click.option("--output", default="reasoning_trace.json", type=click.Path(path_type=Path))
def reasoning_trace(provider, model_name, output):
    """Detect self-correction failure + trace-decision divergence in reasoning models."""
    from oa_bench.battery import ModelConfig
    from oa_bench.models import get_adapter
    from oa_bench.modules.reasoning_trace import run_reasoning_trace_test

    cfg = ModelConfig(provider=provider, name=model_name)
    model = get_adapter(cfg)
    console.print(f"Running reasoning-trace probes against {provider}/{model_name}...")
    result = run_reasoning_trace_test(model)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"[green][ok][/green] Results: {output}")
    if "overall_failure_rate_pct" in result:
        console.print(f"  Trace-faithfulness failure rate: {result['overall_failure_rate_pct']}%")


@cli.command()
def doctor():
    """Environment health check — verifies API keys, optional deps, network reachability."""
    from importlib import import_module
    import shutil

    console.print("[bold]EVAV doctor[/bold]\n")
    rows = []

    # API keys
    keys = {
        "ANTHROPIC_API_KEY": "Anthropic (Claude)",
        "OPENAI_API_KEY": "OpenAI (GPT)",
        "GOOGLE_API_KEY": "Google (Gemini)",
        "DEEPSEEK_API_KEY": "DeepSeek",
        "OPENROUTER_API_KEY": "OpenRouter",
    }
    for env, label in keys.items():
        present = bool(os.environ.get(env))
        rows.append((label, "set" if present else "missing", "ok" if present else "skip"))

    console.print("[bold]API keys[/bold]")
    for label, state, status in rows:
        tag = "[green][ok][/green]" if status == "ok" else "[yellow][skip][/yellow]"
        console.print(f"  {tag}  {label:25s}  {state}")

    console.print("\n[bold]Optional dependencies[/bold]")
    optional = [
        ("garak", "Adversarial / jailbreak (EVAV Adversarial module)"),
        ("presidio_analyzer", "PII detection (EVAV Privacy module)"),
        ("fairlearn", "Bias / disparate impact (EVAV Bias module)"),
        ("inspect_ai", "Tool safety (EVAV Tool Safety module)"),
        ("supabase", "Supabase upload mode"),
    ]
    for mod, label in optional:
        try:
            import_module(mod)
            console.print(f"  [green][ok][/green]   {mod:22s}  installed  ({label})")
        except ImportError:
            console.print(f"  [yellow][skip][/yellow] {mod:22s}  not installed  ({label})")

    console.print("\n[bold]CLI versions[/bold]")
    console.print(f"  evav CLI:         {__version__}")
    console.print(f"  Battery version:  {__battery_version__}")

    console.print("\n[bold]Recommendation[/bold]")
    if not any(os.environ.get(k) for k in keys):
        console.print("  [red]No API keys found.[/red] Set at least one provider key before running `evav run`.")
    else:
        console.print("  [green]Ready to run.[/green] Try: `evav run examples/healthcare.config.json -o ./results`.")


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True)
def export(output_dir: Path, fmt: str, output: Path):
    """Export a completed run as a single JSON or flat CSV for analysis tools."""
    from oa_bench.card import _load_results, _aggregate_card
    cfg, cells = _load_results(output_dir)
    card = _aggregate_card(cfg, cells)
    if fmt == "json":
        output.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    else:
        import csv
        with output.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell_id", "pressure", "doc_tier", "anchor",
                        "n_pairs", "violation_rate_pct", "masking_rate_pct"])
            for c in cells:
                w.writerow([c.cell_id, c.cell.pressure, c.cell.doc_tier, c.cell.anchor,
                            c.n_pairs, c.violation_rate_pct, c.masking_rate_pct])
    console.print(f"[green][ok][/green] Exported {output}")


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def score(output_dir: Path):
    """Compute the EVAV Score for a completed run."""
    from oa_bench.card import _load_results, _aggregate_card
    from oa_bench.score import score_from_card, DISCLAIMER

    cfg, cells = _load_results(output_dir)
    card = _aggregate_card(cfg, cells)
    result = score_from_card(card)

    console.print(f"\n[bold]EVAV Score: {result.overall}[/bold]  [{result.severity_band}]")
    console.print(f"Weakest module: [yellow]{result.weakest_module}[/yellow] (score {result.weakest_module_score})\n")
    console.print("[bold]Per-module breakdown[/bold]")
    for m in result.module_scores:
        console.print(f"  {m.module_name:22s} score={m.score:5.1f}  (failure rate {m.failure_rate_pct:.1f}%)")
    console.print(f"\n[dim]{DISCLAIMER}[/dim]\n")


def main():
    cli(prog_name="evav")


if __name__ == "__main__":
    main()
