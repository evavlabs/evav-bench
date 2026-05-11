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
@click.version_option(__version__, prog_name="EVAV", message="EVAV v%(version)s — Operational Alignment Battery")
@click.option("--log-level", default=None,
              help="Override OA_LOG_LEVEL env var (DEBUG / INFO / WARNING / ERROR).")
def cli(log_level: str | None):
    """EVAV — Operational Alignment Battery CLI.

    Domain-agnostic matched-pair AI deployment-safety auditing.

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
@click.option("--resume/--no-resume", default=True,
              help="Skip cells that have already completed.")
@click.option("--supabase/--local", default=False,
              help="Supabase mode hooks into the existing EVAV Engine (Tier 2/3). "
                   "Local mode (default) calls model APIs directly.")
@click.option("--workers", default=1, type=int,
              help="Concurrent workers. v1 supports 1 only; v1.1 will support N>1.")
@click.option("--dry-run/--no-dry-run", default=False,
              help="Validate, enumerate, and estimate cost without making API calls.")
def run(config_path: Path, output: Path, resume: bool, supabase: bool, workers: int, dry_run: bool):
    """Run a battery and write per-cell results to OUTPUT."""
    cfg = _load_config(config_path)
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


def main():
    cli(prog_name="evav")


if __name__ == "__main__":
    main()
