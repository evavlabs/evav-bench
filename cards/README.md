# Evaluation Cards — Canonical Home

This is the single source of truth for every Evaluation Card artifact: templates, renderer, example renders, supporting data.

Anything downstream (the Lovable website, customer audit PDF exports, marketing assets, the press kit, conference posters, internal dashboards) consumes from this folder. Don't fork — extend here, sync elsewhere.

---

## Folder Layout

```
cards/
├── README.md                       # this file
├── templates/                      # raw HTML templates with {{ placeholders }}
│   ├── deployment_card.html        # Type A — single model, single config
│   ├── benchmark_card.html         # Type B — cross-model matrix
│   └── per_axis_card.html          # Type C — one pressure × all models × all domains (TODO)
├── renderer/
│   └── render.py                   # Python renderer that fills templates from data
├── examples/                       # pre-rendered example cards (the leaderboard set)
│   ├── claude-sonnet-4.html        # Type A render — Claude on healthcare
│   ├── gpt-4o-2024-08-06.html      # ...
│   ├── ... (6 more Type A renders)
│   ├── benchmark-healthcare.html   # Type B render — all 8 models compared
│   └── smoketest-deepseek-live.html # Type A from a real live API run
└── data/
    └── leaderboard.json            # source data for the 8 reference cards
```

---

## The Three Card Types

### Type A — Deployment Card (`deployment_card.html`)

**What it shows:** One model's behavior under all pressure types, in one customer's domain, with their config.

**Used for:**
- The `/run` feature output on evav.ai — a visitor tested their model
- The `/leaderboard/[model_id]` detail pages — drill into one row
- Tier 1 audit deliverables — the headline artifact a paying customer gets
- Each customer engagement produces dozens of these (one per configuration tested)

**Data shape:**
- Header: model identity + test metadata
- Instrument validation (A-group baselines)
- Documentation dial (D1–D3 stratification)
- Pressure surface (B1–B7 + C1, full-width bars)
- Interventions (G1–G3 deltas + transport status)
- Forensics (masking rate, SCF, MEA, latent capture, cross-seed)
- Overall Readiness (4 big numbers + verdict pill)
- Key Findings (model-specific bullets)
- Footer: methodology link, generator version

### Type B — Benchmark Card (`benchmark_card.html`)

**What it shows:** Cross-model comparison. Same test cells, all 8 models in columns.

**Used for:**
- The `/leaderboard` summary page on evav.ai — the credibility surface
- Research-oriented one-pagers (NeurIPS poster, papers, press materials)
- The "state of frontier model alignment under pressure" headline artifact
- Goes out to industry analysts, regulators, journalists

**Data shape:**
- Header: domain + corpus metadata
- 4-cell headline summary: safest, worst, biggest cliff, highest masking
- Matrix: ~19 test cells × 8 models, with section dividers (validation / pressure / docs / interventions / forensics)
- Cross-model observations: 6 findings that ONLY emerge from comparison
- Footer: leaderboard + methodology + paper links

### Type C — Per-Axis Card (`per_axis_card.html`) — TODO

**What it shows:** One pressure axis (e.g., "reward") across ALL models AND ALL domains.

**Used for:**
- Methodology page sections on evav.ai — "how does reward pressure work?"
- Policy briefs — "the disparate-impact effects of optimization pressure across financial services"
- Research deep-dives — drill into one axis of the battery

**Status:** Not built yet. Lower priority than A and B.

---

## Renderer Usage

```bash
cd cards/

# Render a single model card (Type A) from a leaderboard entry
python renderer/render.py --leaderboard-entry data/leaderboard.json#claude-sonnet-4 \
                          --out examples/claude-sonnet-4.html

# Render a Type A card from a live battery run output dir
python renderer/render.py /path/to/oa-bench/results/ --out my_card.html

# Render a Type A card from a JSON card file
python renderer/render.py path/to/card.json --out my_card.html
```

Each render produces a self-contained HTML file (~13 KB for Type A, ~25 KB for Type B). No external assets, no JavaScript, no fonts to load. Embeddable as iframe, downloadable as standalone, printable to PDF via `weasyprint` / `puppeteer`.

---

## Adding a New Card Type

1. Create `templates/new_card_type.html` with `{{ placeholder }}` substitution points
2. Add a renderer function in `renderer/render.py`
3. Document it here (which audience, when used)
4. Add a worked example render to `examples/`
5. If used by the website, sync the template + an example into `web/lovable/assets/templates/` and `web/lovable/assets/cards/`

---

## Brand Consistency

Every card MUST use the canonical EVAV palette:

| Token | Value |
|---|---|
| `--bg` | `#FAFAF8` |
| `--surface` | `#FFFFFF` |
| `--text` | `#0A0A0A` |
| `--text-muted` | `#71717A` |
| `--accent` | `#CC5500` |
| `--border` | `#E7E5E4` |

Inter for body, JetBrains Mono for numbers, tabular figures everywhere.

Severity uses accent saturation, NOT green/yellow/red color coding. Lets the data carry the meaning.

Wordmark is always: `EVAV` (with the "E" in `--accent`) + protocol stamp `Operational Alignment v1.0`.

---

## Versioning

- **Card schema version** — bumps on layout / field changes (currently v1.0)
- **Battery version** — bumps on the underlying instrument (currently v1.0)
- **Cards are not auto-regenerated** — explicit re-render needed when templates change

When a template changes, regenerate all `examples/` and sync `lovable/assets/`. Document the change in `products/distribution/CHANGELOG.md` under the relevant version.

---

## Related Locations

| Where | What |
|---|---|
| `cards/` (this folder) | Templates + renderer + examples — single source of truth |
| `web/lovable/assets/` | COPIES sync'd for Lovable consumption |
| `web/api/oa_api/main.py` | Backend serves rendered cards via `/api/jobs/{id}/card` |
| `cli/oa_bench/card.py` | CLI's local card renderer (the markdown/JSON form) |
| `EVAV_Knowledge/MASTER_RESULTS_CHART.md` | The reference data source for `data/leaderboard.json` |
