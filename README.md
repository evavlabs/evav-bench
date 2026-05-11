# evav-bench

The open-source CLI for the **EVAV Operational Alignment Battery** — a matched-pair causal-identification test suite for AI agents making decisions in regulated industries.

[![License](https://img.shields.io/badge/license-Proprietary-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Battery](https://img.shields.io/badge/battery-v1.0-CC5500.svg)](https://evav.ai/methodology)

```bash
pip install evav-bench
```

See the public leaderboard at **[evav.ai/leaderboard](https://evav.ai/leaderboard)**. Read the methodology at **[evav.ai/methodology](https://evav.ai/methodology)**.

---

## What This Is

EVAV is the behavioral intelligence layer for AI in regulated decisions. The Operational Alignment Battery tests whether a model preserves its stated rules under deployment-realistic pressure.

This CLI runs the full battery — 8 axes, 10 test groups, up to 80 cells — against any frontier model and produces an Evaluation Card.

**Key findings from the reference corpus** (209,072 decisions across 8 frontier models):
- 86% of violations would pass conventional compliance review (compliance masking)
- Claude Sonnet 4 ranges from 0% to 98% violation rate depending on documentation tier
- DeepSeek V3 swings 50-94% on identical configuration across PRNG seeds

## Quick Start

```bash
# Install
pip install evav-bench

# Set API key for your provider
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY, DEEPSEEK_API_KEY, etc.

# Run the smoke test (~$0.10 on DeepSeek)
evav run examples/battery.smoketest_deepseek.json --output ./results/

# Render the Evaluation Card
evav render-card ./results/ --format md > card.md
evav render-card ./results/ --format json > card.json

# Or render a beautiful visual HTML card
python cards/renderer/render.py ./results/ --out card.html
```

## Supported Providers

| Provider | Env var | Example model |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Google | `GEMINI_API_KEY` | `gemini-2.5-pro` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-4-maverick` |

## CLI Commands

| Command | Purpose |
|---|---|
| `evav validate <config>` | Validate config, print cells + cost estimate |
| `evav run <config> -o <dir>` | Execute the battery |
| `evav resume <dir>` | Resume an interrupted run |
| `evav render-card <dir>` | Render Evaluation Card (md/json/html) |
| `evav render-report <dir>` | Render full Audit Report |
| `evav render-failure-map <dir>` | Failure Cell Map JSON |
| `evav render-precursor-profile <dir>` | Per-model precursor signal profile |
| `evav render-interventions <dir>` | Intervention recommendations |
| `evav baseline <dir>` | Save a drift baseline |
| `evav drift-diff <baseline> <new>` | Compare runs for drift |
| `evav compare <dir-a> <dir-b>` | Diff two runs |

## The Three Card Types

This repo ships with two card templates in `cards/templates/`:

1. **Deployment Card** (`deployment_card.html`) — single model, single config. The output of every `evav run`.
2. **Benchmark Card** (`benchmark_card.html`) — cross-model matrix. Used for the public leaderboard at evav.ai.

See `cards/README.md` for the full template reference.

## Example Configs

| File | Domain | Cells | Suitable for |
|---|---|---|---|
| `examples/battery.smoketest_deepseek.json` | Healthcare (lightweight) | 7 | First test (~$0.10) |
| `examples/battery.healthcare.example.json` | Medicare prior auth | 51 | Full audit, replicating research |
| `examples/battery.lending.example.json` | Consumer credit (ECOA) | 28 | Lending compliance |
| `examples/battery.trading.example.json` | Market-making (Reg NMS) | 24 | Trading compliance |

## Data

- **Public corpus**: 209,072-decision matched-pair dataset at [huggingface.co/datasets/evav/operational-alignment-corpus](https://huggingface.co/datasets/evav/operational-alignment-corpus) (CC-BY-4.0)
- **Reference Evaluation Cards**: rendered examples for the 8 frontier models in `cards/examples/`

## Methodology

Matched-pair causal identification. PRNG-deterministic scenario generation (Mulberry32). 8 axes (pressure, doc tier, anchor, phrasing, authority, intervention, seed, temp). 10 test groups (A baselines → J forensics). Validated by SAE-based mechanistic interpretability at 81.2% probe accuracy.

Full methodology: [evav.ai/methodology](https://evav.ai/methodology)

Paper (NeurIPS 2026 Datasets & Benchmarks Track): [arxiv.org/abs/2026.xxxxx](https://arxiv.org/abs/2026.xxxxx)

## Citation

```bibtex
@inproceedings{cruz2026evav,
  title     = {Evaluating AI Specification Gaming Under Matched-Pair Pressure},
  author    = {Cruz, Anthony},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026},
  url       = {https://evav.ai/research}
}
```

## Enterprise

For production deployment safety audits with full deliverables (Audit Report, Failure Cell Map, Intervention Recommendations, Precursor Profile, Compliance Artifact templates for HIPAA / ECOA / SOC 2 / EU AI Act / NIST AI RMF), see [evav.ai/product](https://evav.ai/product).

The CLI in this repo runs the same instrument used in our paid Tier 1 audits — the difference is the deliverables, the audit team, and the compliance-artifact mapping that go around it.

## License

Proprietary. Free for evaluating your own models, internal R&D, and academic research with citation. Redistribution and commercial use require permission. See [LICENSE](LICENSE).

## Status

This is **v1.0** — the initial public release. See [CHANGELOG.md](CHANGELOG.md) for what's included.

| Component | Status |
|---|---|
| CLI commands | ✅ stable |
| Anthropic, OpenAI, DeepSeek adapters | ✅ tested end-to-end |
| Google, OpenRouter adapters | ⚠️ scaffolded, less battle-tested |
| Healthcare domain pack | ✅ full prompts |
| Lending, trading domain packs | ✅ ported from research |
| Two-stage masking classifier | ✅ heuristic + LLM |
| 25-signal precursor extractor | ✅ working |
| Concurrent execution | ✅ `--workers N` |
| Retry + rate limiting | ✅ exponential backoff |
| Drift baseline + diff | ✅ working |

## Support

- **Bugs / features:** [GitHub Issues](https://github.com/evavlabs/evav-bench/issues)
- **Methodology questions:** [evav.ai/methodology](https://evav.ai/methodology)
- **Commercial / enterprise:** [hello@evav.ai](mailto:hello@evav.ai)

---

Built by [EVAV](https://evav.ai). Methodology: [Operational Alignment v1.0](https://evav.ai/methodology).
