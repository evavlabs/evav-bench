# Changelog

All notable changes to `oa-bench` are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.0] — 2026-05-11

### Added — Initial public release

**The Battery**
- OA Evaluation Battery v1.0 specification: 8 axes, 10 test groups (A–J), ~80 test cells per full run
- Matched-pair causal identification protocol
- PRNG-deterministic scenario generation (Mulberry32)

**The CLI**
- 11 commands: `validate`, `run`, `resume`, `render-card`, `render-report`, `render-failure-map`, `render-precursor-profile`, `render-interventions`, `compare`, `baseline`, `drift-diff`, `supabase-upload`
- Concurrent cell execution via thread pool (`--workers N`)
- Resumable runs (per-cell content-addressed result files)
- Cost estimation pre-run
- Structured JSON logging with file rotation

**Model adapters**
- Anthropic (Claude)
- OpenAI (GPT, o1, o3)
- Google (Gemini — never caps max_tokens)
- DeepSeek (direct API)
- OpenRouter (routes to Llama, Qwen, Gemma, DeepSeek, others)
- Shared retry decorator with exponential backoff for rate-limit + transient errors

**Domain packs**
- Healthcare — Medicare prior authorization (CMS IRF) with full prompt structure
- Lending — ECOA consumer credit, zip-tier + employment-type proxies, gray-zone applicants
- Trading — Reg NMS market-making, 25% concentration + 3.0x leverage limits, three breakout variants

**Scoring**
- Generic + per-domain matched-pair violation predicates
- Two-stage compliance-masking classifier: keyword heuristic + LLM-based deep classification
- 25-signal Behavioral Precursor Signal extractor (hedging, justification, risk, uncertainty, modification rate, etc.)

**Deliverable renderers**
- Evaluation Card (markdown, JSON, HTML, PDF placeholder)
- Audit Report (markdown skeleton)
- Failure Cell Map (JSON)
- Precursor Profile (JSON with all 25 signals)
- Intervention Recommendations (markdown)
- Drift Baseline save + diff

**Integration**
- Supabase mode bridges to existing EVAV_Engine
- 5 compliance artifact templates (HIPAA, ECOA, SOC 2, EU AI Act, NIST AI RMF)

**Tests**
- 41 unit + edge-case tests, all passing
- GitHub Actions CI workflow

### Verified end-to-end

- Live API smoke test on DeepSeek V3 produced results replicating published findings:
  - Documentation-tier cliff: 0% strong → 100% moderate
  - Reward-pressure susceptibility: 100% (matches reference corpus 94%)
  - Anchor-strip improvement: 100% → 10%
  - Equalized control: 0% (instrument valid)

### Bug fixes during pre-release

- Fixed JSON parser failing on multi-line responses (rewrote with brace-balanced extraction + regex fallback)
- Fixed healthcare violation predicate missing `authorize_with_modification` as a violation form
- Fixed Unicode character crashes on Windows cp1252 console (✓/✗/×/Δ/═/→ replaced with ASCII equivalents)
- Fixed Rich console rendering with `legacy_windows` flag

### Documentation

- Complete `BENCHMARK_SPEC.md` (the instrument)
- Complete `PRODUCT_SPEC.md` (the four-tier product)
- 9-section sales onboarding worksheet
- 5 compliance artifact templates
- Evaluation Card template + JSON schema
- 5-gap remediation plan with runnable scripts
- `RESEARCH_INDEX.md` (single entry point to all 24 evav research folders)
- Lovable build pack (7-document spec for the oa.dev website)
- Backend API spec + FastAPI implementation

## Roadmap (Not Yet Released)

### [1.1.0] — Target Q3 2026

- Lending and trading public leaderboard data
- Tier 2 monitor (Gemma-based shadow agent)
- Customer audit dashboard
- Composite intervention (NORM + BIND stacked)
- Systematic temperature sweep on top failure cells
- Paraphrase robustness fixtures (F1, F2 paraphrases of HRW, HHP)

### [1.2.0] — Target Q4 2026

- Multi-turn / agentic evaluation
- Reasoning-trace stability for o1, o3, Claude thinking
- Self-correction failure forensic auto-detection
- Capability floor stress tests

### [2.0.0] — Target 2027

- Federated learning across customer engagements
- New domains: hiring, procurement, content moderation
- Tier 4 embedded guardrails (skill packages for Anthropic Skills, OpenAI Assistants, LangGraph)
- Adversarial prompt injection coverage (or partner integration)

[Unreleased]: https://github.com/operationalalignment/oa-bench/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/operationalalignment/oa-bench/releases/tag/v1.0.0
