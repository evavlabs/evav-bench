# EVAV Benchmark v1.0 — Honest Limitations

The EVAV Benchmark is a research instrument. It is useful for cross-model comparison, deployment-readiness sanity checks, and surfacing failure modes that standard evals miss. It is **not** a substitute for testing your specific deployment.

This document enumerates what the benchmark does **not** cover, what assumptions are baked in, and what kinds of conclusions you should and should not draw from EVAV Scores.

---

## 1. Scope limits

### 1.1 General benchmark, not configuration-specific
EVAV Scores measure behavior on a general benchmark with deployment-realistic conditions. Your actual deployment has:
- Its own system prompt
- Its own rules and policies
- Its own pressure surface (which inputs and contexts your users actually produce)
- Its own tool surface

A model that scores 73 on EVAV may fail catastrophically on your specific configuration. A model that scores 50 may be fine for your specific use case. **Pre-deployment testing of your specific configuration is recommended before relying on these scores for go/no-go decisions.**

### 1.2 English-only
Methodology not validated on non-English. Findings from the 8-model reference corpus apply to English prompts and English-language regulatory contexts.

### 1.3 Single modality
Text-in / text-out only. Multimodal models are evaluated on their text behavior; vision/audio modalities not in v1.0.

### 1.4 Pressure battery is domain-anchored
The matched-pair pressure battery (module 1) uses healthcare prior authorization as the reference corpus. Lending and trading domain packs are in development. Other modules (hallucination, adversarial, refusal, privacy, schema, bias, tool safety, reasoning trace) are domain-agnostic.

### 1.5 Reasoning trace module applies to reasoning models only
For non-reasoning models, the reasoning trace module is omitted from the EVAV Score; the mean is computed over the modules that apply.

---

## 2. What's not measured

### 2.1 Multi-turn conversations
v1.0 is single-turn. Long agentic loops, follow-ups, and conversational state are not captured by the pressure battery. Tool safety module is single-call.

### 2.2 Production load behavior
Latency, cost, throughput, rate limit handling, retries, partial responses, streaming behavior — not measured. EVAV asks whether the model **decides** safely; not whether it scales.

### 2.3 Capability degradation under load
The benchmark runs at typical request volumes. Behavior at sustained high QPS, under context-window pressure, or after long conversations is not in scope for v1.0.

### 2.4 Custom safety filters / output classifiers
If your deployment wraps the model in additional moderation layers, the benchmark measures the raw model — not the wrapped system. To evaluate your full stack, run the benchmark through your production gateway.

### 2.5 Adversarial defense against expert humans
The adversarial module (NVIDIA garak + EVAV) tests against established attack categories and the open-source attack library. It does **not** simulate adaptive, sophisticated human red-teamers. For that, engage specialist services.

### 2.6 Long-horizon drift
The benchmark is a snapshot. Model behavior can change between releases. EVAV provides drift-detection tooling (`evav drift-diff`) but the v1.0 leaderboard does not yet show per-version trends.

---

## 3. Methodological assumptions

### 3.1 The matched-pair causal claim
The matched-pair pressure battery (module 1) follows the audit-study lineage (Bertrand & Mullainathan 2004). The causal claim is that **observed differential treatment between base and twin cannot be explained by sample composition** — because the two scenarios are identical except on the manipulated variable.

This claim holds when:
- The PRNG seed produces truly identical applicant pairs across base and twin (verified: 100% applicant identity check)
- The manipulated variable is the only difference (verified by construction)
- Cross-seed analysis shows that violations are scenario-triggered, not random (verified for the reference corpus)

It does **not** claim that the model "intends" to discriminate — only that, under the construction, the variable in question causally affects the decision.

### 3.2 Failure rate inversion
Each module score is `100 - failure_rate_pct`, clamped to [0, 100]. This treats the worst possible failure rate as a score of 0. It is intentionally lossy — it does not preserve information about whether a model fails 50% of the time on 1% of inputs vs 1% of the time on all inputs.

### 3.3 Arithmetic mean across modules
The EVAV Score treats all 10 modules with equal weight. This is a research-framing choice. A user with strong opinions about which module matters most (e.g., privacy is paramount for healthcare deployments) should consult the per-module breakdown, not the headline.

### 3.4 Weakest-module reporting
The weakest module is reported alongside the headline. Two models can share an EVAV Score of 65 but differ on which module fails. The card shows both.

---

## 4. Statistical caveats

### 4.1 N per cell
Default n_per_cell is 100; minimal is 30; full audit is 100 with 4 seeds. Cells with n=30 have wider confidence intervals than cells with n=100. Statistical significance is computed and reported per cell for paid audits; the public leaderboard shows point estimates.

### 4.2 Temperature
Default temperature is 0.3. Behavior at temperature 0 (greedy) and at higher temperatures (0.7+) can differ. The full audit includes a temperature sweep; the public leaderboard reports the default-temperature score.

### 4.3 Provider-side variation
Identical API calls to the same model can return slightly different outputs (load balancing, version pinning, prompt caching). The cross-seed swing metric (reported on the card) captures provider-side variance. Findings are robust where cross-seed swing is < 5pp; less robust where it exceeds 10pp.

### 4.4 Selective failure modes
The public benchmark covers the 10 named modules. There are failure modes outside these 10 that a paid audit may surface (e.g., self-exfiltration, alignment-tax leakage, deceptive compliance). The benchmark is a floor for confidence, not a ceiling.

---

## 5. Open-source module wrapping

Modules 3-9 (hallucination, adversarial, refusal, privacy, schema, bias, tool safety) wrap established open-source evaluation tools:
- garak (NVIDIA) — adversarial
- TruthfulQA + RAGAS — hallucination
- XSTest — refusal calibration
- Microsoft Presidio — PII / privacy
- Fairlearn — bias / disparate impact
- Inspect AI (UK AISI) — tool safety

EVAV does **not** claim originality on these methodologies. EVAV's contribution is:
1. The matched-pair pressure battery (module 1) and compliance-masking detection (module 2)
2. The composite EVAV Score that integrates all 10 modules into a single comparable headline
3. Unified pipeline, reporting, and version-locked deployment

For module-specific methodological depth, refer to each wrapped tool's documentation.

---

## 6. What you can and cannot conclude

### You CAN conclude:
- "Model A scores higher than Model B on the EVAV Benchmark v1.0" (cross-model comparison is the benchmark's primary purpose)
- "Model X's weakest module is Y" (the weakest module is reported separately)
- "Under matched-pair pressure of type Z, Model X violates at N% in healthcare scenarios" (per-cell breakdowns support this)
- "Configuration-specific testing is needed before deployment" (always true)

### You CANNOT conclude:
- "Model A is safe for my deployment" (we don't know your deployment)
- "Model B is unsafe in general" (the benchmark is a sample of failure modes, not a complete enumeration)
- "Model A is 50% safer than Model B" (the scoring is ordinal, not ratio)
- "The EVAV Score predicts production incident rates" (this is not yet validated; we are building the corpus to validate it)

---

## 7. Where to learn more

- Paper: [evav.ai/research](https://evav.ai/research) — NeurIPS 2026 Datasets and Benchmarks Track submission
- Methodology: `products/benchmark/BENCHMARK_SPEC.md`
- Stability contract: `products/benchmark/STABILITY_CONTRACT.md`
- Per-module documentation: `products/cli/oa_bench/modules/`
- Reproducibility guarantee: `products/benchmark/REPRODUCIBILITY.md`
