"""EVAV operational alignment modules beyond the core matched-pair pressure battery.

Each module addresses a different failure-mode family:
  - hallucination       — factual incorrectness, citation faithfulness
  - adversarial         — prompt injection, jailbreaks (via NVIDIA garak wrapper)
  - refusal             — over-refusal under pressure (XSTest, OR-Bench)
  - privacy             — PII leakage (via Microsoft Presidio)
  - schema_fidelity     — output schema reliability under pressure
  - multi_turn          — matched-pair extended across N conversation turns

Pressure (the core matched-pair battery) is in oa_bench.runner / scoring.
"""
