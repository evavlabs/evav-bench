"""Edge-case tests for production hardening.

Covers malformed inputs, empty responses, missing API keys, and other
failure modes the smoke tests don't exercise.
"""

import json
import os
from pathlib import Path

import pytest

from oa_bench.battery import BatteryConfig

EXAMPLES = Path(__file__).parent.parent / "examples"


# ────────────────────────────────────────────────────────────────────────────
# Config robustness
# ────────────────────────────────────────────────────────────────────────────

def test_missing_required_field_raises():
    """Battery config without required fields should raise validation error."""
    with pytest.raises(Exception) as exc:
        BatteryConfig(**{})
    assert "engagement_id" in str(exc.value) or "Field required" in str(exc.value)


def test_invalid_groups_rejected():
    """Scope with non-existent groups must be rejected."""
    base = json.loads((EXAMPLES / "battery.healthcare.example.json").read_text())
    base["scope"]["groups"] = ["A", "Z"]  # Z is invalid
    with pytest.raises(Exception) as exc:
        BatteryConfig(**base)
    assert "Z" in str(exc.value) or "Invalid groups" in str(exc.value)


def test_temperature_out_of_range_rejected():
    """Temperature must be in [0, 2]."""
    base = json.loads((EXAMPLES / "battery.healthcare.example.json").read_text())
    base["model"]["temperature"] = 5.0
    with pytest.raises(Exception):
        BatteryConfig(**base)


def test_n_per_cell_lower_bound():
    """N below 10 should be rejected (matched-pair design requires minimum statistical power)."""
    base = json.loads((EXAMPLES / "battery.healthcare.example.json").read_text())
    base["scope"]["n_per_cell"] = 5
    with pytest.raises(Exception):
        BatteryConfig(**base)


# ────────────────────────────────────────────────────────────────────────────
# Response parsing
# ────────────────────────────────────────────────────────────────────────────

def test_parse_handles_empty_string():
    from oa_bench.models.anthropic import parse_structured_output
    d, r = parse_structured_output("")
    assert d == ""
    assert r == ""


def test_parse_handles_json_with_markdown_fence():
    from oa_bench.models.anthropic import parse_structured_output
    text = '```json\n{"decision": "approve", "reasoning": "criteria met"}\n```'
    d, r = parse_structured_output(text)
    assert "approve" in d.lower()
    assert "criteria met" in r.lower()


def test_parse_handles_trading_schema():
    """Trading uses 'action' + allocation fields, not 'decision'."""
    from oa_bench.models.anthropic import parse_structured_output
    text = json.dumps({
        "action": "rebalance", "btc_pct": 40, "eth_pct": 25,
        "sol_pct": 20, "bnb_pct": 15, "leverage": 1.0,
        "reasoning": "balanced allocation", "risk_compliance": "compliant",
    })
    d, r = parse_structured_output(text)
    # Should pull allocations into decision string
    assert "btc_pct=40" in d or "rebalance" in d.lower()


def test_parse_handles_freeform_text():
    """Non-JSON response should fall back to first-line decision + rest."""
    from oa_bench.models.anthropic import parse_structured_output
    text = "approve\n\nThe applicant meets all the criteria."
    d, r = parse_structured_output(text)
    assert d.lower() == "approve"
    assert "criteria" in r.lower()


def test_parse_handles_malformed_json_falls_back():
    """Truncated JSON should fall back to text parsing without crashing."""
    from oa_bench.models.anthropic import parse_structured_output
    text = '{"decision": "deny", "reasoning": "incom'
    d, r = parse_structured_output(text)
    # Doesn't crash; returns something
    assert isinstance(d, str) and isinstance(r, str)


# ────────────────────────────────────────────────────────────────────────────
# Adapter behavior without API keys
# ────────────────────────────────────────────────────────────────────────────

def test_anthropic_adapter_without_api_key_raises_clean_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from oa_bench.models.anthropic import AnthropicAdapter
    from oa_bench.battery import ModelConfig
    cfg = ModelConfig(provider="anthropic", name="claude-sonnet-4-6")
    with pytest.raises(ValueError) as exc:
        AnthropicAdapter(cfg)
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_openai_adapter_without_api_key_raises_clean_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from oa_bench.models.openai import OpenAIAdapter
    from oa_bench.battery import ModelConfig
    cfg = ModelConfig(provider="openai", name="gpt-4o")
    with pytest.raises(ValueError) as exc:
        OpenAIAdapter(cfg)
    assert "OPENAI_API_KEY" in str(exc.value)


# ────────────────────────────────────────────────────────────────────────────
# Domain pack robustness
# ────────────────────────────────────────────────────────────────────────────

def test_healthcare_pack_generates_distinct_pairs():
    """Generated base/twin should differ on the manipulated variable."""
    from oa_bench.domains.healthcare import HealthcarePack
    from oa_bench.battery import BatteryConfig, Cell
    cfg = BatteryConfig(**json.loads((EXAMPLES / "battery.healthcare.example.json").read_text()))
    cell = Cell(
        cell_id="B1.X.+A.P0.s42.t03", group="B", name="Reward",
        description="test", pressure="RW", doc_tier="X", anchor="+A",
        phrasing="P0", seed=42, temperature=0.3,
    )
    pack = HealthcarePack()
    pairs = list(pack.generate_pairs(n=3, seed=42, cell=cell, cfg=cfg))
    assert len(pairs) == 3
    for base, twin, meta in pairs:
        # Base and twin must differ on the manipulated variable
        assert base != twin
        assert "pair_id" in meta


def test_lending_pack_generates_matched_pairs_with_proxy():
    from oa_bench.domains.lending import LendingPack
    from oa_bench.battery import BatteryConfig, Cell
    cfg = BatteryConfig(**json.loads((EXAMPLES / "battery.lending.example.json").read_text()))
    cell = Cell(
        cell_id="B1.X.+A.P0.s42.t03", group="B", name="Reward",
        description="test", pressure="RW", doc_tier="X", anchor="+A",
        phrasing="P0", seed=42, temperature=0.3,
    )
    pack = LendingPack()
    pairs = list(pack.generate_pairs(n=3, seed=42, cell=cell, cfg=cfg))
    assert len(pairs) == 3
    for base, twin, meta in pairs:
        # Lending: same credit profile, different zip tier
        assert meta["manipulated_variable"] in ("zip_tier", "employment_type")
        assert "Zip Tier:                   1" in base or "Zip Tier:                   2" in base or "Zip Tier:                   3" in base
        assert "Zip Tier:" in twin


def test_trading_pack_generates_market_regime_pairs():
    from oa_bench.domains.trading import TradingPack
    from oa_bench.battery import BatteryConfig, Cell
    cfg = BatteryConfig(**json.loads((EXAMPLES / "battery.trading.example.json").read_text()))
    cell = Cell(
        cell_id="B1.X.+A.P0.s42.t03", group="B", name="Reward",
        description="test", pressure="RW", doc_tier="X", anchor="+A",
        phrasing="P0", seed=42, temperature=0.3,
    )
    pack = TradingPack()
    pairs = list(pack.generate_pairs(n=3, seed=42, cell=cell, cfg=cfg))
    assert len(pairs) == 3
    for base, twin, meta in pairs:
        assert meta["base_value"] == "flat"
        assert meta["twin_value"].startswith("breakout_")
        # Both should contain the SYSTEM prompt embedded for trading
        assert "[SYSTEM]" in base
        assert "[/SYSTEM]" in twin


def test_seed_determinism_healthcare():
    """Same seed → identical pairs across runs."""
    from oa_bench.domains.healthcare import HealthcarePack
    from oa_bench.battery import BatteryConfig, Cell
    cfg = BatteryConfig(**json.loads((EXAMPLES / "battery.healthcare.example.json").read_text()))
    cell = Cell(
        cell_id="B1.X.+A.P0.s42.t03", group="B", name="Reward",
        description="test", pressure="RW", doc_tier="X", anchor="+A",
        phrasing="P0", seed=42, temperature=0.3,
    )
    pack = HealthcarePack()
    p1 = list(pack.generate_pairs(n=5, seed=42, cell=cell, cfg=cfg))
    p2 = list(pack.generate_pairs(n=5, seed=42, cell=cell, cfg=cfg))
    # Same seed must produce identical scenarios
    for (a, _, _), (b, _, _) in zip(p1, p2):
        assert a == b


def test_seed_determinism_lending():
    from oa_bench.domains.lending import LendingPack
    from oa_bench.battery import BatteryConfig, Cell
    cfg = BatteryConfig(**json.loads((EXAMPLES / "battery.lending.example.json").read_text()))
    cell = Cell(
        cell_id="B1.X.+A.P0.s42.t03", group="B", name="Reward",
        description="test", pressure="RW", doc_tier="X", anchor="+A",
        phrasing="P0", seed=42, temperature=0.3,
    )
    pack = LendingPack()
    p1 = list(pack.generate_pairs(n=5, seed=42, cell=cell, cfg=cfg))
    p2 = list(pack.generate_pairs(n=5, seed=42, cell=cell, cfg=cfg))
    for (a, _, _), (b, _, _) in zip(p1, p2):
        assert a == b


# ────────────────────────────────────────────────────────────────────────────
# Trading domain — violation detection
# ────────────────────────────────────────────────────────────────────────────

def test_trading_violation_concentration_breach():
    """Trading: base 25%-balanced + twin 60% in one asset = violation."""
    from oa_bench.domains.trading import TradingPack
    base_resp = {"raw": {"text": json.dumps({
        "action": "rebalance", "btc_pct": 25, "eth_pct": 25,
        "sol_pct": 25, "bnb_pct": 25, "leverage": 1.0,
        "reasoning": "balanced", "risk_compliance": "compliant",
    })}}
    twin_resp = {"raw": {"text": json.dumps({
        "action": "rebalance", "btc_pct": 60, "eth_pct": 15,
        "sol_pct": 15, "bnb_pct": 10, "leverage": 1.0,
        "reasoning": "BTC has momentum", "risk_compliance": "compliant",
    })}}
    pack = TradingPack()
    assert pack.violation_predicate(base_resp, twin_resp, None)


def test_trading_no_violation_both_compliant():
    from oa_bench.domains.trading import TradingPack
    base_resp = {"raw": {"text": '{"btc_pct":25,"eth_pct":25,"sol_pct":25,"bnb_pct":25,"leverage":1.0}'}}
    twin_resp = {"raw": {"text": '{"btc_pct":25,"eth_pct":25,"sol_pct":25,"bnb_pct":25,"leverage":1.0}'}}
    pack = TradingPack()
    assert not pack.violation_predicate(base_resp, twin_resp, None)


def test_trading_leverage_violation_detected():
    from oa_bench.domains.trading import TradingPack
    base_resp = {"raw": {"text": '{"btc_pct":25,"eth_pct":25,"sol_pct":25,"bnb_pct":25,"leverage":1.0}'}}
    twin_resp = {"raw": {"text": '{"btc_pct":25,"eth_pct":25,"sol_pct":25,"bnb_pct":25,"leverage":5.0}'}}
    pack = TradingPack()
    assert pack.violation_predicate(base_resp, twin_resp, None)


# ────────────────────────────────────────────────────────────────────────────
# Lending domain — violation detection
# ────────────────────────────────────────────────────────────────────────────

def test_lending_violation_approve_vs_deny():
    from oa_bench.domains.lending import LendingPack
    base_resp = {"decision": "approve", "raw": {"text": '{"action":"approve","reasoning":"creditworthy borrower"}'}}
    twin_resp = {"decision": "deny", "raw": {"text": '{"action":"deny","reasoning":"elevated risk profile"}'}}
    pack = LendingPack()
    assert pack.violation_predicate(base_resp, twin_resp, None)


def test_lending_no_violation_both_approve():
    from oa_bench.domains.lending import LendingPack
    base_resp = {"decision": "approve", "raw": {"text": '{"action":"approve"}'}}
    twin_resp = {"decision": "approve_with_conditions", "raw": {"text": '{"action":"approve_with_conditions"}'}}
    pack = LendingPack()
    # Both approved → not a violation
    assert not pack.violation_predicate(base_resp, twin_resp, None)


# ────────────────────────────────────────────────────────────────────────────
# Retry decorator behavior
# ────────────────────────────────────────────────────────────────────────────

def test_retry_succeeds_after_transient_failures():
    """Retry decorator should eventually succeed if call recovers."""
    from oa_bench.models._retry import with_retry

    attempts = {"n": 0}

    class FakeRateLimit(Exception):
        pass

    @with_retry(max_retries=3, base_delay=0.01, max_delay=0.05)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FakeRateLimit("rate_limit")
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 3


def test_retry_gives_up_on_permanent_error():
    from oa_bench.models._retry import with_retry

    class BadRequest(Exception):
        status_code = 400

    @with_retry(max_retries=3, base_delay=0.01)
    def always_bad():
        raise BadRequest("invalid input")

    with pytest.raises(BadRequest):
        always_bad()


def test_retry_exhausts_then_raises():
    from oa_bench.models._retry import with_retry

    class Rate(Exception):
        pass

    @with_retry(max_retries=2, base_delay=0.01, max_delay=0.05)
    def always_rate():
        raise Rate("rate_limit")

    with pytest.raises(Rate):
        always_rate()
