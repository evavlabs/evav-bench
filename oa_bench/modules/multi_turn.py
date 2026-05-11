"""Multi-turn / agentic deployment module.

Scaffold for testing models in multi-turn deployment patterns. Production
agentic systems aren't single-shot — they have memory, tool use, reasoning
loops. EVAV needs to extend matched-pair causal identification across turns.

This module:
  1. Runs N-turn conversations with the matched pair
  2. Tracks state evolution: does the violation surface at turn 1, 3, 7?
  3. Identifies WHICH TURN introduces the failure
  4. Measures cascading vs self-correcting behavior

For multi-agent (separate from multi-turn), the same matched-pair approach
applies but at each agent boundary. Requires customer to log per-agent IO.

STATUS: scaffold. The single-turn matched-pair pattern is proven; this extends
it. Real-world multi-turn semantics (tool calls, RAG retrieval, agent
orchestration) need customer-specific integration. Production version provides
this integration layer.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("oa_bench.multi_turn")


@dataclass
class TurnResult:
    turn_index: int
    base_response: dict
    twin_response: dict
    violated_at_this_turn: bool
    cumulative_state: str  # "compliant", "cascading_violation", "self_corrected"


@dataclass
class MultiTurnScenario:
    """One matched-pair scenario across N turns."""
    pair_id: str
    base_turns: list[str]  # user inputs at each turn for base
    twin_turns: list[str]  # user inputs at each turn for twin
    manipulated_variable: str
    base_value: str
    twin_value: str


def run_multi_turn_pair(
    model,
    scenario: MultiTurnScenario,
    domain,
    system_prompt: str,
    temperature: float = 0.3,
    seed: int | None = None,
) -> dict:
    """Run a multi-turn matched-pair scenario.

    For each turn:
      - Send the next user message
      - Capture the model response
      - Apply matched-pair scoring at the current turn
      - Track whether violation has surfaced

    Returns per-turn results + scenario-level summary.
    """
    base_history: list[dict] = [{"role": "system", "content": system_prompt}]
    twin_history: list[dict] = [{"role": "system", "content": system_prompt}]
    per_turn: list[TurnResult] = []
    cumulative_state = "compliant"

    for i, (base_user, twin_user) in enumerate(zip(scenario.base_turns, scenario.twin_turns)):
        base_history.append({"role": "user", "content": base_user})
        twin_history.append({"role": "user", "content": twin_user})

        # Note: model.invoke is single-turn — for true multi-turn we'd need a
        # chat-completions API path. The scaffold treats each turn as
        # independent for now; production version uses conversation history.
        try:
            base_resp = model.invoke(
                system_prompt=system_prompt + "\n" + _format_history(base_history[1:-1]),
                user_input=base_user,
                temperature=temperature, seed=seed,
            )
            twin_resp = model.invoke(
                system_prompt=system_prompt + "\n" + _format_history(twin_history[1:-1]),
                user_input=twin_user,
                temperature=temperature, seed=seed,
            )

            base_history.append({"role": "assistant", "content": base_resp.get("raw", {}).get("text", "")})
            twin_history.append({"role": "assistant", "content": twin_resp.get("raw", {}).get("text", "")})

            # Score this turn
            violated = domain.violation_predicate(base_resp, twin_resp, None) if hasattr(domain, "violation_predicate") else False

            if violated and cumulative_state == "compliant":
                cumulative_state = "cascading_violation"
            elif not violated and cumulative_state == "cascading_violation":
                cumulative_state = "self_corrected"

            per_turn.append(TurnResult(
                turn_index=i,
                base_response={"decision": base_resp.get("decision", ""), "text_preview": (base_resp.get("raw", {}).get("text", "") or "")[:200]},
                twin_response={"decision": twin_resp.get("decision", ""), "text_preview": (twin_resp.get("raw", {}).get("text", "") or "")[:200]},
                violated_at_this_turn=violated,
                cumulative_state=cumulative_state,
            ))
        except Exception as e:
            log.warning("turn %d failed: %s", i, e)

    # Scenario summary
    n_turns = len(per_turn)
    n_violations = sum(1 for t in per_turn if t.violated_at_this_turn)
    first_violation_turn = next((t.turn_index for t in per_turn if t.violated_at_this_turn), None)
    cascading = sum(1 for t in per_turn if t.cumulative_state == "cascading_violation")
    self_corrected = sum(1 for t in per_turn if t.cumulative_state == "self_corrected")

    return {
        "pair_id": scenario.pair_id,
        "n_turns": n_turns,
        "n_violations": n_violations,
        "violation_rate_pct": round(n_violations / max(1, n_turns) * 100, 1),
        "first_violation_turn": first_violation_turn,
        "cascading_violation_turns": cascading,
        "self_corrected_turns": self_corrected,
        "final_state": per_turn[-1].cumulative_state if per_turn else "unknown",
        "per_turn": [
            {
                "turn": t.turn_index,
                "violated": t.violated_at_this_turn,
                "state": t.cumulative_state,
                "base_decision": t.base_response["decision"],
                "twin_decision": t.twin_response["decision"],
            }
            for t in per_turn
        ],
    }


def _format_history(history: list[dict]) -> str:
    """Render history as text for models that don't accept message arrays."""
    parts = []
    for msg in history:
        parts.append(f"[{msg['role'].upper()}] {msg['content']}")
    return "\n".join(parts)


def aggregate_multi_turn_results(scenario_results: list[dict]) -> dict:
    """Aggregate across many multi-turn scenarios in a cell."""
    if not scenario_results:
        return {"error": "no scenarios"}

    n = len(scenario_results)
    n_with_violation = sum(1 for r in scenario_results if r["n_violations"] > 0)
    cascading_count = sum(1 for r in scenario_results if r["final_state"] == "cascading_violation")
    corrected_count = sum(1 for r in scenario_results if r["final_state"] == "self_corrected")

    first_violation_turns = [r["first_violation_turn"] for r in scenario_results if r["first_violation_turn"] is not None]
    avg_first_violation_turn = (sum(first_violation_turns) / len(first_violation_turns)) if first_violation_turns else None

    return {
        "n_scenarios": n,
        "scenarios_with_violation_pct": round(n_with_violation / n * 100, 1),
        "scenarios_cascading_pct": round(cascading_count / n * 100, 1),
        "scenarios_self_corrected_pct": round(corrected_count / n * 100, 1),
        "avg_first_violation_turn": round(avg_first_violation_turn, 1) if avg_first_violation_turn else None,
        "implication": (
            "Cascading violations dominate — model entrenches after first violation. "
            "Per-turn intervention required."
            if cascading_count > corrected_count else
            "Self-correction common — model recovers after initial violation. "
            "End-state monitoring may suffice."
        ),
    }
