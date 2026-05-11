"""Anthropic Claude adapter with retry + rate-limit handling."""

from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Optional

from ._base import ModelAdapter
from ._retry import with_retry
from ..battery import ModelConfig


class AnthropicAdapter(ModelAdapter):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError("anthropic SDK not installed. Run: pip install anthropic") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment.")
        self._client = Anthropic(api_key=api_key)

    def invoke(
        self,
        system_prompt: str,
        user_input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._invoke_with_retry(system_prompt, user_input, temperature, max_tokens, seed)

    @with_retry(max_retries=6, base_delay=1.0, max_delay=60.0)
    def _invoke_with_retry(self, system_prompt, user_input, temperature, max_tokens, seed):
        self.call_count += 1
        t = time.time()
        try:
            resp = self._client.messages.create(
                model=self.cfg.name,
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
                temperature=temperature if temperature is not None else self.cfg.temperature,
                max_tokens=max_tokens or self.cfg.max_tokens,
            )
        except Exception:
            self.error_count += 1
            self.retry_count += 1
            raise

        latency_ms = int((time.time() - t) * 1000)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        decision, reasoning = parse_structured_output(text)

        usage = getattr(resp, "usage", None)
        tokens = 0
        if usage:
            tokens = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
        self.total_tokens += tokens

        return {
            "decision": decision,
            "reasoning": reasoning,
            "raw": {"text": text, "stop_reason": getattr(resp, "stop_reason", "")},
            "tokens": tokens,
            "latency_ms": latency_ms,
        }


def parse_structured_output(text: str) -> tuple[str, str]:
    """Best-effort extraction of decision + reasoning from free-form, fenced, or embedded JSON."""
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip(),
                     flags=re.IGNORECASE | re.MULTILINE)

    # Try clean JSON parse
    for candidate in (cleaned, text):
        try:
            obj = json.loads(candidate)
            return _extract_from_dict(obj)
        except (json.JSONDecodeError, ValueError):
            pass

    # Find balanced {…} using brace-counting (handles nested + multi-line JSON
    # robustly, unlike a regex-only approach)
    obj = _extract_balanced_json(cleaned)
    if obj is not None:
        return _extract_from_dict(obj)

    # Try markdown sections
    dm = re.search(r"##\s*Decision\s*\n+(.+?)(?=\n##|\Z)", text, re.DOTALL | re.IGNORECASE)
    rm = re.search(r"##\s*Reasoning\s*\n+(.+?)(?=\n##|\Z)", text, re.DOTALL | re.IGNORECASE)
    if dm or rm:
        return (
            (dm.group(1).strip().split("\n")[0] if dm else ""),
            (rm.group(1).strip() if rm else ""),
        )

    # Look for `"decision": "..."` and `"reasoning": "..."` patterns directly
    # (handles broken JSON where the structure is preserved but braces are off)
    decision_m = re.search(r'"(?:decision|action)"\s*:\s*"([^"]+)"', text)
    reasoning_m = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
    if decision_m or reasoning_m:
        return (
            decision_m.group(1).strip() if decision_m else "",
            (reasoning_m.group(1).encode("utf-8").decode("unicode_escape", errors="replace")
             if reasoning_m else ""),
        )

    # Fallback: first line as decision, rest as reasoning
    lines = text.strip().split("\n", 1)
    return (lines[0].strip(), lines[1].strip() if len(lines) > 1 else "")


def _extract_balanced_json(text: str) -> dict | None:
    """Find the first balanced {…} block and parse it as JSON. Robust to multi-line + nested."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        break  # try next opening brace
                    break
        start = text.find("{", start + 1)
    return None


def _extract_from_dict(obj: dict) -> tuple[str, str]:
    """Pull decision + reasoning from a parsed JSON object. Handles all known schemas."""
    if not isinstance(obj, dict):
        return str(obj), ""
    # Direct fields
    decision = obj.get("decision") or obj.get("action") or ""
    reasoning = obj.get("reasoning") or obj.get("explanation") or ""
    # Trading-style schema: action=rebalance + allocations
    if "btc_pct" in obj or "rebalance" in str(decision).lower():
        allocs = []
        for k in ("btc_pct", "eth_pct", "sol_pct", "bnb_pct"):
            if k in obj:
                allocs.append(f"{k}={obj[k]}")
        if "leverage" in obj:
            allocs.append(f"leverage={obj['leverage']}")
        decision = f"rebalance ({', '.join(allocs)})"
    return str(decision).strip(), str(reasoning).strip()


# Backwards-compat alias (used by openai.py / openrouter.py)
_parse_structured_output = parse_structured_output
