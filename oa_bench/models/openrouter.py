"""OpenRouter adapter — routes to DeepSeek, Qwen, Llama, Gemma, etc.

Uses OpenAI SDK pointed at OpenRouter base URL.
Memory: PYTHONIOENCODING=utf-8 required for OpenRouter on Windows.
"""

from __future__ import annotations
import os
import time
from typing import Any, Optional

from ._base import ModelAdapter
from ._retry import with_retry
from ..battery import ModelConfig
from .anthropic import parse_structured_output


class OpenRouterAdapter(ModelAdapter):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai SDK required. Run: pip install openai") from e
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment.")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://oa.dev", "X-Title": "OA Eval Battery"},
        )

    def invoke(
        self,
        system_prompt: str,
        user_input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._invoke_with_retry(system_prompt, user_input, temperature, max_tokens, seed)

    @with_retry(max_retries=6, base_delay=1.0, max_delay=90.0)
    def _invoke_with_retry(self, system_prompt, user_input, temperature, max_tokens, seed):
        self.call_count += 1
        t = time.time()
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=temperature if temperature is not None else self.cfg.temperature,
                max_tokens=max_tokens or self.cfg.max_tokens,
            )
        except Exception:
            self.error_count += 1
            self.retry_count += 1
            raise

        latency_ms = int((time.time() - t) * 1000)
        text = resp.choices[0].message.content or ""
        decision, reasoning = parse_structured_output(text)
        tokens = 0
        if resp.usage:
            tokens = (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0)
        self.total_tokens += tokens

        return {
            "decision": decision,
            "reasoning": reasoning,
            "raw": {"text": text},
            "tokens": tokens,
            "latency_ms": latency_ms,
        }
