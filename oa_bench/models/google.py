"""Google Gemini adapter with retry + rate-limit handling.

Memory: never cap max_tokens on Gemini.
"""

from __future__ import annotations
import os
import time
from typing import Any, Optional

from ._base import ModelAdapter
from ._retry import with_retry
from ..battery import ModelConfig
from .anthropic import parse_structured_output


class GoogleAdapter(ModelAdapter):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            ) from e
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment.")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(cfg.name)

    def invoke(
        self,
        system_prompt: str,
        user_input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._invoke_with_retry(system_prompt, user_input, temperature, seed)

    @with_retry(max_retries=6, base_delay=2.0, max_delay=90.0)
    def _invoke_with_retry(self, system_prompt, user_input, temperature, seed):
        self.call_count += 1
        t = time.time()
        try:
            content = f"{system_prompt}\n\n---\n\n{user_input}"
            generation_config = {
                "temperature": temperature if temperature is not None else self.cfg.temperature,
                # Per memory: do NOT cap max_tokens on Gemini.
            }
            resp = self._model.generate_content(content, generation_config=generation_config)
        except Exception:
            self.error_count += 1
            self.retry_count += 1
            raise

        latency_ms = int((time.time() - t) * 1000)
        text = ""
        try:
            text = resp.text if hasattr(resp, "text") else ""
        except Exception:
            # Some safety-block responses don't expose .text
            text = ""
        decision, reasoning = parse_structured_output(text)

        tokens = 0
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            tokens = (
                getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
            ) + (
                getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
            )
        self.total_tokens += tokens

        return {
            "decision": decision,
            "reasoning": reasoning,
            "raw": {"text": text},
            "tokens": tokens,
            "latency_ms": latency_ms,
        }
