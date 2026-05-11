"""Abstract base class for model adapters.

Any new provider implements .invoke() and returns a dict with at minimum:
    {
        "decision": str,
        "reasoning": str,
        "raw": dict,
        "tokens": int,
        "latency_ms": int,
    }

Subclasses get free retry/rate-limit handling by wrapping their inner call
with @with_retry from oa_bench.models._retry.
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from ..battery import ModelConfig

log = logging.getLogger("oa_bench.models")


class ModelAdapter(ABC):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.call_count = 0
        self.retry_count = 0
        self.error_count = 0
        self.total_tokens = 0

    @abstractmethod
    def invoke(
        self,
        system_prompt: str,
        user_input: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def stats(self) -> dict:
        return {
            "call_count": self.call_count,
            "retry_count": self.retry_count,
            "error_count": self.error_count,
            "total_tokens": self.total_tokens,
        }
