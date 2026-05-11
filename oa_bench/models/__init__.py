"""Model adapter registry."""
from __future__ import annotations
from ..battery import ModelConfig
from ._base import ModelAdapter


def get_adapter(cfg: ModelConfig) -> ModelAdapter:
    """Return the adapter for the given provider, instantiated with the config."""
    p = cfg.provider.lower()
    if p == "anthropic":
        from .anthropic import AnthropicAdapter
        return AnthropicAdapter(cfg)
    if p == "openai":
        from .openai import OpenAIAdapter
        return OpenAIAdapter(cfg)
    if p == "google":
        from .google import GoogleAdapter
        return GoogleAdapter(cfg)
    if p in ("openrouter", "or"):
        from .openrouter import OpenRouterAdapter
        return OpenRouterAdapter(cfg)
    if p == "deepseek":
        from .deepseek import DeepSeekAdapter
        return DeepSeekAdapter(cfg)
    raise ValueError(
        f"Unknown provider: {cfg.provider}. "
        f"Supported: anthropic, openai, google, openrouter, deepseek"
    )
