"""Domain pack registry."""
from __future__ import annotations
from ._base import DomainPack


def get_domain_pack(domain: str) -> DomainPack:
    """Return the DomainPack for a domain identifier."""
    d = domain.lower().replace("_", "-").replace(" ", "-")
    if d in ("healthcare", "health", "prior-auth", "medicare-prior-auth"):
        from .healthcare import HealthcarePack
        return HealthcarePack()
    if d in ("lending", "credit", "consumer-lending", "mortgage"):
        from .lending import LendingPack
        return LendingPack()
    if d in ("trading", "market-making", "finance-trading"):
        from .trading import TradingPack
        return TradingPack()
    raise ValueError(
        f"Unknown domain: {domain}. Supported reference packs: "
        f"healthcare, lending, trading. New domains require building a custom DomainPack."
    )
