"""Deterministic confidence policy for model probabilities."""

from __future__ import annotations

from .models import PolicyAction

ALLOW_MAX = 0.20
BLOCK_MIN = 0.80


def route_probability(
    risk_probability: float | None,
    *,
    allow_max: float = ALLOW_MAX,
    block_min: float = BLOCK_MIN,
) -> PolicyAction:
    """Map a risk probability to allow, review, or block."""
    if risk_probability is None:
        return "review"
    if not 0.0 <= allow_max < block_min <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= allow_max < block_min <= 1")
    if not 0.0 <= risk_probability <= 1.0:
        raise ValueError("risk_probability must be between 0 and 1")
    if risk_probability <= allow_max:
        return "allow"
    if risk_probability >= block_min:
        return "block"
    return "review"
