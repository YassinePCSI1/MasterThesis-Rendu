"""Cost functions for producers."""
from __future__ import annotations

from typing import Dict

from .config import AdjustmentCostParams


def variable_cost(q: float, marginal_cost: float) -> float:
    """Constant marginal cost: c * q."""

    return marginal_cost * q


def adjustment_cost(q_now: float, q_prev: float, params: AdjustmentCostParams) -> float:
    """Quadratic adjustment cost for changing output.

    Cost = k/2 * (q_t - q_{t-1})^2 when enabled.
    """

    if not params.enabled:
        return 0.0
    return 0.5 * params.k * (q_now - q_prev) ** 2


def total_adjustment_cost(outputs: Dict[str, float], prev_outputs: Dict[str, float],
                          params: AdjustmentCostParams) -> float:
    """Aggregate adjustment cost across all players."""

    return sum(adjustment_cost(outputs[p], prev_outputs[p], params) for p in outputs)
