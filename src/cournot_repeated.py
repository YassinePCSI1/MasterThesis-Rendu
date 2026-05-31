"""Repeated Cournot game dynamics with optional adjustment costs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import AdjustmentCostParams, CapacityParams, CostParams, DemandParams, RepeatedGameParams
from .demand import price_from_quantity


@dataclass
class RepeatedOutcome:
    quantities: List[Dict[str, float]]
    prices: List[float]
    total_quantities: List[float]
    profits: List[Dict[str, float]]


def _cost_map(costs: CostParams) -> Dict[str, float]:
    return {"US": costs.c_us, "OPEC": costs.c_opec, "RUS": costs.c_rus}


def _cap_map(caps: CapacityParams) -> Dict[str, float]:
    return {"US": caps.cap_us, "OPEC": caps.cap_opec, "RUS": caps.cap_rus}


def best_response(q_others: float,
                  q_prev: float,
                  demand: DemandParams,
                  c_i: float,
                  caps: CapacityParams,
                  player: str,
                  adjustment: AdjustmentCostParams) -> float:
    """Best response with optional adjustment cost and capacity constraints."""

    k = adjustment.k if adjustment.enabled else 0.0
    denom = 2 * demand.b + k
    numer = demand.a - c_i - demand.b * q_others + k * q_prev
    q_i = numer / denom

    if caps.enabled:
        cap_map = _cap_map(caps)
        q_i = float(np.clip(q_i, 0.0, cap_map[player]))
    else:
        q_i = max(0.0, float(q_i))
    return q_i


def simulate_myopic(players: List[str],
                     demand: DemandParams,
                     costs: CostParams,
                     capacities: CapacityParams,
                     repeated: RepeatedGameParams,
                     adjustment: AdjustmentCostParams,
                     initial_outputs: Optional[Dict[str, float]] = None,
                     inertia: Optional[float] = None) -> RepeatedOutcome:
    """Simulate repeated game with myopic best responses and optional inertia."""

    inertia = repeated.inertia if inertia is None else inertia
    cost_map = _cost_map(costs)
    outputs = initial_outputs or {p: 0.0 for p in players}

    quantities: List[Dict[str, float]] = []
    prices: List[float] = []
    total_quantities: List[float] = []
    profits: List[Dict[str, float]] = []

    for _ in range(repeated.T):
        new_outputs: Dict[str, float] = {}
        for p in players:
            q_others = sum(outputs[k] for k in players if k != p)
            br = best_response(q_others, outputs[p], demand, cost_map[p],
                               capacities, p, adjustment)
            new_outputs[p] = (1 - inertia) * outputs[p] + inertia * br

        prev_outputs = outputs
        outputs = new_outputs
        Q = sum(outputs.values())
        price = price_from_quantity(Q, demand)
        period_profits = {}
        for p in players:
            profit = (price - cost_map[p]) * outputs[p]
            if adjustment.enabled:
                profit -= 0.5 * adjustment.k * (outputs[p] - prev_outputs[p]) ** 2
            period_profits[p] = profit

        quantities.append(outputs.copy())
        prices.append(price)
        total_quantities.append(Q)
        profits.append(period_profits)

    assert len(quantities) == repeated.T
    assert len(prices) == repeated.T
    return RepeatedOutcome(
        quantities=quantities,
        prices=prices,
        total_quantities=total_quantities,
        profits=profits,
    )


def simulate_tacit_punishment(players: List[str],
                              demand: DemandParams,
                              costs: CostParams,
                              capacities: CapacityParams,
                              repeated: RepeatedGameParams,
                              cooperative_outputs: Dict[str, float],
                              punishment_outputs: Dict[str, float],
                              deviation_period: Optional[int] = None,
                              deviator: Optional[str] = None) -> RepeatedOutcome:
    """Simulate cooperative path with deviation detection and punishment."""

    outputs = cooperative_outputs.copy()
    quantities: List[Dict[str, float]] = []
    prices: List[float] = []
    total_quantities: List[float] = []
    profits: List[Dict[str, float]] = []

    cost_map = _cost_map(costs)
    punishment_remaining = 0
    permanent_punishment = False

    for t in range(repeated.T):
        # Track whether this period starts in the cooperative phase.
        # Deviation detection must only fire during cooperative periods;
        # checking at the end of a punishment period would compare Nash
        # outputs against cartel quotas and spuriously re-trigger punishment.
        in_cooperative_phase = (not permanent_punishment) and (punishment_remaining == 0)

        if permanent_punishment or punishment_remaining > 0:
            outputs = punishment_outputs.copy()
            if repeated.grim_trigger:
                permanent_punishment = True
            else:
                punishment_remaining = max(0, punishment_remaining - 1)
        else:
            outputs = cooperative_outputs.copy()

        if t == deviation_period and deviator is not None and not permanent_punishment:
            outputs = outputs.copy()
            q_others = sum(outputs[k] for k in players if k != deviator)
            outputs[deviator] = best_response(
                q_others,
                outputs[deviator],
                demand,
                cost_map[deviator],
                capacities,
                deviator,
                AdjustmentCostParams(enabled=False, k=0.0),
            )

        # Only detect deviations when we were in the cooperative phase at the
        # start of this period (not when transitioning out of punishment).
        if in_cooperative_phase and not permanent_punishment:
            if any(
                abs(outputs[p] - cooperative_outputs[p]) > repeated.deviation_tolerance
                for p in players
            ):
                punishment_remaining = repeated.punishment_length

        Q = sum(outputs.values())
        price = price_from_quantity(Q, demand)
        period_profits = {p: (price - cost_map[p]) * outputs[p] for p in players}

        quantities.append(outputs.copy())
        prices.append(price)
        total_quantities.append(Q)
        profits.append(period_profits)

    return RepeatedOutcome(
        quantities=quantities,
        prices=prices,
        total_quantities=total_quantities,
        profits=profits,
    )
