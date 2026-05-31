"""Cooperation, deviation detection, and punishment strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

from .config import CapacityParams, CostParams, DemandParams
from .cournot_static import cournot_equilibrium
from .demand import price_from_quantity


@dataclass
class CooperationResult:
    target_outputs: Dict[str, float]
    target_price: float
    target_profit: Dict[str, float]


@dataclass
class CartelQuotaResult:
    """Cartel quotas via proportional output restriction from Nash.

    Each player cuts production by the same percentage so that the total
    equals the joint-profit-maximising quantity.  This is the standard
    OPEC-style quota mechanism (pro-rata cuts from a baseline) and
    ensures individual rationality without side payments.
    """

    quotas: Dict[str, float]
    quota_price: float
    quota_profits: Dict[str, float]
    total_output: float
    reduction_pct: float


def _cost_map(costs: CostParams) -> Dict[str, float]:
    return {"US": costs.c_us, "OPEC": costs.c_opec, "RUS": costs.c_rus}


def _cap_map(caps: CapacityParams) -> Dict[str, float]:
    return {"US": caps.cap_us, "OPEC": caps.cap_opec, "RUS": caps.cap_rus}


def cooperative_output(players: List[str],
                       demand: DemandParams,
                       costs: CostParams,
                       capacities: CapacityParams) -> CooperationResult:
    """Joint-profit maximizing output profile under constraints."""

    cost_map = _cost_map(costs)
    cap_map = _cap_map(capacities)

    def objective(q_vec: np.ndarray) -> float:
        Q = float(np.sum(q_vec))
        price = price_from_quantity(Q, demand)
        profits = (price - np.array([cost_map[p] for p in players])) * q_vec
        return -float(np.sum(profits))

    bounds = []
    for p in players:
        if capacities.enabled:
            bounds.append((0.0, cap_map[p]))
        else:
            bounds.append((0.0, None))

    x0 = np.array([cap_map[p] * 0.5 if capacities.enabled else 5.0 for p in players])
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    q_star = np.maximum(res.x, 0.0)

    Q = float(np.sum(q_star))
    price = price_from_quantity(Q, demand)
    profits = {p: (price - cost_map[p]) * q_star[i] for i, p in enumerate(players)}

    return CooperationResult(
        target_outputs={p: float(q_star[i]) for i, p in enumerate(players)},
        target_price=price,
        target_profit=profits,
    )


def cartel_quotas(players: List[str],
                  demand: DemandParams,
                  costs: CostParams,
                  capacities: CapacityParams) -> CartelQuotaResult:
    """Compute cartel quotas as proportional reductions from Nash outputs.

    The joint-profit-maximising total Q* is computed first.  Then each
    player's quota is set to q_nash_i * (Q* / Q_nash), preserving
    relative market shares.  All players earn more than at Nash because
    the total supply reduction raises the price for everyone.

    This differs from ``cooperative_output()`` which concentrates
    production on the cheapest player (true joint-max, requires side
    payments).  Cartel quotas are individually rational *without*
    transfers and therefore appropriate for the Folk theorem and
    evolutionary-game payoff matrices.
    """
    coop = cooperative_output(players, demand, costs, capacities)
    nash = cournot_equilibrium(players, demand, costs, capacities)

    Q_coop = sum(coop.target_outputs.values())
    Q_nash = nash.total_quantity

    ratio = Q_coop / Q_nash if Q_nash > 0 else 1.0
    quotas = {p: nash.quantities[p] * ratio for p in players}

    cost_map = _cost_map(costs)
    price = price_from_quantity(Q_coop, demand)
    profits = {p: (price - cost_map[p]) * quotas[p] for p in players}

    return CartelQuotaResult(
        quotas=quotas,
        quota_price=price,
        quota_profits=profits,
        total_output=Q_coop,
        reduction_pct=round((1 - ratio) * 100, 2),
    )


