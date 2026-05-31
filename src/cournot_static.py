"""Static Cournot models: triopoly and duopoly baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import CapacityParams, CostParams, DemandParams, PLAYERS
from .demand import price_from_quantity, consumer_surplus


@dataclass
class CournotResult:
    quantities: Dict[str, float]
    price: float
    total_quantity: float
    profits: Dict[str, float]
    consumer_surplus: float
    total_welfare: float


def profit(q: float, price: float, c: float) -> float:
    return (price - c) * q


def _cost_map(costs: CostParams) -> Dict[str, float]:
    return {"US": costs.c_us, "OPEC": costs.c_opec, "RUS": costs.c_rus}


def _cap_map(caps: CapacityParams) -> Dict[str, float]:
    return {"US": caps.cap_us, "OPEC": caps.cap_opec, "RUS": caps.cap_rus}


def _solve_linear_equilibrium(players: List[str],
                              costs: Dict[str, float],
                              demand: DemandParams) -> Dict[str, float]:
    """Solve for Cournot-Nash in closed form for linear demand with constants."""

    n = len(players)
    b = demand.b
    A = np.full((n, n), b)
    np.fill_diagonal(A, 2 * b)
    rhs = np.array([demand.a - costs[p] for p in players], dtype=float)
    q = np.linalg.solve(A, rhs)
    q = np.maximum(q, 0.0)
    return {p: float(q[i]) for i, p in enumerate(players)}


def _best_response(q_others: float, demand: DemandParams, c_i: float,
                   cap_i: float | None) -> float:
    """Best response for linear demand with optional capacity."""

    q_i = (demand.a - demand.b * q_others - c_i) / (2 * demand.b)
    if cap_i is not None:
        q_i = float(np.clip(q_i, 0.0, cap_i))
    else:
        q_i = max(0.0, q_i)
    return q_i


def _solve_capacity_equilibrium(players: List[str],
                                costs: Dict[str, float],
                                demand: DemandParams,
                                caps: Dict[str, float],
                                max_iter: int = 5000,
                                tol: float = 1e-8,
                                damping: float = 0.5) -> Dict[str, float]:
    """Solve Cournot equilibrium with capacity constraints via iteration."""

    q = {p: 0.5 * caps[p] for p in players}
    for _ in range(max_iter):
        q_old = q.copy()
        for p in players:
            q_others = sum(q[k] for k in players if k != p)
            br = _best_response(q_others, demand, costs[p], caps[p])
            q[p] = (1 - damping) * q[p] + damping * br
        if max(abs(q[p] - q_old[p]) for p in players) < tol:
            break
    return q


def cournot_equilibrium(players: List[str] | None,
                         demand: DemandParams,
                         costs: CostParams,
                         capacities: CapacityParams) -> CournotResult:
    """Compute Cournot equilibrium for the selected players."""

    players = players or PLAYERS
    cost_map = _cost_map(costs)
    cap_map = _cap_map(capacities)
    if capacities.enabled:
        quantities = _solve_capacity_equilibrium(players, cost_map, demand, cap_map)
    else:
        quantities = _solve_linear_equilibrium(players, cost_map, demand)

    total_q = sum(quantities.values())
    price = price_from_quantity(total_q, demand)
    profits = {p: profit(quantities[p], price, cost_map[p]) for p in players}
    cs = consumer_surplus(total_q, price, demand)
    total_welfare = cs + sum(profits.values())

    assert total_q >= 0.0
    assert np.isfinite(price)
    return CournotResult(
        quantities=quantities,
        price=price,
        total_quantity=total_q,
        profits=profits,
        consumer_surplus=cs,
        total_welfare=total_welfare,
    )


def duopoly_vs_triopoly(demand: DemandParams,
                        costs: CostParams,
                        capacities: CapacityParams) -> Tuple[CournotResult, CournotResult]:
    """Compute duopoly (US vs OPEC) and triopoly outcomes."""

    duo = cournot_equilibrium(["US", "OPEC"], demand, costs, capacities)
    tri = cournot_equilibrium(["US", "OPEC", "RUS"], demand, costs, capacities)
    return duo, tri
