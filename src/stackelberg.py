"""Stackelberg quantity leadership model for the global crude oil market.

In the Stackelberg model the leader commits to a quantity first; followers then
play a Cournot Nash game taking the leader's quantity as given.  This is the
standard model for OPEC's role as a dominant first-mover in world oil supply.

We also expose a generic solver so any one of the three players can act as
leader (e.g. 'US shale revolution' scenario where US capacity growth is the
de-facto first-mover).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .config import CapacityParams, CostParams, DemandParams
from .cournot_static import CournotResult, profit, _cost_map, _cap_map
from .demand import price_from_quantity, consumer_surplus


@dataclass
class StackelbergResult:
    leader: str
    followers: List[str]
    quantities: Dict[str, float]
    price: float
    total_quantity: float
    profits: Dict[str, float]
    consumer_surplus: float
    total_welfare: float
    leader_advantage: float   # leader profit minus Nash profit


def _followers_cournot_given_leader(
    leader_q: float,
    followers: List[str],
    cost_map: Dict[str, float],
    demand: DemandParams,
    cap_map: Dict[str, float],
    caps_enabled: bool,
    max_iter: int = 3000,
    tol: float = 1e-10,
    damping: float = 0.5,
) -> Dict[str, float]:
    """Solve followers' Cournot subgame for a given leader quantity.

    Followers treat leader_q as fixed and play Cournot Nash among themselves.
    For the two-follower, asymmetric-cost case this has a closed-form solution;
    we use iterative best-response here so the function generalises to any
    number of followers and to capacity constraints.
    """
    q = {p: 0.0 for p in followers}

    for _ in range(max_iter):
        q_old = q.copy()
        for p in followers:
            q_others_followers = sum(q[k] for k in followers if k != p)
            q_others_total = leader_q + q_others_followers
            br = (demand.a - cost_map[p] - demand.b * q_others_total) / (2 * demand.b)
            if caps_enabled:
                br = float(np.clip(br, 0.0, cap_map[p]))
            else:
                br = max(0.0, br)
            q[p] = (1 - damping) * q[p] + damping * br
        if max(abs(q[p] - q_old[p]) for p in followers) < tol:
            break

    return q


def _stackelberg_leader_profit(
    leader_q: float,
    leader: str,
    followers: List[str],
    cost_map: Dict[str, float],
    demand: DemandParams,
    cap_map: Dict[str, float],
    caps_enabled: bool,
) -> float:
    """Return negative leader profit (for minimisation)."""
    if leader_q < 0:
        return 1e12
    q_followers = _followers_cournot_given_leader(
        leader_q, followers, cost_map, demand, cap_map, caps_enabled
    )
    Q = leader_q + sum(q_followers.values())
    price = price_from_quantity(Q, demand)
    return -(price - cost_map[leader]) * leader_q


def stackelberg_equilibrium(
    leader: str,
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    nash_result: Optional[CournotResult] = None,
) -> StackelbergResult:
    """Compute the Stackelberg equilibrium given *leader* as the first-mover.

    Parameters
    ----------
    leader:
        The player who moves first (e.g. 'OPEC').
    players:
        Full list of active players.
    nash_result:
        Pre-computed Cournot-Nash result for the same players, used to
        compute the *leader_advantage* statistic.
    """
    followers = [p for p in players if p != leader]
    cost_map = _cost_map(costs)
    cap_map = _cap_map(capacities)
    caps_enabled = capacities.enabled

    # Search range for leader quantity
    if caps_enabled:
        q_upper = cap_map[leader]
    else:
        q_upper = demand.a / demand.b  # physical maximum

    result = minimize_scalar(
        _stackelberg_leader_profit,
        bounds=(0.0, q_upper),
        method="bounded",
        args=(leader, followers, cost_map, demand, cap_map, caps_enabled),
        options={"xatol": 1e-10},
    )
    q_leader = float(max(0.0, result.x))

    q_followers = _followers_cournot_given_leader(
        q_leader, followers, cost_map, demand, cap_map, caps_enabled
    )

    quantities = {leader: q_leader, **q_followers}
    Q = sum(quantities.values())
    price = price_from_quantity(Q, demand)
    profits = {p: profit(quantities[p], price, cost_map[p]) for p in players}
    cs = consumer_surplus(Q, price, demand)
    total_welfare = cs + sum(profits.values())

    leader_advantage = 0.0
    if nash_result is not None and leader in nash_result.profits:
        leader_advantage = profits[leader] - nash_result.profits[leader]

    return StackelbergResult(
        leader=leader,
        followers=followers,
        quantities=quantities,
        price=price,
        total_quantity=Q,
        profits=profits,
        consumer_surplus=cs,
        total_welfare=total_welfare,
        leader_advantage=leader_advantage,
    )
