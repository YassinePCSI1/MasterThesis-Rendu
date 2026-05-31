"""Cooperative game theory: Shapley values, coalition stability, and Folk theorem.

This module analyses the cooperative structure of the global crude oil market.

Key concepts
------------
Characteristic function v(S)
    The maximum *total* profit coalition S can guarantee regardless of the
    outside players' strategies.  We implement the Nash-threat version:
    v(S) = profit of S when S plays optimally against the Cournot-Nash
    response of the complementary coalition N \\ S.

Shapley value
    The unique fair allocation of v(N) satisfying efficiency, symmetry,
    dummy, and additivity axioms.  Tells us each producer's expected
    marginal contribution to every possible coalition.

Folk theorem δ*
    Minimum discount factor for which the grim-trigger cooperative agreement
    is individually rational.  If δ ≥ δ*_i for all i, cooperation is
    self-sustaining without external enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from .config import CapacityParams, CostParams, DemandParams
from .cournot_static import _cost_map, _cap_map
from .demand import price_from_quantity, consumer_surplus


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def _coalition_vs_complement(
    coalition: List[str],
    complement: List[str],
    cost_map: Dict[str, float],
    demand: DemandParams,
    cap_map: Dict[str, float],
    caps_enabled: bool,
) -> float:
    """Total profit of *coalition* when it maximises jointly against *complement*.

    The complement players play Cournot Nash among themselves taking the
    coalition's aggregate output as given (Nash-threat bargaining).

    Implementation: bilevel optimisation solved by simple outer grid search
    over coalition aggregate output, with the complement's Nash sub-game
    solved at each point.
    """
    if not coalition:
        return 0.0
    if not complement:
        # Grand coalition: pure monopoly / joint profit max
        return _grand_coalition_profit(coalition, cost_map, demand, cap_map, caps_enabled)

    # Find the coalition's optimal aggregate by scalar search
    q_max = demand.a / demand.b if not caps_enabled else sum(cap_map[p] for p in coalition)

    def neg_coalition_profit(q_c: float) -> float:
        if q_c < 0:
            return 1e12
        # Complement plays Cournot best-response given q_c
        q_comp = _complement_nash(q_c, complement, cost_map, demand, cap_map, caps_enabled)
        Q_comp = sum(q_comp.values())
        Q_total = q_c + Q_comp
        price = price_from_quantity(Q_total, demand)
        # Coalition puts production on cheapest member
        c_min = min(cost_map[p] for p in coalition)
        return -(price - c_min) * q_c

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        neg_coalition_profit,
        bounds=(0.0, q_max),
        method="bounded",
        options={"xatol": 1e-10},
    )
    q_c_star = float(max(0.0, res.x))
    q_comp = _complement_nash(q_c_star, complement, cost_map, demand, cap_map, caps_enabled)
    Q_total = q_c_star + sum(q_comp.values())
    price = price_from_quantity(Q_total, demand)
    c_min = min(cost_map[p] for p in coalition)
    return (price - c_min) * q_c_star


def _complement_nash(
    q_coalition: float,
    complement: List[str],
    cost_map: Dict[str, float],
    demand: DemandParams,
    cap_map: Dict[str, float],
    caps_enabled: bool,
    max_iter: int = 3000,
    tol: float = 1e-10,
    damping: float = 0.5,
) -> Dict[str, float]:
    """Complement players play Cournot Nash given the coalition's total output."""
    q = {p: 0.0 for p in complement}
    for _ in range(max_iter):
        q_old = q.copy()
        for p in complement:
            q_others = q_coalition + sum(q[k] for k in complement if k != p)
            br = (demand.a - cost_map[p] - demand.b * q_others) / (2 * demand.b)
            if caps_enabled:
                br = float(np.clip(br, 0.0, cap_map[p]))
            else:
                br = max(0.0, br)
            q[p] = (1 - damping) * q[p] + damping * br
        if max(abs(q[p] - q_old[p]) for p in complement) < tol:
            break
    return q


def _grand_coalition_profit(
    players: List[str],
    cost_map: Dict[str, float],
    demand: DemandParams,
    cap_map: Dict[str, float],
    caps_enabled: bool,
) -> float:
    """Maximum joint profit for the grand coalition (monopoly solution)."""

    def neg_profit(q_vec: np.ndarray) -> float:
        Q = float(np.sum(q_vec))
        price = price_from_quantity(Q, demand)
        profits = (price - np.array([cost_map[p] for p in players])) * q_vec
        return -float(np.sum(profits))

    bounds = [(0.0, cap_map[p] if caps_enabled else None) for p in players]
    x0 = np.array([cap_map[p] * 0.3 if caps_enabled else 5.0 for p in players])
    res = minimize(neg_profit, x0=x0, bounds=bounds, method="L-BFGS-B")
    return float(-res.fun)


def characteristic_function(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
) -> Dict[Tuple[str, ...], float]:
    """Compute v(S) for all 2^n subsets of *players*."""
    cost_map = _cost_map(costs)
    cap_map = _cap_map(capacities)
    caps_enabled = capacities.enabled
    n = len(players)
    v: Dict[Tuple[str, ...], float] = {}

    for mask in range(1 << n):
        coalition = tuple(sorted(players[i] for i in range(n) if mask & (1 << i)))
        complement = [players[i] for i in range(n) if not (mask & (1 << i))]
        v[coalition] = _coalition_vs_complement(
            list(coalition), complement, cost_map, demand, cap_map, caps_enabled
        )

    v[()] = 0.0
    return v


# ---------------------------------------------------------------------------
# Shapley values
# ---------------------------------------------------------------------------

@dataclass
class ShapleyResult:
    values: Dict[str, float]
    characteristic: Dict[Tuple[str, ...], float]
    grand_coalition_value: float
    core_stable: bool
    stability_margin: Dict[str, float]


def shapley_values(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
) -> ShapleyResult:
    """Compute Shapley values for the oil-market cooperative game.

    Also checks *core stability*: the grand coalition is in the core if
    no sub-coalition can do better on its own, i.e.
    sum_{i in S} φ_i ≥ v(S) for all S.
    """
    v = characteristic_function(players, demand, costs, capacities)
    n = len(players)
    shapley: Dict[str, float] = {p: 0.0 for p in players}

    for perm in permutations(players):
        for idx, player in enumerate(perm):
            s_without = tuple(sorted(perm[:idx]))
            s_with = tuple(sorted(perm[: idx + 1]))
            marginal = v.get(s_with, 0.0) - v.get(s_without, 0.0)
            shapley[player] += marginal

    coeff = 1.0 / factorial(n)
    shapley = {p: shapley[p] * coeff for p in players}

    grand = v.get(tuple(sorted(players)), 0.0)

    # Check core: for every non-empty proper subset S, sum_i shapley[i] >= v(S)
    core_stable = True
    stability_margin: Dict[str, float] = {}
    n_pl = len(players)
    for mask in range(1, (1 << n_pl) - 1):
        sub = tuple(sorted(players[i] for i in range(n_pl) if mask & (1 << i)))
        allocated = sum(shapley[p] for p in sub)
        margin = allocated - v.get(sub, 0.0)
        stability_margin["+".join(sub)] = round(margin, 4)
        if margin < -1e-6:
            core_stable = False

    return ShapleyResult(
        values=shapley,
        characteristic=v,
        grand_coalition_value=grand,
        core_stable=core_stable,
        stability_margin=stability_margin,
    )


# ---------------------------------------------------------------------------
# Folk theorem: minimum discount factor for cooperation
# ---------------------------------------------------------------------------

@dataclass
class FolkTheoremResult:
    delta_star: Dict[str, float]
    delta_binding: float
    binding_player: str
    pi_cooperative: Dict[str, float]
    pi_nash: Dict[str, float]
    pi_deviation: Dict[str, float]
    cooperation_sustainable: bool
    delta_actual: float


def folk_theorem_delta_star(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    delta_actual: float = 0.95,
) -> FolkTheoremResult:
    """Compute the critical discount factor δ* for grim-trigger cooperation.

    The grim-trigger IC constraint for player i is:
        π_coop_i / (1 - δ) ≥ π_dev_i + δ * π_nash_i / (1 - δ)

    Rearranging:
        δ ≥ (π_dev_i - π_coop_i) / (π_dev_i - π_nash_i) ≡ δ*_i

    Cooperation is self-sustaining iff δ ≥ max_i δ*_i.

    Cooperative profile: proportional quota reduction from Nash (OPEC-style
    pro-rata cuts).  Each player reduces output by the same percentage so
    the total equals the joint-profit-maximising quantity.  This ensures
    π_coop_i > π_nash_i for all i (individual rationality without transfers).
    Deviation: one-shot best response to others' cooperative quotas.
    Punishment: revert to Cournot-Nash forever.
    """
    from .cooperation_punishment import cartel_quotas
    from .cournot_static import cournot_equilibrium
    from .cournot_repeated import best_response, AdjustmentCostParams

    cost_map = _cost_map(costs)
    adj = AdjustmentCostParams(enabled=False, k=0.0)

    # --- Nash profits ---
    nash = cournot_equilibrium(players, demand, costs, capacities)
    pi_nash = dict(nash.profits)

    # --- Cooperative profits (proportional cartel quotas) ---
    cartel = cartel_quotas(players, demand, costs, capacities)
    pi_cooperative = dict(cartel.quota_profits)

    # --- Deviation profits ---
    pi_deviation: Dict[str, float] = {}
    for deviator in players:
        q_others_coop = sum(cartel.quotas[p] for p in players if p != deviator)
        q_dev = best_response(
            q_others_coop, cartel.quotas[deviator], demand, cost_map[deviator],
            capacities, deviator, adj,
        )
        q_total_dev = q_dev + q_others_coop
        p_dev = price_from_quantity(q_total_dev, demand)
        pi_deviation[deviator] = (p_dev - cost_map[deviator]) * q_dev

    # --- Critical discount factors ---
    delta_star: Dict[str, float] = {}
    for p in players:
        numerator = pi_deviation[p] - pi_cooperative[p]
        denominator = pi_deviation[p] - pi_nash[p]
        if abs(denominator) < 1e-10:
            delta_star[p] = 0.0
        else:
            delta_star[p] = max(0.0, min(1.0, numerator / denominator))

    binding_player = max(delta_star, key=lambda p: delta_star[p])
    delta_binding = delta_star[binding_player]

    return FolkTheoremResult(
        delta_star=delta_star,
        delta_binding=delta_binding,
        binding_player=binding_player,
        pi_cooperative=pi_cooperative,
        pi_nash=pi_nash,
        pi_deviation=pi_deviation,
        cooperation_sustainable=(delta_actual >= delta_binding),
        delta_actual=delta_actual,
    )
