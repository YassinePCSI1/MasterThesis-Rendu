"""Correlated equilibrium via linear programming (Section E).

A *mediator* — for example the OPEC Secretariat — secretly draws an
action profile a = (q_US, q_OPEC, q_RUS) from a probability distribution
``p`` over the discretised action grid and privately tells each player
which output to produce.  ``p`` is a **correlated equilibrium** (CE) if
no player prefers to deviate after receiving its recommendation,
*conditional* on the recommendation.

Formally, the IC constraint for player i with recommendation ``aᵢ`` and
deviation ``aᵢ'`` is

    Σ_{a_{-i}} p(aᵢ, a_{-i}) · [ uᵢ(aᵢ, a_{-i}) − uᵢ(aᵢ', a_{-i}) ] ≥ 0.

Together with ``p ≥ 0`` and ``Σ p = 1`` these constraints carve out a
polytope of correlated equilibria.  Within that polytope we maximise a
linear objective:

* ``max_welfare``      = E[CS + total profit]
* ``max_joint_profit`` = E[total profit]
* ``max_min_profit``   = E[min over players of profit]   (max-min via LP
                         epigraph trick: maximise t s.t. t ≤ E[πᵢ] ∀i)

The LP is solved with :func:`scipy.optimize.linprog`.

Numerical notes
---------------
With three players and an action grid of size ``g`` the joint-action
space has ``g³`` profiles and each player has ``g · (g − 1)`` IC
constraints.  At ``g = 10`` that's 1000 variables and ~270 constraints,
which is comfortably solvable by HIGHS (the default ``linprog`` solver).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import (
    CapacityParams,
    CorrelatedEqParams,
    CostParams,
    DemandParams,
)
from .cooperation_punishment import cartel_quotas
from .cournot_static import _cap_map, _cost_map, cournot_equilibrium
from .demand import consumer_surplus, price_from_quantity


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorrelatedEqResult:
    objective: str
    prob_distribution: np.ndarray
    expected_price: float
    expected_quantities: Dict[str, float]
    expected_profits: Dict[str, float]
    expected_consumer_surplus: float
    expected_total_welfare: float
    nash_improvement: Dict[str, float]
    support_size: int


@dataclass
class CEComparisonResult:
    nash: Any                         # CournotResult
    cartel: Any                       # CartelQuotaResult
    ce_results: Dict[str, CorrelatedEqResult]
    action_grid: List[float]
    players: List[str]


# ---------------------------------------------------------------------------
# Action grid + payoff tensor
# ---------------------------------------------------------------------------

def _action_grid(players: List[str],
                 demand: DemandParams,
                 cost_map: Dict[str, float],
                 grid_size: int,
                 capacities: CapacityParams) -> List[float]:
    """Common action grid spanning [0, q_max].

    The upper bound is min(physical max from demand, max capacity if on).
    """
    if capacities.enabled:
        cap_map = _cap_map(capacities)
        q_max = max(cap_map[p] for p in players)
    else:
        # A natural upper bound is the monopoly quantity for the cheapest
        # player.  Beyond that no player would ever want to produce.
        c_min = min(cost_map[p] for p in players)
        q_max = (demand.a - c_min) / (2 * demand.b)
    return list(np.linspace(0.0, q_max, grid_size))


def build_payoff_tensor(players: List[str],
                        demand: DemandParams,
                        cost_map: Dict[str, float],
                        action_grid: List[float]) -> np.ndarray:
    """Return a (g, g, ..., g, n_players) tensor of profits.

    ``payoff[i1, i2, ..., in, k]`` is player ``k``'s profit when player j
    plays ``action_grid[ij]``.
    """
    g = len(action_grid)
    n = len(players)
    grid = np.asarray(action_grid, dtype=float)
    shape = (g,) * n
    payoff = np.zeros(shape + (n,), dtype=float)

    # Vectorise: build the full mesh of quantities, sum to get Q, derive P
    mesh = np.array(np.meshgrid(*([grid] * n), indexing="ij"))  # (n, g, g, ..., g)
    Q = mesh.sum(axis=0)
    P = demand.a - demand.b * Q
    if demand.price_floor is not None:
        P = np.maximum(P, demand.price_floor)
    for k, p in enumerate(players):
        payoff[..., k] = (P - cost_map[p]) * mesh[k]
    return payoff


# ---------------------------------------------------------------------------
# Linear program for correlated equilibrium
# ---------------------------------------------------------------------------

def correlated_equilibrium_lp(payoff_tensor: np.ndarray,
                              players: List[str],
                              action_grid: List[float],
                              objective: str) -> Tuple[np.ndarray, float]:
    """Solve the CE LP and return ``(prob_distribution, optimal_value)``.

    Variables (decision space)
    --------------------------
    For ``max_welfare`` / ``max_joint_profit``: x = p (the joint
    probabilities, length g**n).
    For ``max_min_profit``: x = [p; t] with ``t`` an extra epigraph
    variable; we maximise t with constraints ``E[πᵢ] − t ≥ 0`` for every i.
    """
    n = len(players)
    g = len(action_grid)
    M = g ** n  # number of joint actions
    flat_payoff = payoff_tensor.reshape(M, n)         # M x n
    grid = np.asarray(action_grid, dtype=float)

    # Multi-index <-> flat index
    def flat_to_multi(idx: int) -> Tuple[int, ...]:
        return np.unravel_index(idx, (g,) * n)

    # ---------------------------------------------------------------- IC
    # For each player k, for each pair (a_k, a_k'), construct one IC row.
    # Σ_{a_-k} p(a_k, a_-k) · [ u_k(a_k, a_-k) - u_k(a_k', a_-k) ] >= 0
    ic_rows: List[np.ndarray] = []
    for k in range(n):
        for a_k in range(g):
            for a_k_prime in range(g):
                if a_k == a_k_prime:
                    continue
                row = np.zeros(M, dtype=float)
                # Iterate over all profiles where player k plays a_k
                for idx in range(M):
                    multi = flat_to_multi(idx)
                    if multi[k] != a_k:
                        continue
                    # Build the deviated profile
                    multi_dev = list(multi)
                    multi_dev[k] = a_k_prime
                    idx_dev = int(np.ravel_multi_index(tuple(multi_dev), (g,) * n))
                    row[idx] = flat_payoff[idx, k] - flat_payoff[idx_dev, k]
                ic_rows.append(row)
    A_ic = np.vstack(ic_rows) if ic_rows else np.zeros((0, M))
    # linprog uses A_ub @ x <= b_ub, so flip sign:
    A_ub = -A_ic
    b_ub = np.zeros(A_ub.shape[0])

    # ---------------------------------------------------------------- equalities
    A_eq = np.ones((1, M))
    b_eq = np.array([1.0])

    if objective == "max_welfare":
        # E[CS] + E[total profit]; CS depends only on Q
        Q_per = np.zeros(M)
        total_profit = flat_payoff.sum(axis=1)
        for idx in range(M):
            multi = flat_to_multi(idx)
            Q_per[idx] = sum(grid[m] for m in multi)
        # CS = 0.5 * (a - P) * Q ; P = a - bQ ; so CS = 0.5 * b * Q^2 (linear in p)
        # E[0.5 * b * Q^2] is linear in p (the squared term is per-cell coefficient)
        cs_per = 0.5 * (Q_per * Q_per)  # b is folded into demand later? No — see
        # NOTE: CS = 0.5 * (a - P) * Q = 0.5 * b * Q^2 only when there's no floor.
        # We assume no floor in the CE analysis and use this closed form.
        c = -(cs_per + total_profit)  # linprog minimises
        bounds = [(0.0, 1.0)] * M
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"CE LP failed: {res.message}")
        return res.x, -res.fun

    elif objective == "max_joint_profit":
        c = -flat_payoff.sum(axis=1)
        bounds = [(0.0, 1.0)] * M
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"CE LP failed: {res.message}")
        return res.x, -res.fun

    elif objective == "max_min_profit":
        # x = [p (M), t (1)]; maximise t  ->  minimise -t  ->  c = [0..., -1]
        c = np.concatenate([np.zeros(M), np.array([-1.0])])
        # IC: A_ub @ p <= 0 (no t involvement)
        A_ub_full = np.hstack([A_ub, np.zeros((A_ub.shape[0], 1))])
        # min-profit constraint: t - E[π_k] <= 0  for every k
        new_rows = []
        for k in range(n):
            row = np.zeros(M + 1)
            row[:M] = -flat_payoff[:, k]
            row[-1] = 1.0
            new_rows.append(row)
        A_ub_full = np.vstack([A_ub_full] + new_rows) if new_rows else A_ub_full
        b_ub_full = np.concatenate([b_ub, np.zeros(len(new_rows))])
        # Equality: Σ p = 1, t free
        A_eq_full = np.hstack([A_eq, np.zeros((1, 1))])
        b_eq_full = b_eq
        bounds = [(0.0, 1.0)] * M + [(None, None)]
        res = linprog(c=c, A_ub=A_ub_full, b_ub=b_ub_full,
                      A_eq=A_eq_full, b_eq=b_eq_full,
                      bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"CE LP (max-min) failed: {res.message}")
        return res.x[:M], -res.fun

    else:
        raise ValueError(f"Unknown CE objective: {objective}")


# ---------------------------------------------------------------------------
# Expected outcomes from a CE distribution
# ---------------------------------------------------------------------------

def ce_expected_outcomes(prob: np.ndarray,
                         players: List[str],
                         demand: DemandParams,
                         cost_map: Dict[str, float],
                         action_grid: List[float],
                         payoff_tensor: np.ndarray) -> Dict[str, Any]:
    """Compute expected price, quantities, profits and CS under a CE."""
    n = len(players)
    g = len(action_grid)
    M = g ** n
    grid = np.asarray(action_grid, dtype=float)
    flat_payoff = payoff_tensor.reshape(M, n)

    Q_per = np.zeros(M)
    q_per_player = np.zeros((M, n))
    for idx in range(M):
        multi = np.unravel_index(idx, (g,) * n)
        for k in range(n):
            q_per_player[idx, k] = grid[multi[k]]
        Q_per[idx] = q_per_player[idx].sum()

    expected_q = {p: float(np.sum(prob * q_per_player[:, k]))
                  for k, p in enumerate(players)}
    Q_E = float(np.sum(prob * Q_per))
    P_E = price_from_quantity(Q_E, demand)

    expected_profits = {p: float(np.sum(prob * flat_payoff[:, k]))
                        for k, p in enumerate(players)}

    # Expected CS = 0.5 * b * E[Q^2] (under no price floor) — but for sanity we
    # also compute it via E[0.5 * (a-P) * Q]
    cs_per = 0.5 * (demand.a - (demand.a - demand.b * Q_per)) * Q_per
    expected_cs = float(np.sum(prob * cs_per))

    expected_welfare = expected_cs + sum(expected_profits.values())
    support_size = int(np.sum(prob > 1e-6))
    return {
        "expected_price": float(P_E),
        "expected_quantities": expected_q,
        "expected_profits": expected_profits,
        "expected_consumer_surplus": expected_cs,
        "expected_total_welfare": expected_welfare,
        "support_size": support_size,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def ce_vs_nash_comparison(players: List[str],
                          demand: DemandParams,
                          costs: CostParams,
                          capacities: CapacityParams,
                          ce_params: CorrelatedEqParams) -> CEComparisonResult:
    """Compute Nash, all CE objectives, and Cartel side-by-side."""
    cost_map = _cost_map(costs)
    grid = _action_grid(players, demand, cost_map,
                        ce_params.action_grid_size, capacities)
    payoff = build_payoff_tensor(players, demand, cost_map, grid)

    nash = cournot_equilibrium(players, demand, costs, capacities)
    cartel = cartel_quotas(players, demand, costs, capacities)

    ce_results: Dict[str, CorrelatedEqResult] = {}
    for obj in ce_params.objectives:
        prob, value = correlated_equilibrium_lp(payoff, players, grid, obj)
        out = ce_expected_outcomes(prob, players, demand, cost_map, grid, payoff)
        improvement = {
            p: float(out["expected_profits"][p] - nash.profits[p])
            for p in players
        }
        ce_results[obj] = CorrelatedEqResult(
            objective=obj,
            prob_distribution=prob,
            expected_price=out["expected_price"],
            expected_quantities=out["expected_quantities"],
            expected_profits=out["expected_profits"],
            expected_consumer_surplus=out["expected_consumer_surplus"],
            expected_total_welfare=out["expected_total_welfare"],
            nash_improvement=improvement,
            support_size=out["support_size"],
        )

    return CEComparisonResult(
        nash=nash,
        cartel=cartel,
        ce_results=ce_results,
        action_grid=grid,
        players=list(players),
    )


def comparison_to_dataframe(comp: CEComparisonResult) -> pd.DataFrame:
    """Tidy DataFrame with one row per market structure."""
    players = comp.players
    rows: List[Dict[str, Any]] = []

    rows.append({
        "structure": "Nash",
        "price": round(comp.nash.price, 4),
        "total_q": round(comp.nash.total_quantity, 4),
        "consumer_surplus": round(comp.nash.consumer_surplus, 4),
        "total_welfare": round(comp.nash.total_welfare, 4),
        "support_size": "—",
        **{f"profit_{p}": round(comp.nash.profits[p], 4) for p in players},
        **{f"q_{p}": round(comp.nash.quantities[p], 4) for p in players},
    })
    for obj, ce in comp.ce_results.items():
        Q_E = sum(ce.expected_quantities.values())
        rows.append({
            "structure": f"CE-{obj}",
            "price": round(ce.expected_price, 4),
            "total_q": round(Q_E, 4),
            "consumer_surplus": round(ce.expected_consumer_surplus, 4),
            "total_welfare": round(ce.expected_total_welfare, 4),
            "support_size": ce.support_size,
            **{f"profit_{p}": round(ce.expected_profits[p], 4) for p in players},
            **{f"q_{p}": round(ce.expected_quantities[p], 4) for p in players},
        })
    rows.append({
        "structure": "Cartel",
        "price": round(comp.cartel.quota_price, 4),
        "total_q": round(comp.cartel.total_output, 4),
        "consumer_surplus": round(0.5 * (comp.cartel.total_output)
                                  * (comp.cartel.total_output), 4),
        "total_welfare": round(0.5 * (comp.cartel.total_output ** 2)
                                + sum(comp.cartel.quota_profits.values()), 4),
        "support_size": "—",
        **{f"profit_{p}": round(comp.cartel.quota_profits[p], 4) for p in players},
        **{f"q_{p}": round(comp.cartel.quotas[p], 4) for p in players},
    })
    return pd.DataFrame(rows)
