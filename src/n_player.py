"""N-player sensitivity (Section D).

The whole baseline analysis is calibrated on **three** players (US,
OPEC, RUS).  This module asks how the conclusions change as the number
of producers varies from 2 to 6 — simulating OPEC fragmentation, the
emergence of new entrants (Brazil pre-salt, Canada oil sands, Norway,
Iraq breakaway), or the consolidation back to a duopoly.

Why a separate module?
----------------------
The existing :func:`cournot_static.cournot_equilibrium` and the
companion helpers hard-code the cost map to ``c_us / c_opec / c_rus``.
Rather than touching the existing API we re-implement the pure-Cournot
linear solver here in a way that accepts an arbitrary
``Dict[str, float]`` cost map.  This is **option (b)** from the
specification: a generalised standalone solver that does not modify
the calibrated three-player code.

Outputs
-------
:class:`NPlayerSweepResult` — Nash, cartel, Folk-theorem delta*,
HHI, and OPEC-specific power statistics for every n in {2, ..., max_n}.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .config import CapacityParams, CostParams, DemandParams, NPlayerParams
from .demand import consumer_surplus, price_from_quantity


# ---------------------------------------------------------------------------
# Cost map / config helpers
# ---------------------------------------------------------------------------

BASE_COST_BY_NAME = {"US": "c_us", "OPEC": "c_opec", "RUS": "c_rus"}


def _extended_cost_map(
    base_costs: CostParams,
    extra_names: List[str],
    extra_costs: List[float],
) -> Dict[str, float]:
    """Build a dict[player -> marginal cost] from the calibrated three plus extras."""
    base = {"US": base_costs.c_us, "OPEC": base_costs.c_opec, "RUS": base_costs.c_rus}
    base.update(dict(zip(extra_names, extra_costs)))
    return base


def build_n_player_config(
    n: int,
    base_players: List[str],
    base_costs: CostParams,
    extra_names: List[str],
    extra_costs: List[float],
) -> Tuple[List[str], Dict[str, float]]:
    """Return ``(players, cost_map)`` for ``n`` producers.

    Convention:
      * n = 2  -> ['OPEC', 'US']
      * n = 3  -> ['US', 'OPEC', 'RUS'] (baseline)
      * n > 3  -> baseline + extras[:n-3]
    """
    if n == 2:
        players = ["OPEC", "US"]
    elif n == 3:
        players = list(base_players)
    else:
        if n - 3 > len(extra_names):
            raise ValueError(
                f"n={n} requires {n - 3} extra players but only "
                f"{len(extra_names)} are configured"
            )
        players = list(base_players) + list(extra_names[: n - 3])
    cost_map = _extended_cost_map(base_costs, extra_names, extra_costs)
    return players, cost_map


# ---------------------------------------------------------------------------
# Generalised Cournot solver
# ---------------------------------------------------------------------------

def n_player_cournot(players: List[str],
                     cost_map: Dict[str, float],
                     demand: DemandParams,
                     capacities: Optional[Dict[str, float]] = None,
                     ) -> Dict[str, float]:
    """Solve unconstrained linear Cournot with arbitrary players.

    Closed form: with linear demand ``P = a − bQ`` and constant marginal
    costs ``cᵢ``, FOCs give ``2 b qᵢ + b Σ_{j≠i} qⱼ = a − cᵢ``.  This
    is a linear system with diagonal-dominant matrix.
    """
    n = len(players)
    A = np.full((n, n), demand.b)
    np.fill_diagonal(A, 2 * demand.b)
    rhs = np.array([demand.a - cost_map[p] for p in players], dtype=float)
    q = np.linalg.solve(A, rhs)
    q = np.maximum(q, 0.0)
    if capacities is not None:
        # Iterative best-response with capacities
        q = _capacity_iter(players, cost_map, demand, capacities)
    return {p: float(q[i]) for i, p in enumerate(players)} if isinstance(q, np.ndarray) \
        else q


def _capacity_iter(players: List[str],
                   cost_map: Dict[str, float],
                   demand: DemandParams,
                   caps: Dict[str, float],
                   max_iter: int = 5000,
                   tol: float = 1e-10,
                   damping: float = 0.5) -> Dict[str, float]:
    q = {p: 0.5 * caps.get(p, demand.a / demand.b) for p in players}
    for _ in range(max_iter):
        q_old = dict(q)
        for p in players:
            others = sum(q[k] for k in players if k != p)
            br = (demand.a - cost_map[p] - demand.b * others) / (2 * demand.b)
            cap_p = caps.get(p, None)
            if cap_p is not None:
                br = float(np.clip(br, 0.0, cap_p))
            else:
                br = max(0.0, br)
            q[p] = (1 - damping) * q[p] + damping * br
        if max(abs(q[p] - q_old[p]) for p in players) < tol:
            break
    return q


def n_player_cooperative(players: List[str],
                         cost_map: Dict[str, float],
                         demand: DemandParams,
                         capacities: Optional[Dict[str, float]] = None
                         ) -> Dict[str, float]:
    """Joint-profit-maximising output profile with arbitrary players."""
    def neg_profit(q_vec: np.ndarray) -> float:
        Q = float(np.sum(q_vec))
        price = price_from_quantity(Q, demand)
        profits = (price - np.array([cost_map[p] for p in players])) * q_vec
        return -float(np.sum(profits))

    bounds = []
    x0 = []
    for p in players:
        if capacities is not None and p in capacities:
            bounds.append((0.0, capacities[p]))
            x0.append(0.5 * capacities[p])
        else:
            bounds.append((0.0, None))
            x0.append(5.0)
    res = minimize(neg_profit, x0=np.array(x0), bounds=bounds, method="L-BFGS-B")
    q_star = np.maximum(res.x, 0.0)
    return {p: float(q_star[i]) for i, p in enumerate(players)}


def _n_player_best_response(q_others: float, c_i: float,
                            demand: DemandParams) -> float:
    """Unconstrained best response under linear demand."""
    return max(0.0, (demand.a - demand.b * q_others - c_i) / (2 * demand.b))


def _folk_delta_star(players: List[str],
                     cost_map: Dict[str, float],
                     demand: DemandParams,
                     nash_q: Dict[str, float],
                     cartel_q: Dict[str, float]) -> Tuple[float, str, Dict[str, float]]:
    """Compute the Folk-theorem delta* for an n-player Cournot game.

    Uses the **same** definition as the existing ``coalition.folk_theorem_delta_star``:
    grim-trigger trigger strategy, deviation = best response to the rivals'
    cooperative quotas, punishment = revert to Nash forever.
    """
    Q_nash = sum(nash_q.values())
    P_nash = price_from_quantity(Q_nash, demand)
    pi_nash = {p: (P_nash - cost_map[p]) * nash_q[p] for p in players}

    Q_coop = sum(cartel_q.values())
    P_coop = price_from_quantity(Q_coop, demand)
    pi_coop = {p: (P_coop - cost_map[p]) * cartel_q[p] for p in players}

    pi_dev: Dict[str, float] = {}
    for d in players:
        q_others = sum(cartel_q[p] for p in players if p != d)
        q_dev = _n_player_best_response(q_others, cost_map[d], demand)
        Q_dev = q_dev + q_others
        P_dev = price_from_quantity(Q_dev, demand)
        pi_dev[d] = (P_dev - cost_map[d]) * q_dev

    delta_star: Dict[str, float] = {}
    for p in players:
        num = pi_dev[p] - pi_coop[p]
        den = pi_dev[p] - pi_nash[p]
        if abs(den) < 1e-10:
            delta_star[p] = 0.0
        else:
            delta_star[p] = max(0.0, min(1.0, num / den))

    binding = max(delta_star, key=lambda p: delta_star[p])
    return delta_star[binding], binding, delta_star


def _generalised_shapley(players: List[str],
                         cost_map: Dict[str, float],
                         demand: DemandParams,
                         capacities: Optional[Dict[str, float]] = None
                         ) -> Dict[str, float]:
    """Compute Shapley values via the *cooperative-profit characteristic*
    function v(S) = max joint profit of S given S is the only player set.

    For the n-player sweep this is a coarse but tractable approximation
    (Nash-threat for every coalition becomes prohibitively expensive at
    n=6).  When the user wants the *exact* Nash-threat Shapley they can
    still run the existing 3-player ``coalition.shapley_values``.
    """
    def coalition_profit(coal: Tuple[str, ...]) -> float:
        if not coal:
            return 0.0
        sub_costs = {p: cost_map[p] for p in coal}
        # Standalone joint-profit maximisation by the coalition
        q = n_player_cooperative(list(coal), sub_costs, demand, capacities)
        Q = sum(q.values())
        P = price_from_quantity(Q, demand)
        return float(sum((P - cost_map[p]) * q[p] for p in coal))

    n = len(players)
    shap = {p: 0.0 for p in players}
    # Iterate over all permutations (n! — n=6 = 720, still cheap)
    for perm in permutations(players):
        for idx, p in enumerate(perm):
            without = tuple(sorted(perm[:idx]))
            with_ = tuple(sorted(perm[: idx + 1]))
            shap[p] += coalition_profit(with_) - coalition_profit(without)
    coeff = 1.0 / factorial(n)
    return {p: shap[p] * coeff for p in players}


# ---------------------------------------------------------------------------
# Top-level sweep dataclass + driver
# ---------------------------------------------------------------------------

@dataclass
class NPlayerSweepResult:
    n_values: List[int]
    nash_prices: List[float]
    nash_total_quantities: List[float]
    cartel_prices: List[float]
    delta_star_binding: List[float]
    hhi_values: List[float]
    opec_profits_nash: List[float]
    opec_profits_cartel: List[float]
    opec_shapley: List[float]
    opec_market_shares: List[float]
    player_lists: List[List[str]]


def n_player_sweep(
    base_players: List[str],
    demand: DemandParams,
    base_costs: CostParams,
    capacities: CapacityParams,
    n_player_params: NPlayerParams,
) -> NPlayerSweepResult:
    """Sweep n from 2 to ``max_players`` and record summary statistics."""
    n_vals: List[int] = []
    nash_prices: List[float] = []
    nash_qs: List[float] = []
    cartel_prices: List[float] = []
    delta_stars: List[float] = []
    hhis: List[float] = []
    opec_nash: List[float] = []
    opec_cartel: List[float] = []
    opec_shap: List[float] = []
    opec_share: List[float] = []
    player_lists: List[List[str]] = []

    caps_map = (
        {"US": capacities.cap_us, "OPEC": capacities.cap_opec,
         "RUS": capacities.cap_rus}
        if capacities.enabled else None
    )

    for n in range(2, n_player_params.max_players + 1):
        players, cost_map = build_n_player_config(
            n, base_players, base_costs,
            n_player_params.extra_player_names,
            n_player_params.extra_player_costs,
        )
        # When capacities are off we don't need a cap_map; otherwise extras
        # get a generous capacity (their entry would otherwise be 0).
        if caps_map is None:
            cap_arg: Optional[Dict[str, float]] = None
        else:
            cap_arg = {p: caps_map.get(p, demand.a / demand.b) for p in players}

        nash = n_player_cournot(players, cost_map, demand, cap_arg)
        joint_max = n_player_cooperative(players, cost_map, demand, cap_arg)

        Q_nash = sum(nash.values())
        Q_joint = sum(joint_max.values())
        # Use *proportional* cartel quotas (like cartel_quotas in
        # cooperation_punishment.py) so that the Folk-theorem delta* remains
        # comparable with the existing 3-player coalition.folk_theorem_delta_star.
        ratio = Q_joint / Q_nash if Q_nash > 0 else 1.0
        cartel = {p: nash[p] * ratio for p in players}
        Q_cartel = sum(cartel.values())
        P_nash = price_from_quantity(Q_nash, demand)
        P_cartel = price_from_quantity(Q_cartel, demand)

        delta_b, binding_p, _ = _folk_delta_star(players, cost_map, demand, nash, cartel)
        shares = {p: (nash[p] / Q_nash if Q_nash > 0 else 0.0) for p in players}
        hhi = sum(s * s for s in shares.values()) * 10_000

        # OPEC-specific stats
        if "OPEC" in players:
            pi_opec_nash = (P_nash - cost_map["OPEC"]) * nash["OPEC"]
            pi_opec_cartel = (P_cartel - cost_map["OPEC"]) * cartel["OPEC"]
            opec_nash.append(float(pi_opec_nash))
            opec_cartel.append(float(pi_opec_cartel))
            opec_share.append(float(shares["OPEC"]))
            shap = _generalised_shapley(players, cost_map, demand, cap_arg)
            opec_shap.append(float(shap.get("OPEC", 0.0)))
        else:
            opec_nash.append(0.0)
            opec_cartel.append(0.0)
            opec_share.append(0.0)
            opec_shap.append(0.0)

        n_vals.append(n)
        nash_prices.append(float(P_nash))
        nash_qs.append(float(Q_nash))
        cartel_prices.append(float(P_cartel))
        delta_stars.append(float(delta_b))
        hhis.append(float(hhi))
        player_lists.append(list(players))

    return NPlayerSweepResult(
        n_values=n_vals,
        nash_prices=nash_prices,
        nash_total_quantities=nash_qs,
        cartel_prices=cartel_prices,
        delta_star_binding=delta_stars,
        hhi_values=hhis,
        opec_profits_nash=opec_nash,
        opec_profits_cartel=opec_cartel,
        opec_shapley=opec_shap,
        opec_market_shares=opec_share,
        player_lists=player_lists,
    )


def n_player_comparison_table(sweep: NPlayerSweepResult) -> pd.DataFrame:
    """Return a tidy DataFrame summary of the full sweep."""
    return pd.DataFrame({
        "n": sweep.n_values,
        "players": [",".join(pl) for pl in sweep.player_lists],
        "nash_price": [round(v, 4) for v in sweep.nash_prices],
        "nash_total_q": [round(v, 4) for v in sweep.nash_total_quantities],
        "cartel_price": [round(v, 4) for v in sweep.cartel_prices],
        "delta_star_binding": [round(v, 4) for v in sweep.delta_star_binding],
        "hhi": [round(v, 2) for v in sweep.hhi_values],
        "opec_profit_nash":   [round(v, 4) for v in sweep.opec_profits_nash],
        "opec_profit_cartel": [round(v, 4) for v in sweep.opec_profits_cartel],
        "opec_shapley":       [round(v, 4) for v in sweep.opec_shapley],
        "opec_share_nash":    [round(v, 4) for v in sweep.opec_market_shares],
    })
