"""Differentiated-products Bertrand competition (Section A — robustness check).

The entire core thesis uses **Cournot (quantity) competition** as the
microfoundation.  This module asks whether the qualitative conclusions
(OPEC dominance, sustainability of cooperation, gap between Nash and
joint-profit outcomes) survive when firms compete on **price** instead of
quantity.

Why differentiated rather than homogeneous Bertrand?
----------------------------------------------------
With perfectly substitutable goods, classical Bertrand collapses to
marginal-cost pricing (the Bertrand paradox) and the oligopoly question
becomes degenerate.  Real crude oil is not homogeneous: WTI, Brent, Urals
and the various OPEC basket grades differ in sulfur content, API gravity,
location and quality, so producers face **partially substitutable**
demand.  We model this with a linear-quasi-symmetric demand system
parametrised by σ ∈ [0, 1] where σ = 0 = independent monopolies and
σ → 1 = the homogeneous-Bertrand limit.

Demand system
-------------
For player i with rivals R = N \\ {i},

    q_i(p) = max(0, base_demand − p_i + σ · ( p̄_R − p_i ))

where ``p̄_R = mean(p_R)`` is the average rival price.  The first
``− p_i`` term is the own-price effect; the σ-term is the
substitutability effect — when rivals charge more on average, demand
shifts towards the own product.  At σ = 0 the cross-price terms vanish
(each player is a monopolist on its own demand); at σ → 1 the demand for
i is highly sensitive to relative prices, and equilibrium prices collapse
toward marginal cost.

Each player solves ``max π_i = (p_i − c_i) · q_i(p)``; the Nash
equilibrium is found by iterated best-response.  The joint-profit
("cooperative") problem maximises Σ π_i over the price vector.

Outputs
-------
- :class:`BertrandResult`              : single-equilibrium summary.
- :class:`BertrandSweepResult`         : Nash + cooperative results across
                                         a grid of σ values.
- :class:`BertrandVsCournotComparison` : side-by-side Nash comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

from .config import BertrandParams, CapacityParams, CostParams, DemandParams
from .cournot_static import _cost_map, cournot_equilibrium
from .cooperation_punishment import cartel_quotas


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BertrandResult:
    """Equilibrium of a differentiated-products Bertrand game."""

    sigma: float
    prices: Dict[str, float]
    quantities: Dict[str, float]
    profits: Dict[str, float]
    total_quantity: float
    average_price: float
    consumer_surplus: float
    total_welfare: float


@dataclass
class BertrandSweepResult:
    """Sweep of Bertrand Nash and cooperative outcomes across σ values."""

    sigma_values: List[float]
    nash_results: List[BertrandResult]
    coop_results: List[BertrandResult]


@dataclass
class BertrandVsCournotComparison:
    """Side-by-side Nash comparison: Cournot vs Bertrand at the calibrated σ."""

    cournot_nash: Dict[str, float]   # {price, total_q, profits per player, CS, welfare}
    bertrand_nash: BertrandResult
    cournot_cartel_price: float
    bertrand_cooperative: BertrandResult


# ---------------------------------------------------------------------------
# Demand and best-response helpers
# ---------------------------------------------------------------------------

def _bertrand_demand(p: Dict[str, float],
                     player: str,
                     sigma: float,
                     base_demand: float) -> float:
    """Linear differentiated demand for a single player.

    q_i = max(0, base_demand − p_i + σ · ( mean(p_-i) − p_i ))
    """
    rivals = [k for k in p if k != player]
    if rivals:
        p_avg_rivals = float(np.mean([p[k] for k in rivals]))
    else:
        p_avg_rivals = p[player]
    q = base_demand - p[player] + sigma * (p_avg_rivals - p[player])
    return float(max(0.0, q))


def _profit(p: Dict[str, float],
            player: str,
            cost_map: Dict[str, float],
            sigma: float,
            base_demand: float) -> float:
    q = _bertrand_demand(p, player, sigma, base_demand)
    return (p[player] - cost_map[player]) * q


def bertrand_best_response(p_rivals: Dict[str, float],
                           c_i: float,
                           sigma: float,
                           base_demand: float) -> float:
    """Closed-form best-response price for player i.

    Differentiating π_i = (p_i − c_i) · (base_demand − p_i + σ·(p̄_R − p_i))
    with respect to p_i and setting to zero yields

        p_i* = ½ · ( base_demand + σ · p̄_R + (1 + σ) · c_i ) / (1 + σ)

    which simplifies to

        p_i* = ( base_demand + σ · p̄_R + (1 + σ) · c_i ) / ( 2 · (1 + σ) ).
    """
    if not p_rivals:
        # Pure monopoly
        return 0.5 * (base_demand + c_i)
    p_avg_rivals = float(np.mean(list(p_rivals.values())))
    return float(
        (base_demand + sigma * p_avg_rivals + (1.0 + sigma) * c_i)
        / (2.0 * (1.0 + sigma))
    )


# ---------------------------------------------------------------------------
# Equilibria
# ---------------------------------------------------------------------------

def _build_result(prices: Dict[str, float],
                  cost_map: Dict[str, float],
                  sigma: float,
                  base_demand: float,
                  demand: DemandParams) -> BertrandResult:
    quantities = {p: _bertrand_demand(prices, p, sigma, base_demand) for p in prices}
    profits = {p: (prices[p] - cost_map[p]) * quantities[p] for p in prices}
    Q = float(sum(quantities.values()))
    avg_p = float(np.mean(list(prices.values())))
    # Approximate consumer surplus on the *aggregate* linear demand reading: the
    # area between the residual demand intercept (base_demand × n_players) and
    # the average price, evaluated at total quantity Q.  This is a coarse
    # measure but is comparable across σ values.
    n = len(prices)
    cs = max(0.0, 0.5 * (base_demand * n - avg_p) * Q)
    return BertrandResult(
        sigma=sigma,
        prices=dict(prices),
        quantities=quantities,
        profits=profits,
        total_quantity=Q,
        average_price=avg_p,
        consumer_surplus=cs,
        total_welfare=cs + sum(profits.values()),
    )


def bertrand_nash(players: List[str],
                  costs: CostParams,
                  bertrand_params: BertrandParams,
                  demand: DemandParams) -> BertrandResult:
    """Iterated best-response Bertrand-Nash equilibrium.

    Starts every price at the marginal cost and applies the closed-form
    best-response repeatedly until prices converge (max change < 1e-9).
    Convergence is geometric for σ < 1; at σ → 1 the equilibrium tends to
    marginal-cost pricing.
    """
    cost_map = _cost_map(costs)
    sigma = bertrand_params.sigma
    base = bertrand_params.base_demand_scale

    p = {pl: cost_map[pl] for pl in players}

    for _ in range(2000):
        p_new = dict(p)
        for pl in players:
            rivals = {k: p[k] for k in players if k != pl}
            p_new[pl] = bertrand_best_response(rivals, cost_map[pl], sigma, base)
        if max(abs(p_new[pl] - p[pl]) for pl in players) < 1e-9:
            p = p_new
            break
        p = p_new

    return _build_result(p, cost_map, sigma, base, demand)


def bertrand_cooperative(players: List[str],
                         costs: CostParams,
                         bertrand_params: BertrandParams,
                         demand: DemandParams) -> BertrandResult:
    """Joint-profit-maximising prices under Bertrand competition.

    Solves max Σ π_i over the price vector with a numerical optimiser.
    The cooperative price is *strictly above* the Bertrand-Nash price
    whenever σ > 0 — this is the Bertrand analogue of the cartel premium.
    """
    cost_map = _cost_map(costs)
    sigma = bertrand_params.sigma
    base = bertrand_params.base_demand_scale

    def neg_total_profit(p_vec: np.ndarray) -> float:
        prices = {pl: float(p_vec[i]) for i, pl in enumerate(players)}
        return -sum(
            (prices[pl] - cost_map[pl]) * _bertrand_demand(prices, pl, sigma, base)
            for pl in players
        )

    # Sensible starting point: 25% above the cost
    x0 = np.array([1.25 * cost_map[pl] + 0.5 * base for pl in players])
    bounds = [(cost_map[pl], base * 1.5) for pl in players]
    res = minimize(neg_total_profit, x0=x0, bounds=bounds, method="L-BFGS-B")

    p = {pl: float(res.x[i]) for i, pl in enumerate(players)}
    return _build_result(p, cost_map, sigma, base, demand)


# ---------------------------------------------------------------------------
# Sweeps and comparisons
# ---------------------------------------------------------------------------

def bertrand_sigma_sweep(players: List[str],
                         costs: CostParams,
                         bertrand_params: BertrandParams,
                         demand: DemandParams) -> BertrandSweepResult:
    """Compute Nash and cooperative equilibria along the substitutability grid σ."""
    nash_results: List[BertrandResult] = []
    coop_results: List[BertrandResult] = []
    for sigma in bertrand_params.sigma_sweep:
        bp = BertrandParams(
            sigma=sigma,
            sigma_sweep=bertrand_params.sigma_sweep,
            base_demand_scale=bertrand_params.base_demand_scale,
        )
        nash_results.append(bertrand_nash(players, costs, bp, demand))
        coop_results.append(bertrand_cooperative(players, costs, bp, demand))
    return BertrandSweepResult(
        sigma_values=list(bertrand_params.sigma_sweep),
        nash_results=nash_results,
        coop_results=coop_results,
    )


def bertrand_vs_cournot_comparison(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    bertrand_params: BertrandParams,
) -> BertrandVsCournotComparison:
    """Build a side-by-side Cournot vs Bertrand comparison at the calibrated σ."""
    nash_cournot = cournot_equilibrium(players, demand, costs, capacities)
    cartel = cartel_quotas(players, demand, costs, capacities)
    nash_bertrand = bertrand_nash(players, costs, bertrand_params, demand)
    coop_bertrand = bertrand_cooperative(players, costs, bertrand_params, demand)

    cournot_nash = {
        "price": float(nash_cournot.price),
        "total_q": float(nash_cournot.total_quantity),
        "consumer_surplus": float(nash_cournot.consumer_surplus),
        "total_welfare": float(nash_cournot.total_welfare),
        **{f"profit_{p}": float(nash_cournot.profits[p]) for p in players},
    }

    return BertrandVsCournotComparison(
        cournot_nash=cournot_nash,
        bertrand_nash=nash_bertrand,
        cournot_cartel_price=float(cartel.quota_price),
        bertrand_cooperative=coop_bertrand,
    )
