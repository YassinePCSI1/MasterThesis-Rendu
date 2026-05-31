"""Welfare and deadweight-loss analysis (Section C).

Provides the *normative* layer of the thesis.  We quantify how much total
social welfare each market structure leaves on the table relative to the
perfectly-competitive benchmark, and how that gap interacts with carbon
taxation.

Definitions used throughout
---------------------------
For linear demand ``P(Q) = a − bQ``:

* Consumer surplus  CS = ½ · (a − P) · Q
* Producer surplus  PS = Σᵢ (P − cᵢ) · qᵢ
* Total welfare     W  = CS + PS
* Deadweight loss   DWL = W_competitive − W_actual
                       = ½ · b · (Q_comp − Q_actual)²
  (the standard linear-demand triangle).

Reference market structures
---------------------------
1.  **Perfect competition** — every firm with marginal cost < a produces
    until the price equals its marginal cost.  Under heterogeneous costs
    we use the cheapest producer (OPEC) as the marginal supplier; in
    practice for our calibration that yields ``P = c_OPEC = 20`` and a
    very high total Q.
2.  **Cournot Nash** (existing model).
3.  **Stackelberg leader = OPEC** (existing model).
4.  **Cartel quotas** (joint-profit maximisation, existing model).

Carbon-tax interaction
----------------------
Adding a per-barrel tax τ to every producer's marginal cost shifts the
supply curve up.  We recompute Nash and cartel under each τ and report
how the *collusion premium* (cartel profit ÷ Nash profit − 1) evolves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import CapacityParams, CostParams, DemandParams, WelfareParams
from .cooperation_punishment import cartel_quotas
from .cournot_static import CournotResult, _cap_map, _cost_map, cournot_equilibrium
from .demand import consumer_surplus, price_from_quantity
from .stackelberg import stackelberg_equilibrium


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompetitiveResult:
    """Perfect-competition benchmark."""

    quantities: Dict[str, float]
    price: float
    total_quantity: float
    profits: Dict[str, float]
    consumer_surplus: float
    total_welfare: float


@dataclass
class WelfareDecomposition:
    """Welfare decomposition for a single market structure."""

    structure_label: str
    price: float
    total_quantity: float
    consumer_surplus: float
    producer_surplus: float
    deadweight_loss: float
    total_welfare: float
    profits_by_player: Dict[str, float] = field(default_factory=dict)


@dataclass
class CarbonTaxResult:
    """Carbon-tax point on the Nash and cartel equilibrium."""

    tax: float
    nash_price: float
    cartel_price: float
    nash_total_q: float
    cartel_total_q: float
    nash_total_profit: float
    cartel_total_profit: float
    dwl_nash: float
    dwl_cartel: float
    collusion_premium_pct: float


# ---------------------------------------------------------------------------
# Competitive benchmark
# ---------------------------------------------------------------------------

def competitive_equilibrium(players: List[str],
                            demand: DemandParams,
                            costs: CostParams,
                            capacities: CapacityParams,
                            tax: float = 0.0) -> CompetitiveResult:
    """Perfect-competition benchmark with heterogeneous costs.

    Order producers by cost; the marginal supplier sets the price at its
    marginal cost.  We assume the lowest-cost producer can in principle
    expand to clear the market, but if capacities are enabled the higher-
    cost producers are also activated as soon as the price exceeds their
    cost.

    The implementation is simple and robust: identify the lowest-cost
    producer with cost ``c_min`` (after adding the tax), set ``P = c_min``,
    aggregate demand at that price gives total Q, and we then split the
    quantity by capacities (lowest-cost first, fill its capacity, then
    next).  When capacities are disabled the entire output is assigned to
    the cheapest producer (yielding zero profit at the margin).
    """
    cost_map = {p: _cost_map(costs)[p] + tax for p in players}
    cap_map = _cap_map(capacities)

    min_cost = min(cost_map.values())
    P = min_cost
    Q_total = max(0.0, (demand.a - P) / demand.b)

    # Allocate Q_total to producers in order of cost
    sorted_players = sorted(players, key=lambda p: cost_map[p])
    quantities: Dict[str, float] = {p: 0.0 for p in players}
    remaining = Q_total
    for p in sorted_players:
        if remaining <= 0:
            break
        cap_p = cap_map[p] if capacities.enabled else float("inf")
        q_p = min(remaining, cap_p)
        # Activate only producers whose cost <= P
        if cost_map[p] > P + 1e-9:
            break
        quantities[p] = q_p
        remaining -= q_p

    # If capacities are tight, residual demand is unmet — price rises to clear
    if capacities.enabled and remaining > 1e-9:
        # We need to lift the price so that activated capacity equals demand.
        # Iterate over sorted players, activating each at price = c_p, and
        # check whether available capacity covers demand at that price.
        for p in sorted_players:
            P_try = cost_map[p]
            Q_try = max(0.0, (demand.a - P_try) / demand.b)
            cum_cap = sum(cap_map[k] for k in sorted_players
                          if cost_map[k] <= P_try + 1e-9)
            if cum_cap >= Q_try - 1e-9:
                P = P_try
                quantities = {pl: 0.0 for pl in players}
                left = Q_try
                for k in sorted_players:
                    if cost_map[k] > P_try + 1e-9 or left <= 0:
                        break
                    q_k = min(left, cap_map[k])
                    quantities[k] = q_k
                    left -= q_k
                Q_total = sum(quantities.values())
                break

    profits = {p: (P - cost_map[p]) * quantities[p] for p in players}
    cs = consumer_surplus(Q_total, P, demand)
    total_welfare = cs + sum(profits.values())

    return CompetitiveResult(
        quantities=quantities,
        price=P,
        total_quantity=Q_total,
        profits=profits,
        consumer_surplus=cs,
        total_welfare=total_welfare,
    )


# ---------------------------------------------------------------------------
# Deadweight loss and decomposition
# ---------------------------------------------------------------------------

def deadweight_loss(actual_total_q: float,
                    competitive: CompetitiveResult,
                    demand: DemandParams) -> float:
    """Linear-demand DWL: ½ · b · (Q_comp − Q_actual)²."""
    deficit = max(0.0, competitive.total_quantity - actual_total_q)
    return 0.5 * demand.b * deficit * deficit


def _make_decomposition(label: str,
                        price: float,
                        total_q: float,
                        profits: Dict[str, float],
                        competitive: CompetitiveResult,
                        demand: DemandParams) -> WelfareDecomposition:
    cs = consumer_surplus(total_q, price, demand)
    ps = float(sum(profits.values()))
    dwl = deadweight_loss(total_q, competitive, demand)
    return WelfareDecomposition(
        structure_label=label,
        price=float(price),
        total_quantity=float(total_q),
        consumer_surplus=float(cs),
        producer_surplus=float(ps),
        deadweight_loss=float(dwl),
        total_welfare=float(cs + ps),
        profits_by_player={p: float(v) for p, v in profits.items()},
    )


def welfare_decomposition(players: List[str],
                          demand: DemandParams,
                          costs: CostParams,
                          capacities: CapacityParams) -> Dict[str, Any]:
    """Compute welfare decomposition for every market structure.

    Returns a dict with keys ``competitive``, ``decompositions`` (a list
    of :class:`WelfareDecomposition` objects, one per structure).
    """
    comp = competitive_equilibrium(players, demand, costs, capacities)
    nash = cournot_equilibrium(players, demand, costs, capacities)
    cartel = cartel_quotas(players, demand, costs, capacities)

    decompositions: List[WelfareDecomposition] = []
    decompositions.append(WelfareDecomposition(
        structure_label="Competitive",
        price=float(comp.price),
        total_quantity=float(comp.total_quantity),
        consumer_surplus=float(comp.consumer_surplus),
        producer_surplus=float(sum(comp.profits.values())),
        deadweight_loss=0.0,
        total_welfare=float(comp.total_welfare),
        profits_by_player={p: float(v) for p, v in comp.profits.items()},
    ))
    decompositions.append(_make_decomposition(
        "Nash", nash.price, nash.total_quantity, nash.profits, comp, demand,
    ))

    # Stackelberg with each leader
    for leader in players:
        sk = stackelberg_equilibrium(leader, players, demand, costs, capacities,
                                     nash_result=nash)
        decompositions.append(_make_decomposition(
            f"Stack-{leader}", sk.price, sk.total_quantity, sk.profits, comp, demand,
        ))

    decompositions.append(_make_decomposition(
        "Cartel", cartel.quota_price, cartel.total_output,
        cartel.quota_profits, comp, demand,
    ))

    return {"competitive": comp, "decompositions": decompositions}


# ---------------------------------------------------------------------------
# Carbon-tax interaction
# ---------------------------------------------------------------------------

def carbon_tax_interaction(players: List[str],
                           demand: DemandParams,
                           costs: CostParams,
                           capacities: CapacityParams,
                           tax_values: List[float]) -> List[CarbonTaxResult]:
    """For each tax level, recompute Nash and cartel and the collusion premium."""
    results: List[CarbonTaxResult] = []
    for tax in tax_values:
        taxed_costs = CostParams(
            c_us=costs.c_us + tax,
            c_opec=costs.c_opec + tax,
            c_rus=costs.c_rus + tax,
        )
        nash = cournot_equilibrium(players, demand, taxed_costs, capacities)
        cartel = cartel_quotas(players, demand, taxed_costs, capacities)
        comp = competitive_equilibrium(players, demand, costs, capacities, tax=tax)

        nash_total_profit = float(sum(nash.profits.values()))
        cartel_total_profit = float(sum(cartel.quota_profits.values()))
        premium = (
            (cartel_total_profit / nash_total_profit - 1.0) * 100.0
            if nash_total_profit > 1e-9
            else 0.0
        )
        results.append(CarbonTaxResult(
            tax=float(tax),
            nash_price=float(nash.price),
            cartel_price=float(cartel.quota_price),
            nash_total_q=float(nash.total_quantity),
            cartel_total_q=float(cartel.total_output),
            nash_total_profit=nash_total_profit,
            cartel_total_profit=cartel_total_profit,
            dwl_nash=float(deadweight_loss(nash.total_quantity, comp, demand)),
            dwl_cartel=float(deadweight_loss(cartel.total_output, comp, demand)),
            collusion_premium_pct=float(premium),
        ))
    return results


# ---------------------------------------------------------------------------
# Lightweight wrappers used by orchestrator/plots
# ---------------------------------------------------------------------------

@dataclass
class _CarbonResultLite:
    """Compatibility wrapper used by the carbon-tax plot, which expects
    ``.nash.price`` etc."""

    tax: float
    nash: object
    cartel: object
    dwl_nash: float
    dwl_cartel: float
    collusion_premium_pct: float


def carbon_results_to_lite(results: List[CarbonTaxResult]) -> List[_CarbonResultLite]:
    """Build the lite wrapper consumed by the plotting helper."""
    out = []
    for r in results:
        nash_obj = type("nash_obj", (), {"price": r.nash_price})()
        cartel_obj = type("cartel_obj", (), {"quota_price": r.cartel_price})()
        out.append(_CarbonResultLite(
            tax=r.tax,
            nash=nash_obj,
            cartel=cartel_obj,
            dwl_nash=r.dwl_nash,
            dwl_cartel=r.dwl_cartel,
            collusion_premium_pct=r.collusion_premium_pct,
        ))
    return out


def decomposition_to_dataframe(decompositions: List[WelfareDecomposition],
                               players: List[str]) -> pd.DataFrame:
    rows = []
    for d in decompositions:
        row = {
            "structure": d.structure_label,
            "price": round(d.price, 4),
            "total_quantity": round(d.total_quantity, 4),
            "consumer_surplus": round(d.consumer_surplus, 4),
            "producer_surplus": round(d.producer_surplus, 4),
            "deadweight_loss": round(d.deadweight_loss, 4),
            "total_welfare": round(d.total_welfare, 4),
        }
        for p in players:
            row[f"profit_{p}"] = round(d.profits_by_player.get(p, 0.0), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def carbon_results_to_dataframe(results: List[CarbonTaxResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "tax": round(r.tax, 4),
        "nash_price": round(r.nash_price, 4),
        "cartel_price": round(r.cartel_price, 4),
        "nash_total_q": round(r.nash_total_q, 4),
        "cartel_total_q": round(r.cartel_total_q, 4),
        "nash_total_profit": round(r.nash_total_profit, 4),
        "cartel_total_profit": round(r.cartel_total_profit, 4),
        "dwl_nash": round(r.dwl_nash, 4),
        "dwl_cartel": round(r.dwl_cartel, 4),
        "collusion_premium_pct": round(r.collusion_premium_pct, 4),
    } for r in results])
