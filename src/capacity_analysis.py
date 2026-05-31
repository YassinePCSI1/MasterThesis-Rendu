"""Capacity-constraint activation analysis (Section B).

The :class:`CapacityParams` dataclass already encodes per-player capacity
limits ``cap_us = 30 mbd``, ``cap_opec = 40 mbd``, ``cap_rus = 35 mbd``,
but the entire baseline pipeline runs with ``capacities.enabled = False``.
This module re-runs the core equilibria with capacities **on** and
quantifies the differences:

* How much do capacity limits move the Nash price and Nash profits?
* Do they make the cartel quotas easier or harder to sustain (delta*)?
* Which players are constrained at Nash? At the cartel quota?
* What is the effect of changing OPEC's capacity (the dominant
  swing-producer) on the equilibrium?

The module reuses ``cournot_equilibrium``, ``cartel_quotas``,
``folk_theorem_delta_star`` and ``shapley_values`` from the existing
codebase — the *only* difference is the value of
``capacities.enabled``.

Outputs
-------
- :class:`CapacityComparisonResult` for the binary on/off comparison.
- :class:`CapacityOpecSweep` for the OPEC-capacity sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import CapacityParams, CapacitySweepParams, CostParams, DemandParams
from .coalition import folk_theorem_delta_star, shapley_values
from .cooperation_punishment import cartel_quotas
from .cournot_static import _cap_map, _cost_map, cournot_equilibrium
from .market_power import market_power_metrics


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CapacityComparisonResult:
    """Constrained vs unconstrained comparison of every key equilibrium."""

    unconstrained: Dict[str, Any]
    constrained: Dict[str, Any]
    binding_players_nash: List[str]
    binding_players_cartel: List[str]


@dataclass
class CapacityOpecSweep:
    """OPEC capacity sweep results — comparative statics on the dominant player."""

    cap_values: List[float]
    nash_prices: List[float]
    opec_profits: List[float]
    delta_star_binding: List[float]
    hhi_values: List[float]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _equilibrium_bundle(players: List[str],
                        demand: DemandParams,
                        costs: CostParams,
                        capacities: CapacityParams) -> Dict[str, Any]:
    """Compute Nash, cartel, folk-theorem delta* and Shapley for a given setting."""
    nash = cournot_equilibrium(players, demand, costs, capacities)
    cartel = cartel_quotas(players, demand, costs, capacities)
    folk = folk_theorem_delta_star(players, demand, costs, capacities)
    shap = shapley_values(players, demand, costs, capacities)
    mp = market_power_metrics(nash, costs)
    caps_map = _cap_map(capacities)
    return {
        "nash": nash,
        "cartel": cartel,
        "folk": folk,
        "shapley": shap,
        "hhi": mp.hhi,
        "caps_map": dict(caps_map),
    }


def binding_constraint_analysis(
    nash_quantities: Dict[str, float],
    cartel_quantities: Dict[str, float],
    caps_map: Dict[str, float],
    tol: float = 1e-6,
) -> Dict[str, List[str]]:
    """Identify which players hit their capacity ceiling at Nash and at cartel.

    A constraint is "binding" if the chosen output is within ``tol`` of the
    capacity ceiling.  Returns a dict with two lists.
    """
    binding_nash = [p for p, q in nash_quantities.items()
                    if q >= caps_map[p] - tol]
    binding_cartel = [p for p, q in cartel_quantities.items()
                      if q >= caps_map[p] - tol]
    return {"nash": binding_nash, "cartel": binding_cartel}


def run_constrained_vs_unconstrained(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
) -> CapacityComparisonResult:
    """Run the core equilibria twice — without and with capacity constraints."""
    caps_off = CapacityParams(
        enabled=False,
        cap_us=capacities.cap_us,
        cap_opec=capacities.cap_opec,
        cap_rus=capacities.cap_rus,
    )
    caps_on = CapacityParams(
        enabled=True,
        cap_us=capacities.cap_us,
        cap_opec=capacities.cap_opec,
        cap_rus=capacities.cap_rus,
    )
    unconstrained = _equilibrium_bundle(players, demand, costs, caps_off)
    constrained = _equilibrium_bundle(players, demand, costs, caps_on)

    binding = binding_constraint_analysis(
        constrained["nash"].quantities,
        constrained["cartel"].quotas,
        constrained["caps_map"],
    )

    return CapacityComparisonResult(
        unconstrained=unconstrained,
        constrained=constrained,
        binding_players_nash=binding["nash"],
        binding_players_cartel=binding["cartel"],
    )


def opec_capacity_sweep(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    sweep_params: CapacitySweepParams,
) -> CapacityOpecSweep:
    """Sweep OPEC capacity across a grid; record price, OPEC profit, delta* and HHI."""
    cap_values = list(sweep_params.opec_cap_values)
    nash_prices, opec_profits, delta_stars, hhis = [], [], [], []

    for cap_opec in cap_values:
        caps = CapacityParams(
            enabled=True,
            cap_us=capacities.cap_us,
            cap_opec=cap_opec,
            cap_rus=capacities.cap_rus,
        )
        nash = cournot_equilibrium(players, demand, costs, caps)
        folk = folk_theorem_delta_star(players, demand, costs, caps)
        mp = market_power_metrics(nash, costs)
        nash_prices.append(float(nash.price))
        opec_profits.append(float(nash.profits.get("OPEC", 0.0)))
        delta_stars.append(float(folk.delta_binding))
        hhis.append(float(mp.hhi))

    return CapacityOpecSweep(
        cap_values=cap_values,
        nash_prices=nash_prices,
        opec_profits=opec_profits,
        delta_star_binding=delta_stars,
        hhi_values=hhis,
    )


def comparison_to_dataframe(comp: CapacityComparisonResult,
                            players: List[str]) -> pd.DataFrame:
    """Long-form DataFrame of every metric for both regimes."""
    rows = []
    for label, bundle in [("unconstrained", comp.unconstrained),
                          ("constrained", comp.constrained)]:
        nash = bundle["nash"]
        cartel = bundle["cartel"]
        folk = bundle["folk"]
        row = {
            "regime": label,
            "nash_price": round(nash.price, 4),
            "nash_total_q": round(nash.total_quantity, 4),
            "cartel_price": round(cartel.quota_price, 4),
            "cartel_total_q": round(cartel.total_output, 4),
            "delta_star_binding": round(folk.delta_binding, 4),
            "binding_player": folk.binding_player,
            "hhi": round(bundle["hhi"], 2),
        }
        for p in players:
            row[f"nash_q_{p}"] = round(nash.quantities[p], 4)
            row[f"nash_profit_{p}"] = round(nash.profits[p], 4)
            row[f"cartel_q_{p}"] = round(cartel.quotas[p], 4)
            row[f"delta_star_{p}"] = round(folk.delta_star[p], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def opec_sweep_to_dataframe(sweep: CapacityOpecSweep) -> pd.DataFrame:
    return pd.DataFrame({
        "opec_cap": sweep.cap_values,
        "nash_price": [round(v, 4) for v in sweep.nash_prices],
        "opec_profit": [round(v, 4) for v in sweep.opec_profits],
        "delta_star": [round(v, 4) for v in sweep.delta_star_binding],
        "hhi": [round(v, 2) for v in sweep.hhi_values],
    })
