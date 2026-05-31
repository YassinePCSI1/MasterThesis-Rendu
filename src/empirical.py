"""Empirical validation against historical price-war episodes (Section F).

This module grounds the theoretical model in three well-documented
structural breaks in the global oil market:

* **1985 OPEC price war** — Saudi Arabia "opens the taps" to discipline
  internal cheaters; Brent falls from ~28 → ~10 $/bbl.
* **2014 OPEC market-share war** — OPEC defends share against US shale;
  Brent falls from ~110 → ~26 $/bbl.
* **2020 Russia–Saudi price war** — collapse of the OPEC+ deal during
  the COVID demand shock; Brent falls from ~65 → ~20 $/bbl.

The data are *hardcoded representative summary statistics* (no live
APIs).  We compare the magnitude and direction of the historical price
move against the model's own prediction of a Nash-vs-cooperative price
gap (the punishment-phase price under grim trigger).

A coarse "fit score" reports the fraction of episodes correctly matched
on three criteria (direction, magnitude within ±30 %, mechanism
qualitatively right).  The intent is *not* a forecasting test but a
sanity check that the modelled mechanisms — Folk-theorem punishment,
Stackelberg shifts, demand shocks — line up with reality at the level
of qualitative dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from .config import (
    CapacityParams,
    CostParams,
    DemandParams,
    EmpiricalParams,
    RepeatedGameParams,
    StackelbergParams,
)
from .cooperation_punishment import cartel_quotas
from .cournot_static import cournot_equilibrium
from .stackelberg import stackelberg_equilibrium


# ---------------------------------------------------------------------------
# Hardcoded historical data
# ---------------------------------------------------------------------------

HISTORICAL_EPISODES: Dict[str, Dict[str, Any]] = {
    "1985_opec_price_war": {
        "description":
            "OPEC abandons quotas; Saudi Arabia floods market to punish cheaters",
        "pre_price": 28.0,
        "trough_price": 10.0,
        "recovery_price": 18.0,
        "duration_quarters": 8,
        "trigger": "OPEC members exceeded quotas; Saudi opened taps",
        "model_prediction": "punishment_regime",
    },
    "2014_opec_market_share": {
        "description":
            "OPEC maintains output despite US shale surge; prices crash",
        "pre_price": 110.0,
        "trough_price": 26.0,
        "recovery_price": 55.0,
        "duration_quarters": 10,
        "trigger":
            "US shale boom eroded OPEC market share; OPEC chose volume over price",
        "model_prediction": "stackelberg_shift",
    },
    "2020_russia_saudi_price_war": {
        "description":
            "Russia refuses OPEC+ cuts during COVID demand collapse",
        "pre_price": 65.0,
        "trough_price": 20.0,
        "recovery_price": 45.0,
        "duration_quarters": 3,
        "trigger": "COVID demand shock + Russia-Saudi disagreement on cuts",
        "model_prediction": "demand_shock_plus_punishment",
    },
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmpiricalValidationResult:
    """Per-episode comparison plus an aggregate fit score."""

    episodes: List[Dict[str, Any]]
    overall_fit_score: float
    mechanism_match: Dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model side: derive cooperative- and punishment-phase prices
# ---------------------------------------------------------------------------

def model_predicted_price_war(players: List[str],
                              demand: DemandParams,
                              costs: CostParams,
                              capacities: CapacityParams,
                              ) -> Dict[str, float]:
    """Model's own punishment-phase price (= Nash) and cooperation-phase
    price (= cartel) from the calibrated three-player Cournot game.

    Under grim trigger the *punishment phase* reverts to Nash forever;
    so the modelled price drop during a war is

        Δ_model = P_coop − P_nash.

    For the 2014 episode we additionally compute the Stackelberg-OPEC
    price as the "competitive surrender" price that OPEC could enforce.
    """
    nash = cournot_equilibrium(players, demand, costs, capacities)
    cartel = cartel_quotas(players, demand, costs, capacities)
    stack_opec = stackelberg_equilibrium("OPEC", players, demand, costs,
                                         capacities, nash_result=nash)
    return {
        "nash_price": float(nash.price),
        "cartel_price": float(cartel.quota_price),
        "stackelberg_opec_price": float(stack_opec.price),
    }


def compare_price_war_depth(historical: Dict[str, Any],
                            model_coop_price: float,
                            model_nash_price: float) -> Dict[str, float]:
    """Compare the *normalised* price drops between history and model.

    Both are expressed as a fraction of the pre-war price so that we can
    compare a 1985-style ~$18/bbl drop with a 2014-style ~$84/bbl drop
    on the same scale.  The model's drop is normalised to the cooperative
    price as the "pre-war" reference.
    """
    hist_drop = historical["pre_price"] - historical["trough_price"]
    hist_drop_pct = hist_drop / historical["pre_price"] if historical["pre_price"] > 0 else 0.0
    model_drop = max(0.0, model_coop_price - model_nash_price)
    model_drop_pct = model_drop / model_coop_price if model_coop_price > 0 else 0.0
    return {
        "historical_drop": float(hist_drop),
        "historical_drop_pct": float(hist_drop_pct),
        "model_drop": float(model_drop),
        "model_drop_pct": float(model_drop_pct),
    }


def compare_duration(historical_quarters: int,
                     model_punishment_length: int) -> Dict[str, Any]:
    """Match historical and modelled punishment durations."""
    hist = int(historical_quarters)
    model = int(model_punishment_length)
    return {
        "historical_quarters": hist,
        "model_punishment_length": model,
        "duration_match": abs(hist - model) <= max(2, hist // 2),
    }


def model_fit_score(episodes_summary: List[Dict[str, Any]]) -> float:
    """Coarse qualitative fit score: 0 = no match, 1 = perfect.

    For each episode we award 1/3 for direction, 1/3 for magnitude
    (within 30 % of the historical drop %), and 1/3 for mechanism
    (correctly matched mechanism flag).  The episode's overall score is
    the sum of those three; the aggregate score is the mean across
    episodes.
    """
    if not episodes_summary:
        return 0.0
    scores = []
    for e in episodes_summary:
        s = 0.0
        if e["direction"]:
            s += 1 / 3
        if e["magnitude_30pct"]:
            s += 1 / 3
        if e["mechanism"]:
            s += 1 / 3
        scores.append(s)
    return float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_empirical_validation(players: List[str],
                             demand: DemandParams,
                             costs: CostParams,
                             capacities: CapacityParams,
                             repeated: RepeatedGameParams,
                             stackelberg_params: StackelbergParams,
                             selected_episodes: List[str],
                             ) -> EmpiricalValidationResult:
    """Run the full empirical validation for the requested episodes."""
    model = model_predicted_price_war(players, demand, costs, capacities)

    rows: List[Dict[str, Any]] = []
    mechanism_match: Dict[str, bool] = {}

    for key in selected_episodes:
        if key not in HISTORICAL_EPISODES:
            continue
        hist = HISTORICAL_EPISODES[key]
        depth = compare_price_war_depth(
            hist, model["cartel_price"], model["nash_price"],
        )
        dur = compare_duration(hist["duration_quarters"],
                               repeated.punishment_length)

        # Direction match: model predicts Nash < cartel (price falls during war)
        direction_match = depth["model_drop"] > 0 and depth["historical_drop"] > 0
        magnitude_match = (
            abs(depth["model_drop_pct"] - depth["historical_drop_pct"])
            / max(depth["historical_drop_pct"], 1e-6) <= 0.30
        )
        # Mechanism match: each historical episode names an expected mechanism;
        # we award the match if the model has a structure consistent with it.
        mech = hist["model_prediction"]
        if mech == "punishment_regime":
            mech_match = direction_match  # Folk-theorem punishment is on
        elif mech == "stackelberg_shift":
            # The 2014 episode is best matched by a regime shift between
            # Stackelberg-OPEC (when OPEC was leader) and Cournot-Nash;
            # we count it as matched if those two prices straddle the trough.
            sk = model["stackelberg_opec_price"]
            mech_match = (sk > model["nash_price"]
                          and hist["pre_price"] >= sk
                          and hist["trough_price"] >= model["nash_price"])
        else:
            mech_match = direction_match
        mechanism_match[key] = bool(mech_match)

        overall = direction_match and magnitude_match and mech_match
        rows.append({
            "episode": key,
            "description": hist["description"],
            "trigger": hist["trigger"],
            "expected_mechanism": mech,
            "historical_pre": hist["pre_price"],
            "historical_trough": hist["trough_price"],
            "historical_recovery": hist["recovery_price"],
            "historical_drop": depth["historical_drop"],
            "historical_drop_pct": depth["historical_drop_pct"],
            "historical_duration_quarters": hist["duration_quarters"],
            "model_pre": model["cartel_price"],
            "model_trough": model["nash_price"],
            "model_drop": depth["model_drop"],
            "model_drop_pct": depth["model_drop_pct"],
            "model_punishment_length": repeated.punishment_length,
            "duration_match": dur["duration_match"],
            "direction": direction_match,
            "magnitude_30pct": magnitude_match,
            "mechanism": mech_match,
            "overall": overall,
        })

    fit = model_fit_score(rows)
    return EmpiricalValidationResult(
        episodes=rows,
        overall_fit_score=fit,
        mechanism_match=mechanism_match,
    )


def validation_to_dataframe(result: EmpiricalValidationResult) -> pd.DataFrame:
    """Convert the validation result to a CSV-friendly DataFrame."""
    if not result.episodes:
        return pd.DataFrame([])
    return pd.DataFrame([{
        "episode": e["episode"],
        "description": e["description"],
        "trigger": e["trigger"],
        "expected_mechanism": e["expected_mechanism"],
        "historical_pre": round(e["historical_pre"], 2),
        "historical_trough": round(e["historical_trough"], 2),
        "historical_drop": round(e["historical_drop"], 2),
        "historical_drop_pct": round(e["historical_drop_pct"], 4),
        "historical_duration_quarters": e["historical_duration_quarters"],
        "model_pre": round(e["model_pre"], 2),
        "model_trough": round(e["model_trough"], 2),
        "model_drop": round(e["model_drop"], 2),
        "model_drop_pct": round(e["model_drop_pct"], 4),
        "model_punishment_length": e["model_punishment_length"],
        "duration_match": e["duration_match"],
        "direction_match": e["direction"],
        "magnitude_30pct_match": e["magnitude_30pct"],
        "mechanism_match": e["mechanism"],
        "overall_match": e["overall"],
    } for e in result.episodes] + [{
        "episode": "OVERALL",
        "description": "Aggregate qualitative fit score (0–1)",
        "trigger": "—",
        "expected_mechanism": "—",
        "historical_pre": "",
        "historical_trough": "",
        "historical_drop": "",
        "historical_drop_pct": "",
        "historical_duration_quarters": "",
        "model_pre": "",
        "model_trough": "",
        "model_drop": "",
        "model_drop_pct": "",
        "model_punishment_length": "",
        "duration_match": "",
        "direction_match": "",
        "magnitude_30pct_match": "",
        "mechanism_match": "",
        "overall_match": round(result.overall_fit_score, 4),
    }])
