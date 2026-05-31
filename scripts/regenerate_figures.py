"""Regenerate selected thesis figures with updated formatting.

This script avoids re-running the full simulation pipeline.  It either
re-uses the CSV artefacts already in ``outputs/`` or re-runs the
specific lightweight pipeline step needed to rebuild a single figure.
Typical wall time is well under ten seconds.

Figures regenerated
-------------------
1. ``comparative_statics.png``         — two side-by-side panels with
                                         discrete markers (was a single
                                         dual-y-axis line plot where the
                                         two series were visually
                                         indistinguishable).
2. ``welfare_surplus_distribution.png``— distinct purple colour for the
                                         Consumers stratum (was the same
                                         blue as Producer-US, making the
                                         two indistinguishable).
3. ``correlated_eq_support_all.png``   — NEW: 1x3 panel showing the CE
                                         marginal recommendation
                                         distribution for every solved
                                         objective (max_welfare,
                                         max_joint_profit, max_min_profit).
                                         The original single-panel
                                         ``correlated_eq_support.png`` is
                                         left in place for backward
                                         compatibility.
4. ``repeated_dynamics_combined.png``  — NEW: single 2x2 figure
                                         consolidating the four
                                         near-duplicate myopic-convergence
                                         plots (per-player quantities,
                                         industry total, price, profits).
                                         The four originals are left in
                                         place.

Usage
-----
    python scripts/regenerate_figures.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

# Make the project root importable when invoked as `python scripts/...`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import default_simulation_params
from src.cooperation_punishment import cartel_quotas
from src.correlated_eq import ce_vs_nash_comparison
from src.plotting import (
    plot_comparative_statics,
    plot_correlated_eq_support_all,
    plot_repeated_dynamics_combined,
    plot_welfare_surplus_distribution,
)
from src.welfare import welfare_decomposition

OUTPUT_DIR = os.path.join(ROOT, "outputs")


def regen_comparative_statics() -> str:
    """Replot from outputs/comparative_statics.csv — no simulation needed."""
    csv_path = os.path.join(OUTPUT_DIR, "comparative_statics.csv")
    df = pd.read_csv(csv_path)
    # Filter to the demand-intercept sweep (column 'param' == 'a')
    if "param" in df.columns:
        df = df[df["param"] == "a"]
    results = [
        {
            "param_value": float(row.param_value),
            "price": float(row.price),
            "total_quantity": float(row.total_quantity),
        }
        for row in df.itertuples()
    ]
    out = os.path.join(OUTPUT_DIR, "comparative_statics.png")
    plot_comparative_statics(results, out)
    return out


def regen_welfare_surplus_distribution() -> str:
    """Recompute the per-structure welfare decomposition (fast: < 1s) and
    redraw the stacked-bar figure with the new consumer colour."""
    params = default_simulation_params()
    decomp = welfare_decomposition(
        params.players, params.demand, params.costs, params.capacities,
    )
    out = os.path.join(OUTPUT_DIR, "welfare_surplus_distribution.png")
    plot_welfare_surplus_distribution(decomp["decompositions"], params.players, out)
    return out


def regen_correlated_eq_support_all() -> str:
    """Re-solve the correlated-equilibrium LP under every objective and
    plot the recommendation distributions side by side."""
    params = default_simulation_params()
    cmp = ce_vs_nash_comparison(
        params.players, params.demand, params.costs, params.capacities,
        params.correlated_eq,
    )
    out = os.path.join(OUTPUT_DIR, "correlated_eq_support_all.png")
    plot_correlated_eq_support_all(cmp, out)
    return out


def regen_repeated_dynamics_combined() -> str:
    """Build the 2x2 combined myopic-dynamics figure from
    outputs/repeated_myopic.csv and outputs/static_equilibrium.csv."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "repeated_myopic.csv"))
    static_df = pd.read_csv(os.path.join(OUTPUT_DIR, "static_equilibrium.csv"))
    nash_row = static_df[static_df["model"] == "triopoly"].iloc[0]

    prices = df["P"].tolist()
    total_quantities = df["Q"].tolist()
    per_player_quantities = [
        {"US": r.q_US, "OPEC": r.q_OPEC, "RUS": r.q_RUS} for r in df.itertuples()
    ]
    per_player_profits = [
        {"US": r.profit_US, "OPEC": r.profit_OPEC, "RUS": r.profit_RUS}
        for r in df.itertuples()
    ]
    nash_price = float(nash_row["P"])
    nash_total_q = float(nash_row["Q"])
    nash_profits = {
        "US": float(nash_row["profit_US"]),
        "OPEC": float(nash_row["profit_OPEC"]),
        "RUS": float(nash_row["profit_RUS"]),
    }

    # Cartel benchmarks (re-derived; the static CSV does not store them)
    params = default_simulation_params()
    cartel = cartel_quotas(
        params.players, params.demand, params.costs, params.capacities,
    )
    cartel_profits = dict(cartel.quota_profits)

    out = os.path.join(OUTPUT_DIR, "repeated_dynamics_combined.png")
    plot_repeated_dynamics_combined(
        prices, total_quantities, per_player_quantities, per_player_profits,
        out,
        nash_price=nash_price, nash_total_q=nash_total_q,
        nash_profits=nash_profits, cartel_profits=cartel_profits,
    )
    return out


def main() -> None:
    print("Regenerating selected figures (no full simulation run)...")
    for label, fn in [
        ("comparative_statics", regen_comparative_statics),
        ("welfare_surplus_distribution", regen_welfare_surplus_distribution),
        ("correlated_eq_support_all", regen_correlated_eq_support_all),
        ("repeated_dynamics_combined", regen_repeated_dynamics_combined),
    ]:
        try:
            path = fn()
            print(f"  [ok] {label:32s} -> {os.path.relpath(path, ROOT)}")
        except Exception as exc:
            print(f"  [FAIL] {label:32s} -> {exc!r}")
            raise
    print("Done.")


if __name__ == "__main__":
    main()
