"""Re-run the MARL forced-deviation stress-test and compute the mechanical
vs.\ strategic decomposition of the observed price drop, in line with the
pre-registered criterion of report.md s 18.6.

Outputs
-------
outputs/marl_forced_deviation.csv          aggregated (overwritten)
outputs/marl_forced_deviation_decomp.csv   per-window per-player quantities
                                           + mechanical / strategic shares
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import default_simulation_params                # noqa: E402
from src.cournot_static import cournot_equilibrium              # noqa: E402
from src.cooperation_punishment import cartel_quotas            # noqa: E402
from src.rl_multiagent import run_marl_forced_deviation         # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override MultiAgentRLParams.episodes")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=700)
    args = parser.parse_args()

    params = default_simulation_params()
    if args.episodes is None:
        frac = float(params.multi_rl.stress_episodes_fraction)
        episodes = max(50, int(round(params.multi_rl.episodes * frac)))
    else:
        episodes = int(args.episodes)
    stress_params = replace(params.multi_rl, episodes=episodes)
    n_seeds = int(args.n_seeds)

    nash = cournot_equilibrium(
        params.players, params.demand, params.costs, params.capacities,
    )
    cartel = cartel_quotas(
        params.players, params.demand, params.costs, params.capacities,
    )
    nash_price = float(nash.price)
    cartel_price = float(cartel.quota_price)

    # Mirror the stress-test budget conventions from simulations.run_marl_stress_tests
    total_steps = int(stress_params.episodes * stress_params.steps_per_episode)
    deviation_start = int(0.75 * total_steps)
    deviation_duration = max(50, total_steps // 40)
    nash_q_opec = float(nash.quantities["OPEC"])

    print(f"Budget: episodes={episodes}, steps/ep={stress_params.steps_per_episode}, "
          f"total_steps={total_steps}, n_seeds={n_seeds}")
    print(f"Forced deviation: OPEC pinned at q={nash_q_opec:.2f} mbd from step "
          f"{deviation_start} for {deviation_duration} steps.")
    print()

    t0 = time.time()
    result = run_marl_forced_deviation(
        learning_players=list(params.players),
        all_players=list(params.players),
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=stress_params,
        nash_price=nash_price,
        cartel_price=cartel_price,
        deviator="OPEC",
        deviation_q=nash_q_opec,
        deviation_start=deviation_start,
        deviation_duration=deviation_duration,
        n_seeds=n_seeds,
        base_seed=int(args.base_seed),
    )
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.1f} min")

    out_dir = os.path.join(ROOT, "outputs")

    # ---- Aggregated CSV (matches the original schema of run_marl_stress_tests)
    agg_keys = [
        "deviator", "deviation_q", "deviation_start", "deviation_duration",
        "mean_pre_price", "mean_during_price", "mean_post_price",
        "window", "n_seeds",
        "mean_collusion_index", "std_collusion_index",
        "ci_95_low", "ci_95_high",
        "mean_converged_price", "std_converged_price",
        "mean_q_US", "std_q_US",
        "mean_q_OPEC", "std_q_OPEC",
        "mean_q_RUS", "std_q_RUS",
    ]
    agg_row = {"experiment": "marl_forced_deviation"}
    for k in agg_keys:
        if k in result and isinstance(result[k], (int, float)):
            agg_row[k] = round(float(result[k]), 4)
        elif k in result:
            agg_row[k] = result[k]
    agg_csv = os.path.join(out_dir, "marl_forced_deviation.csv")
    pd.DataFrame([agg_row]).to_csv(agg_csv, index=False)
    print(f"Wrote: {agg_csv}")

    # ---- Decomposition CSV --------------------------------------------------
    decomp_keys = [
        "deviator", "deviation_q", "deviation_start", "deviation_duration",
        "window",
        "mean_pre_price", "mean_during_price", "mean_post_price",
        "pre_q_US", "pre_q_OPEC", "pre_q_RUS",
        "during_q_US", "during_q_OPEC", "during_q_RUS",
        "post_q_US", "post_q_OPEC", "post_q_RUS",
        "delta_q_dev_during_minus_pre",
        "delta_q_non_dev_total",
        "observed_price_drop",
        "mechanical_price_drop",
        "strategic_price_drop",
        "mechanical_share",
        "strategic_share",
    ]
    dec_row = {}
    for k in decomp_keys:
        if k in result and isinstance(result[k], (int, float)):
            dec_row[k] = round(float(result[k]), 4)
        elif k in result:
            dec_row[k] = result[k]
    dec_csv = os.path.join(out_dir, "marl_forced_deviation_decomp.csv")
    pd.DataFrame([dec_row]).to_csv(dec_csv, index=False)
    print(f"Wrote: {dec_csv}")

    # ---- Stdout summary ------------------------------------------------------
    print()
    print("=" * 72)
    print("Forced-deviation decomposition (mean across {} seeds)".format(n_seeds))
    print("=" * 72)
    print(f"  Pre-window  : mean price = {result['mean_pre_price']:.2f} $/bbl")
    for p in params.players:
        print(f"    pre_q_{p:<4s} = {result[f'pre_q_{p}']:.2f} mbd")
    print(f"  During window (deviator pinned at q={nash_q_opec:.1f}):")
    print(f"    mean price = {result['mean_during_price']:.2f} $/bbl")
    for p in params.players:
        print(f"    during_q_{p:<4s} = {result[f'during_q_{p}']:.2f} mbd")
    print(f"  Post-window  : mean price = {result['mean_post_price']:.2f} $/bbl")
    print()
    print(f"  Observed price drop                = {result['observed_price_drop']:+.3f} $/bbl")
    print(f"  Mechanical price drop (b * dQ_dev) = {result['mechanical_price_drop']:+.3f} $/bbl")
    print(f"  Non-dev total dQ                   = {result['delta_q_non_dev_total']:+.3f} mbd")
    print(f"  Strategic price drop (residual)    = {result['strategic_price_drop']:+.3f} $/bbl")
    print(f"  --> Mechanical share = {result['mechanical_share']:.3f}")
    print(f"  --> Strategic share  = {result['strategic_share']:.3f}")
    print()
    threshold = 0.30
    verdict = ("YES - algorithmic punishment DETECTED"
               if result['strategic_share'] >= threshold
               else "NO  - algorithmic punishment NOT detected")
    print(f"  Pre-registered threshold for retaliation: strategic_share >= {threshold:.2f}")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
