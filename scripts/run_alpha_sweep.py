"""Sweep the learning rate alpha for the MARL triopoly headline and report
robustness to this hyperparameter.

Outputs
-------
outputs/marl_alpha_sweep.csv              aggregated (one row per alpha)
outputs/marl_alpha_sweep_per_seed.csv     per-seed CI / price
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from math import sqrt
from typing import List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import default_simulation_params               # noqa: E402
from src.cournot_static import cournot_equilibrium             # noqa: E402
from src.cooperation_punishment import cartel_quotas           # noqa: E402
from src.rl_multiagent import run_multiagent_robustness        # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.25])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=900)
    args = parser.parse_args()

    params = default_simulation_params()
    # Use the same stress-test budget convention as the rest of Part III:
    # Option B fraction (0.5) of the headline budget.
    if args.episodes is None:
        frac = float(params.multi_rl.stress_episodes_fraction)
        episodes = max(50, int(round(params.multi_rl.episodes * frac)))
    else:
        episodes = int(args.episodes)
    n_seeds = int(args.n_seeds)
    alphas: List[float] = [float(a) for a in args.alphas]

    nash = cournot_equilibrium(
        params.players, params.demand, params.costs, params.capacities,
    )
    cartel = cartel_quotas(
        params.players, params.demand, params.costs, params.capacities,
    )
    nash_price = float(nash.price)
    cartel_price = float(cartel.quota_price)

    print(f"Alpha sweep: alphas={alphas}, episodes={episodes}, n_seeds={n_seeds}")
    print(f"Budget per alpha: {episodes * params.multi_rl.steps_per_episode * 3 * n_seeds:.0f} transitions")
    print()

    out_dir = os.path.join(ROOT, "outputs")
    agg_rows = []
    per_seed_rows = []

    t0 = time.time()
    for alpha in alphas:
        ta = time.time()
        print(f"=== alpha = {alpha:.3f} ===")
        stress = replace(params.multi_rl, episodes=episodes, alpha=float(alpha))
        result = run_multiagent_robustness(
            learning_players=list(params.players),
            all_players=list(params.players),
            demand=params.demand,
            costs=params.costs,
            capacities=params.capacities,
            params=stress,
            nash_price=nash_price,
            cartel_price=cartel_price,
            n_seeds=n_seeds,
            base_seed=int(args.base_seed),
        )
        elapsed_a = time.time() - ta
        print(f"  done in {elapsed_a:.1f}s.  "
              f"mean tail CI = {result['mean_collusion_index']:.3f}  "
              f"mean greedy CI = {result['mean_greedy_collusion_index']:.3f}  "
              f"mean price = {result['mean_converged_price']:.2f} $/bbl")

        agg_rows.append({
            "alpha": float(alpha),
            "n_seeds": int(n_seeds),
            "mean_tail_ci":   round(float(result["mean_collusion_index"]), 4),
            "std_tail_ci":    round(float(result["std_collusion_index"]), 4),
            "tail_ci_95_low": round(float(result["ci_95_low"]), 4),
            "tail_ci_95_high":round(float(result["ci_95_high"]), 4),
            "mean_greedy_ci": round(float(result["mean_greedy_collusion_index"]), 4),
            "std_greedy_ci":  round(float(result["std_greedy_collusion_index"]), 4),
            "greedy_ci_95_low":  round(float(result["greedy_ci_95_low"]), 4),
            "greedy_ci_95_high": round(float(result["greedy_ci_95_high"]), 4),
            "mean_converged_price": round(float(result["mean_converged_price"]), 4),
            "std_converged_price":  round(float(result["std_converged_price"]), 4),
            "mean_greedy_price":    round(float(result["mean_greedy_price"]), 4),
            "std_greedy_price":     round(float(result["std_greedy_price"]), 4),
            "mean_q_undervisited_pct": round(float(result["mean_q_undervisited_pct"]), 4),
            "mean_q_visit_mean":       round(float(result["mean_q_visit_mean"]), 4),
        })
        for s, ci, gci, pr, gpr in zip(
            result["seeds"], result["collusion_indices"],
            result["greedy_collusion_indices"], result["converged_prices"],
            result["greedy_prices"],
        ):
            per_seed_rows.append({
                "alpha": float(alpha),
                "seed": int(s),
                "tail_ci": round(float(ci), 4),
                "greedy_ci": round(float(gci), 4),
                "converged_price": round(float(pr), 4),
                "greedy_price": round(float(gpr), 4),
            })

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    agg_path = os.path.join(out_dir, "marl_alpha_sweep.csv")
    per_path = os.path.join(out_dir, "marl_alpha_sweep_per_seed.csv")
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    pd.DataFrame(per_seed_rows).to_csv(per_path, index=False)
    print(f"Wrote: {agg_path}")
    print(f"Wrote: {per_path}")


if __name__ == "__main__":
    main()
