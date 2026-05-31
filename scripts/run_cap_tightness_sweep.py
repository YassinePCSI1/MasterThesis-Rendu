"""Cap-tightness sweep for s 18.8.  Vary the capacity caps multiplicatively
(cap_factor in {0.50, 0.75, 1.00, 1.25, 1.50}) around the calibrated baseline
and measure whether tightening capacity moves the converged collusion index
in either direction.

Each cap_factor re-solves the Cournot Nash and cartel benchmarks under the
*new* caps before computing the collusion index, so the metric is comparable
across rows (CI = 0 iff converged price = Nash for THAT cap configuration).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
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
    parser.add_argument("--factors", type=float, nargs="+",
                        default=[0.50, 0.75, 1.00, 1.25, 1.50])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=750)
    parser.add_argument("--base-seed", type=int, default=1100)
    args = parser.parse_args()

    params = default_simulation_params()
    episodes = int(args.episodes)
    n_seeds = int(args.n_seeds)
    factors: List[float] = [float(f) for f in args.factors]

    base_caps = params.capacities
    print(f"Cap baseline (cap_factor=1.0): cap_US={base_caps.cap_us}, "
          f"cap_OPEC={base_caps.cap_opec}, cap_RUS={base_caps.cap_rus}")
    print(f"Factors: {factors}")
    print(f"Episodes per cell: {episodes}, n_seeds={n_seeds}")
    print()

    rows = []
    per_seed_rows = []

    t0 = time.time()
    for factor in factors:
        tf = time.time()
        caps_f = replace(
            base_caps,
            enabled=True,
            cap_us=float(factor * base_caps.cap_us),
            cap_opec=float(factor * base_caps.cap_opec),
            cap_rus=float(factor * base_caps.cap_rus),
        )

        # Re-solve Nash and cartel under THIS cap factor.
        nash = cournot_equilibrium(params.players, params.demand, params.costs, caps_f)
        cartel = cartel_quotas(params.players, params.demand, params.costs, caps_f)
        nash_price = float(nash.price)
        cartel_price = float(cartel.quota_price)

        stress = replace(params.multi_rl, episodes=episodes)

        print(f"=== cap_factor = {factor:.2f}  "
              f"(caps={caps_f.cap_us:.1f}/{caps_f.cap_opec:.1f}/{caps_f.cap_rus:.1f}, "
              f"Nash P={nash_price:.2f}, cartel P={cartel_price:.2f}) ===")
        result = run_multiagent_robustness(
            learning_players=list(params.players),
            all_players=list(params.players),
            demand=params.demand,
            costs=params.costs,
            capacities=caps_f,
            params=stress,
            nash_price=nash_price,
            cartel_price=cartel_price,
            n_seeds=n_seeds,
            base_seed=int(args.base_seed),
        )
        elapsed_f = time.time() - tf
        print(f"  done in {elapsed_f:.1f}s.  "
              f"mean tail CI = {result['mean_collusion_index']:.3f}  "
              f"mean greedy CI = {result['mean_greedy_collusion_index']:.3f}  "
              f"mean price = {result['mean_converged_price']:.2f}")

        rows.append({
            "cap_factor": float(factor),
            "cap_us":   round(float(caps_f.cap_us),   3),
            "cap_opec": round(float(caps_f.cap_opec), 3),
            "cap_rus":  round(float(caps_f.cap_rus),  3),
            "nash_price_this_caps":   round(nash_price, 4),
            "cartel_price_this_caps": round(cartel_price, 4),
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
        })

        for s, ci, gci, pr in zip(
            result["seeds"], result["collusion_indices"],
            result["greedy_collusion_indices"], result["converged_prices"],
        ):
            per_seed_rows.append({
                "cap_factor": float(factor),
                "seed": int(s),
                "tail_ci": round(float(ci), 4),
                "greedy_ci": round(float(gci), 4),
                "converged_price": round(float(pr), 4),
            })

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    out_dir = os.path.join(ROOT, "outputs")
    agg_path = os.path.join(out_dir, "marl_cap_tightness_sweep.csv")
    per_path = os.path.join(out_dir, "marl_cap_tightness_sweep_per_seed.csv")
    pd.DataFrame(rows).to_csv(agg_path, index=False)
    pd.DataFrame(per_seed_rows).to_csv(per_path, index=False)
    print(f"Wrote: {agg_path}")
    print(f"Wrote: {per_path}")


if __name__ == "__main__":
    main()
