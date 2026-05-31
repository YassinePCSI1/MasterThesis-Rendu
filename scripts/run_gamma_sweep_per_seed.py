"""Re-run the MARL gamma sweep, this time saving per-seed values.

Designed to be lightweight: uses the *stress* training budget by default
(0.5 x MultiAgentRLParams.episodes, matching Option B's stress-test calibre).
Override --episodes and --n-seeds for quicker iteration.

Outputs
-------
outputs/marl_gamma_sweep.csv             (aggregated, overwrites existing)
outputs/marl_gamma_sweep_per_seed.csv    (new -- 1 row per (gamma, seed))
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

from src.config import default_simulation_params  # noqa: E402
from src.cournot_static import cournot_equilibrium  # noqa: E402
from src.cooperation_punishment import cartel_quotas  # noqa: E402
from src.rl_multiagent import run_marl_gamma_sweep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override MultiAgentRLParams.episodes (default: "
                             "stress_episodes_fraction * episodes).")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Independent seeds per gamma (default 20).")
    parser.add_argument("--gamma-values", type=str,
                        default="0.50,0.70,0.85,0.95,0.99",
                        help="Comma-separated gamma grid.")
    parser.add_argument("--base-seed", type=int, default=500)
    args = parser.parse_args()

    params = default_simulation_params()
    if args.episodes is None:
        frac = float(params.multi_rl.stress_episodes_fraction)
        episodes = max(50, int(round(params.multi_rl.episodes * frac)))
    else:
        episodes = int(args.episodes)
    stress_params = replace(params.multi_rl, episodes=episodes)
    gamma_values = [float(x) for x in args.gamma_values.split(",")]
    n_seeds = int(args.n_seeds)

    nash = cournot_equilibrium(
        params.players, params.demand, params.costs, params.capacities,
    )
    cartel = cartel_quotas(
        params.players, params.demand, params.costs, params.capacities,
    )
    nash_price = float(nash.price)
    cartel_price = float(cartel.quota_price)

    total = len(gamma_values) * n_seeds
    transitions_per_run = episodes * stress_params.steps_per_episode * len(params.players)
    print(f"Budget: {episodes} episodes x {stress_params.steps_per_episode} steps "
          f"x {len(params.players)} agents = {transitions_per_run:,} transitions/run")
    print(f"Total runs: {len(gamma_values)} gammas x {n_seeds} seeds = {total}")
    print(f"Total transitions: {transitions_per_run * total:,}")
    print(f"Gamma grid: {gamma_values}")
    print()

    t0 = time.time()
    result = run_marl_gamma_sweep(
        learning_players=list(params.players),
        all_players=list(params.players),
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=stress_params,
        nash_price=nash_price,
        cartel_price=cartel_price,
        gamma_values=gamma_values,
        n_seeds=n_seeds,
        base_seed=int(args.base_seed),
        return_per_seed=True,
    )
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.1f} min")

    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    agg_csv = os.path.join(out_dir, "marl_gamma_sweep.csv")
    ps_csv = os.path.join(out_dir, "marl_gamma_sweep_per_seed.csv")

    pd.DataFrame([{k: round(float(v), 4) if isinstance(v, (int, float)) else v
                   for k, v in row.items()} for row in result["aggregated"]]).to_csv(
        agg_csv, index=False,
    )
    pd.DataFrame([{k: round(float(v), 4) if isinstance(v, (int, float)) else v
                   for k, v in row.items()} for row in result["per_seed"]]).to_csv(
        ps_csv, index=False,
    )
    print(f"Wrote: {agg_csv}")
    print(f"Wrote: {ps_csv}  ({len(result['per_seed'])} rows)")


if __name__ == "__main__":
    main()
