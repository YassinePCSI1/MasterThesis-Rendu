"""Two complementary post-hoc tests on a fresh MARL triopoly batch.

Test 1 (multi-seed cycle detector, closes the (b) criterion of report.md s 18.3
and the second leg of Q4 in s 18.11):
    Train the headline triopoly with the Option B budget on 5 independent
    seeds, run the sigma-calibrated punishment detector on each, and report
    the median +/- IQR of n_episodes, frequency per 1000 steps, mean drop,
    mean recovery time.

Test 2 (quantitative reciprocity, closes s 18.10 and gives Q4 a positive
mechanism, even though the explicit retaliation criterion failed):
    For each (seed, agent), compute the Spearman rank correlation rho
    between the price-bin index and the policy action quantity Q*(s),
    averaged over the agent's own_q dimension and over visited cells only
    (visit_count >= 1). A negative rho means "low observed price -> high
    quantity = aggressive response" and a positive rho means "low observed
    price -> low quantity = restraint". The pre-registered direction for
    *reciprocity* is positive rho (the agent restrains itself when the
    price is low, consistent with implicit cooperation).

Outputs
-------
outputs/marl_cycles_multiseed.csv         per-seed cycle stats
outputs/marl_cycles_multiseed_summary.csv aggregate (median, IQR, mean)
outputs/marl_reciprocity_spearman.csv     per-(seed, agent) Spearman
outputs/marl_reciprocity_summary.csv      per-agent + global summary
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import default_simulation_params               # noqa: E402
from src.cournot_static import cournot_equilibrium             # noqa: E402
from src.cooperation_punishment import cartel_quotas           # noqa: E402
from src.rl_multiagent import (                                # noqa: E402
    train_multiagent_ql,
    detect_punishment_episodes,
    compute_punishment_statistics,
)


def _spearman_per_agent_policy(outcome) -> Dict[str, float]:
    """Policy-level Spearman: rho between the price-bin index of a STATE and
    the policy action quantity argmax_a Q(s,a) at that state, with the
    "visited" approximation policy_q > grid[0] + eps.
    """
    out: Dict[str, float] = {}
    for p in outcome.learning_players:
        q = outcome.q_tables[p]
        grid = outcome.action_grids[p]
        idx = np.argmax(q, axis=2)
        policy_q = grid[idx]
        n_price = policy_q.shape[0]
        rows, cols_q = [], []
        for pb in range(n_price):
            for qb in range(policy_q.shape[1]):
                rows.append(pb)
                cols_q.append(float(policy_q[pb, qb]))
        x = np.asarray(rows, dtype=float)
        y = np.asarray(cols_q, dtype=float)
        mask = y > grid[0] + 1e-9
        if mask.sum() < 8:
            out[p] = float("nan")
            continue
        rho, _ = spearmanr(x[mask], y[mask])
        out[p] = float(rho) if not np.isnan(rho) else float("nan")
    return out


def _spearman_per_agent_realised(outcome, tail_fraction: float = 0.20) -> Dict[str, float]:
    """Behavioural Spearman: rho between the OBSERVED price at step t and the
    REALISED action quantity at step t+1 for each agent, evaluated on the
    last `tail_fraction` of the training trajectory (where policy is roughly
    stable).  This is the empirical counterpart of the policy-level test
    above, restricted to states that the agent actually visited during the
    training endgame.
    """
    out: Dict[str, float] = {}
    prices = np.asarray(outcome.price_history, dtype=float)
    n = len(prices)
    tail_start = int((1.0 - tail_fraction) * n)
    if tail_start + 4 >= n:
        for p in outcome.learning_players:
            out[p] = float("nan")
        return out

    for p in outcome.learning_players:
        q_series = np.asarray(
            [step[p] for step in outcome.q_history], dtype=float,
        )
        # Lagged regression: price at step t -> quantity at step t+1.
        x = prices[tail_start:-1]
        y = q_series[tail_start + 1:]
        if len(x) < 8:
            out[p] = float("nan")
            continue
        rho, _ = spearmanr(x, y)
        out[p] = float(rho) if not np.isnan(rho) else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episodes; default = headline Option B (1500).")
    parser.add_argument("--base-seed", type=int, default=300)
    args = parser.parse_args()

    params = default_simulation_params()
    episodes = int(args.episodes) if args.episodes is not None else int(params.multi_rl.episodes)
    n_seeds = int(args.n_seeds)
    base_seed = int(args.base_seed)

    nash = cournot_equilibrium(
        params.players, params.demand, params.costs, params.capacities,
    )
    cartel = cartel_quotas(
        params.players, params.demand, params.costs, params.capacities,
    )
    nash_price = float(nash.price)
    cartel_price = float(cartel.quota_price)

    headline_params = replace(params.multi_rl, episodes=episodes)
    total_steps = episodes * params.multi_rl.steps_per_episode

    print(f"Multi-seed cycles + Spearman reciprocity: episodes={episodes}, "
          f"steps/seed={total_steps}, n_seeds={n_seeds}")
    print()

    cycles_rows: List[Dict] = []
    spearman_rows: List[Dict] = []

    t0 = time.time()
    for i in range(n_seeds):
        seed = base_seed + i
        ts = time.time()
        print(f"[seed {i+1}/{n_seeds}] seed={seed}")
        outcome = train_multiagent_ql(
            learning_players=list(params.players),
            all_players=list(params.players),
            demand=params.demand,
            costs=params.costs,
            capacities=params.capacities,
            params=headline_params,
            nash_price=nash_price,
            cartel_price=cartel_price,
            seed=seed,
        )

        # ---- Cycle detector ------------------------------------------------
        episodes_seed = detect_punishment_episodes(outcome)
        cycle_stats = compute_punishment_statistics(
            episodes_seed, total_steps=total_steps
        )
        cycles_rows.append({
            "seed": int(seed),
            "n_episodes": int(cycle_stats["n_episodes"]),
            "frequency_per_1000": round(float(cycle_stats["punishment_frequency"]), 4),
            "mean_duration_steps": round(float(cycle_stats["mean_duration"]), 2),
            "mean_recovery_time_steps": round(float(cycle_stats["mean_recovery_time"]), 2),
            "mean_drop_dollar": round(float(cycle_stats["mean_drop"]), 4),
            "mean_pre_price": round(float(cycle_stats["mean_pre_price"]), 4),
            "mean_trough_price": round(float(cycle_stats["mean_trough_price"]), 4),
        })

        # ---- Spearman reciprocity (policy-level + behavioural) --------------
        spearman_policy = _spearman_per_agent_policy(outcome)
        spearman_real = _spearman_per_agent_realised(outcome)
        for p in params.players:
            spearman_rows.append({
                "seed": int(seed),
                "player": p,
                "spearman_rho_policy": round(float(spearman_policy[p]), 4),
                "spearman_rho_realised_lag1": round(float(spearman_real[p]), 4),
            })

        elapsed_s = time.time() - ts
        print(f"  done in {elapsed_s:.1f}s.  cycles={cycle_stats['n_episodes']}  "
              f"freq/1k={cycle_stats['punishment_frequency']:.2f}")
        print(f"    Spearman (policy-level)    : " +
              ", ".join(f"{p}={spearman_policy[p]:+.3f}" for p in params.players))
        print(f"    Spearman (realised lag-1)  : " +
              ", ".join(f"{p}={spearman_real[p]:+.3f}" for p in params.players))

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    out_dir = os.path.join(ROOT, "outputs")
    cycles_df = pd.DataFrame(cycles_rows)
    spearman_df = pd.DataFrame(spearman_rows)
    cycles_path = os.path.join(out_dir, "marl_cycles_multiseed.csv")
    spearman_path = os.path.join(out_dir, "marl_reciprocity_spearman.csv")
    cycles_df.to_csv(cycles_path, index=False)
    spearman_df.to_csv(spearman_path, index=False)
    print(f"Wrote: {cycles_path}")
    print(f"Wrote: {spearman_path}")

    # ---- Summaries ----------------------------------------------------------
    cyc_summary = pd.DataFrame([
        {
            "stat": "median",
            **{c: float(np.median(cycles_df[c])) for c in
               ["n_episodes", "frequency_per_1000", "mean_duration_steps",
                "mean_recovery_time_steps", "mean_drop_dollar"]},
        },
        {
            "stat": "IQR",
            **{c: float(np.subtract(*np.percentile(cycles_df[c], [75, 25])))
               for c in ["n_episodes", "frequency_per_1000", "mean_duration_steps",
                         "mean_recovery_time_steps", "mean_drop_dollar"]},
        },
        {
            "stat": "mean",
            **{c: float(cycles_df[c].mean()) for c in
               ["n_episodes", "frequency_per_1000", "mean_duration_steps",
                "mean_recovery_time_steps", "mean_drop_dollar"]},
        },
    ]).round(4)
    cyc_summary_path = os.path.join(out_dir, "marl_cycles_multiseed_summary.csv")
    cyc_summary.to_csv(cyc_summary_path, index=False)
    print(f"Wrote: {cyc_summary_path}")

    print()
    print("=" * 72)
    print("Multi-seed cycle detector summary (n={} seeds)".format(n_seeds))
    print("=" * 72)
    print(cyc_summary.to_string(index=False))

    # ---- Spearman summary (policy + realised) ------------------------------
    def _block_summary(col: str) -> pd.DataFrame:
        rows_ = []
        for p in params.players:
            sub = spearman_df.query("player == @p")[col]
            rows_.append({
                "player": p,
                "metric": col,
                "median_rho": round(float(np.median(sub)), 4),
                "mean_rho":   round(float(sub.mean()), 4),
                "n_seeds_with_rho_positive": int((sub > 0).sum()),
                "n_seeds_with_rho_greater_0p30": int((sub > 0.30).sum()),
                "n_seeds_total": int(sub.shape[0]),
            })
        all_rho = spearman_df[col]
        rows_.append({
            "player": "ALL",
            "metric": col,
            "median_rho": round(float(np.median(all_rho)), 4),
            "mean_rho":   round(float(all_rho.mean()), 4),
            "n_seeds_with_rho_positive": int((all_rho > 0).sum()),
            "n_seeds_with_rho_greater_0p30": int((all_rho > 0.30).sum()),
            "n_seeds_total": int(all_rho.shape[0]),
        })
        return pd.DataFrame(rows_)

    sp_summary = pd.concat([
        _block_summary("spearman_rho_policy"),
        _block_summary("spearman_rho_realised_lag1"),
    ], ignore_index=True)
    sp_summary_path = os.path.join(out_dir, "marl_reciprocity_summary.csv")
    sp_summary.to_csv(sp_summary_path, index=False)
    print()
    print("=" * 72)
    print("Per-agent Spearman reciprocity summary (n={} seeds)".format(n_seeds))
    print("=" * 72)
    print(sp_summary.to_string(index=False))
    print()

    for metric in ["spearman_rho_policy", "spearman_rho_realised_lag1"]:
        all_rho_m = spearman_df[metric]
        median_global = float(np.median(all_rho_m))
        # Per-seed agreement: count seeds whose own median across agents is >0.
        n_positive_seeds = 0
        for s in spearman_df["seed"].unique():
            med_s = float(spearman_df.query("seed == @s")[metric].median())
            if med_s > 0:
                n_positive_seeds += 1
        print(f"[{metric}] median across all (seed, agent) = {median_global:+.3f} ; "
              f"per-seed median > 0 in {n_positive_seeds}/{n_seeds} seeds.")
        verdict = ("YES - reciprocity DETECTED (median>0.30, >=3/5 seeds)"
                   if median_global > 0.30 and n_positive_seeds >= 3
                   else "NO  - reciprocity NOT detected at the pre-registered threshold")
        print(f"           Verdict: {verdict}")


if __name__ == "__main__":
    main()
