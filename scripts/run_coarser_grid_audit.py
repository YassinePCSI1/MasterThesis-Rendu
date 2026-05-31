"""Close the open question of report.md s 18.12 limitation #8 :
path-dependent fixed-point lock-in vs weak reciprocity below the
detection floor.

Strategy
--------
Repeat the headline triopoly batch and the two Spearman tests of s 18.10
with a *coarser* discretisation -- (action_grid_size=10, price_bins=9,
own_q_bins=7) vs the headline (15, 12, 10).  This lifts the average
visits per (s, a) cell from ~70 to ~240, removing the unvisited-cell
noise from the policy-level Spearman test.

Pre-registered (this script) decision rule
------------------------------------------
* If the **realised** median Spearman ρ across (seed, agent) at the
  coarser grid jumps above the pre-registered +0.30 threshold AND
  the sign is reproducible in >= 3/5 seeds, conclude that
  *weak reciprocity below the detection floor* is the right
  interpretation -- the original null was a discretisation artefact.
* If the median realised ρ stays below ~+0.15, conclude that
  *path-dependent fixed-point lock-in* is the right interpretation --
  reciprocity is genuinely absent, the basin is sustained by static
  joint-output tuples chosen during early exploration.

For traceability we also report the headline collusion index at the
coarser grid (it should reproduce the cooperative basin if the result
is robust to discretisation) and the Q-table coverage diagnostics.

Outputs
-------
outputs/marl_coarser_grid_headline.csv          aggregated headline at coarse grid
outputs/marl_coarser_grid_per_seed.csv          per-seed CI / price / coverage
outputs/marl_coarser_grid_spearman.csv          per-(seed, agent) Spearman
outputs/marl_coarser_grid_summary.csv           verdict summary
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from math import sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import default_simulation_params               # noqa: E402
from src.cournot_static import cournot_equilibrium             # noqa: E402
from src.cooperation_punishment import cartel_quotas           # noqa: E402
from src.rl_multiagent import train_multiagent_ql              # noqa: E402


def _spearman_policy(outcome) -> Dict[str, float]:
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


def _spearman_realised(outcome, tail_fraction: float = 0.20) -> Dict[str, float]:
    out: Dict[str, float] = {}
    prices = np.asarray(outcome.price_history, dtype=float)
    n = len(prices)
    tail_start = int((1.0 - tail_fraction) * n)
    if tail_start + 4 >= n:
        for p in outcome.learning_players:
            out[p] = float("nan")
        return out
    for p in outcome.learning_players:
        q_series = np.asarray([step[p] for step in outcome.q_history], dtype=float)
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
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1500,
                        help="Headline budget by default (Option B headline).")
    parser.add_argument("--action-grid-size", type=int, default=10)
    parser.add_argument("--price-bins", type=int, default=9)
    parser.add_argument("--own-q-bins", type=int, default=7)
    parser.add_argument("--base-seed", type=int, default=1300)
    args = parser.parse_args()

    params = default_simulation_params()
    episodes = int(args.episodes)
    n_seeds = int(args.n_seeds)
    base_seed = int(args.base_seed)

    coarse_params = replace(
        params.multi_rl,
        episodes=episodes,
        action_grid_size=int(args.action_grid_size),
        price_bins=int(args.price_bins),
        own_q_bins=int(args.own_q_bins),
    )
    total_steps = episodes * coarse_params.steps_per_episode

    nash = cournot_equilibrium(
        params.players, params.demand, params.costs, params.capacities,
    )
    cartel = cartel_quotas(
        params.players, params.demand, params.costs, params.capacities,
    )
    nash_price = float(nash.price)
    cartel_price = float(cartel.quota_price)

    # State-action cell count comparison.
    fine_cells = (12 + 1) * (10 + 1) * 15           # original headline
    coarse_cells = (args.price_bins + 1) * (args.own_q_bins + 1) * args.action_grid_size
    print(f"Coarser-grid audit (s 18.12 limitation #8 closer)")
    print(f"  fine grid   = (15, 12, 10) -> {fine_cells} cells/agent")
    print(f"  coarse grid = ({args.action_grid_size}, {args.price_bins}, {args.own_q_bins}) "
          f"-> {coarse_cells} cells/agent")
    print(f"  expected ratio of mean visits/cell at same budget = "
          f"{fine_cells / coarse_cells:.2f}x")
    print(f"  episodes per seed = {episodes}, n_seeds = {n_seeds}, "
          f"steps/seed = {total_steps}")
    print()

    headline_rows: List[Dict] = []
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
            params=coarse_params,
            nash_price=nash_price,
            cartel_price=cartel_price,
            seed=seed,
        )
        sp_policy = _spearman_policy(outcome)
        sp_real = _spearman_realised(outcome)
        for p in params.players:
            spearman_rows.append({
                "seed": int(seed),
                "player": p,
                "spearman_rho_policy": round(float(sp_policy[p]), 4),
                "spearman_rho_realised_lag1": round(float(sp_real[p]), 4),
            })
        headline_rows.append({
            "seed": int(seed),
            "tail_ci": round(float(outcome.collusion_index), 4),
            "greedy_ci": round(float(outcome.greedy_collusion_index), 4),
            "converged_price": round(float(outcome.converged_price), 4),
            "greedy_price": round(float(outcome.greedy_price), 4),
            "q_undervisited_pct": round(float(outcome.q_undervisited_pct), 4),
            "q_visit_mean": round(float(outcome.q_visit_mean), 4),
            "q_visit_min": int(outcome.q_visit_min),
        })
        dt_s = time.time() - ts
        print(f"  done in {dt_s:.1f}s.  tail CI = {outcome.collusion_index:+.3f}, "
              f"greedy CI = {outcome.greedy_collusion_index:+.3f}, "
              f"visits/cell mean = {outcome.q_visit_mean:.1f}")
        print(f"    Spearman policy   : " +
              ", ".join(f"{p}={sp_policy[p]:+.3f}" for p in params.players))
        print(f"    Spearman realised : " +
              ", ".join(f"{p}={sp_real[p]:+.3f}" for p in params.players))

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    out_dir = os.path.join(ROOT, "outputs")
    head_df = pd.DataFrame(headline_rows)
    sp_df = pd.DataFrame(spearman_rows)
    head_per_path = os.path.join(out_dir, "marl_coarser_grid_per_seed.csv")
    sp_path = os.path.join(out_dir, "marl_coarser_grid_spearman.csv")
    head_df.to_csv(head_per_path, index=False)
    sp_df.to_csv(sp_path, index=False)
    print(f"Wrote: {head_per_path}")
    print(f"Wrote: {sp_path}")

    # ---- Aggregated headline ------------------------------------------------
    halfwidth = lambda c: 1.96 * head_df[c].std(ddof=1) / sqrt(n_seeds)
    head_agg = pd.DataFrame([{
        "n_seeds": int(n_seeds),
        "fine_cells_per_agent":   int(fine_cells),
        "coarse_cells_per_agent": int(coarse_cells),
        "mean_tail_ci":   round(float(head_df["tail_ci"].mean()), 4),
        "std_tail_ci":    round(float(head_df["tail_ci"].std(ddof=1)), 4),
        "tail_ci_95_low":  round(float(head_df["tail_ci"].mean() - halfwidth("tail_ci")), 4),
        "tail_ci_95_high": round(float(head_df["tail_ci"].mean() + halfwidth("tail_ci")), 4),
        "mean_greedy_ci": round(float(head_df["greedy_ci"].mean()), 4),
        "std_greedy_ci":  round(float(head_df["greedy_ci"].std(ddof=1)), 4),
        "greedy_ci_95_low":  round(float(head_df["greedy_ci"].mean() - halfwidth("greedy_ci")), 4),
        "greedy_ci_95_high": round(float(head_df["greedy_ci"].mean() + halfwidth("greedy_ci")), 4),
        "mean_converged_price": round(float(head_df["converged_price"].mean()), 4),
        "std_converged_price":  round(float(head_df["converged_price"].std(ddof=1)), 4),
        "mean_q_undervisited_pct": round(float(head_df["q_undervisited_pct"].mean()), 4),
        "mean_q_visit_mean":       round(float(head_df["q_visit_mean"].mean()), 4),
        "min_q_visit_min":         int(head_df["q_visit_min"].min()),
    }])
    head_agg_path = os.path.join(out_dir, "marl_coarser_grid_headline.csv")
    head_agg.to_csv(head_agg_path, index=False)
    print(f"Wrote: {head_agg_path}")

    # ---- Spearman summary + verdict ----------------------------------------
    def _block(col: str) -> pd.DataFrame:
        rows_ = []
        for p in params.players:
            sub = sp_df.query("player == @p")[col]
            rows_.append({
                "player": p,
                "metric": col,
                "median_rho": round(float(np.median(sub)), 4),
                "mean_rho":   round(float(sub.mean()), 4),
                "n_seeds_with_rho_positive": int((sub > 0).sum()),
                "n_seeds_with_rho_greater_0p30": int((sub > 0.30).sum()),
                "n_seeds_total": int(sub.shape[0]),
            })
        all_rho = sp_df[col]
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
        _block("spearman_rho_policy"),
        _block("spearman_rho_realised_lag1"),
    ], ignore_index=True)
    sp_summary_path = os.path.join(out_dir, "marl_coarser_grid_summary.csv")
    sp_summary.to_csv(sp_summary_path, index=False)
    print(f"Wrote: {sp_summary_path}")

    print()
    print("=" * 72)
    print("Coarser-grid headline (n={} seeds)".format(n_seeds))
    print("=" * 72)
    print(head_agg.T.to_string())
    print()
    print("=" * 72)
    print("Coarser-grid Spearman summary".format(n_seeds))
    print("=" * 72)
    print(sp_summary.to_string(index=False))
    print()

    # Verdict
    print("=" * 72)
    print("VERDICT (s 18.12 limitation #8 closer)")
    print("=" * 72)
    for metric in ["spearman_rho_policy", "spearman_rho_realised_lag1"]:
        all_rho = sp_df[metric]
        median_global = float(np.median(all_rho))
        n_pos_seeds = 0
        for s in sp_df["seed"].unique():
            med_s = float(sp_df.query("seed == @s")[metric].median())
            if med_s > 0:
                n_pos_seeds += 1
        passes = median_global > 0.30 and n_pos_seeds >= 3
        print(f"  [{metric}] median = {median_global:+.3f} ; per-seed med > 0 in "
              f"{n_pos_seeds}/{n_seeds}")
        print(f"    -> threshold (median > 0.30 AND >= 3/5 seeds): "
              f"{'YES weak reciprocity REVEALED' if passes else 'NO still below floor'}")
    print()
    print("Compare with fine-grid baseline (s 18.10):")
    print("  policy   : median = +0.016 ; per-seed med > 0 in 3/5 seeds  -> NOT detected")
    print("  realised : median = +0.073 ; per-seed med > 0 in 4/5 seeds  -> NOT detected")
    print()
    print("Interpretation: if the coarser grid lifts realised median above ~+0.15,")
    print("the original null is a discretisation artefact (weak reciprocity hypothesis).")
    print("If the realised median stays near zero, path-dependence is the right reading.")


if __name__ == "__main__":
    main()
