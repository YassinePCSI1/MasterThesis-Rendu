"""Re-run the MARL Green-Porter shock experiment and compute the in-window
scalars that the s 18.5 pre-registered criterion actually asks about:

* mean in-window price drop (averaged over shock windows, multi-seed),
* recovery half-life (steps from end-of-shock until rolling-mean price
  returns within 1 * sigma_P of the pre-shock level),
* relative drop magnitude (drop / sigma_P).

These are exactly the scalars flagged as "to be logged as follow-up work"
in the s 18.5 limitations and resolve the gap between the headline
post-shock-training CI and the in-shock dynamics.

Outputs
-------
outputs/marl_shock_window_scalars.csv  per-seed per-shock-window stats
outputs/marl_shock_window_summary.csv  aggregated (median, IQR, mean)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import default_simulation_params               # noqa: E402
from src.cournot_static import cournot_equilibrium             # noqa: E402
from src.cooperation_punishment import cartel_quotas           # noqa: E402
from src.rl_multiagent import train_multiagent_ql              # noqa: E402


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return np.full_like(x, x.mean())
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    pad = np.full(w - 1, out[0])
    return np.concatenate([pad, out])


def _window_scalars(
    prices: np.ndarray,
    shock_starts: List[int],
    shock_ends: List[int],
    roll_window: int = 50,
    sigma_window_frac: float = 0.20,
) -> Tuple[List[Dict], float]:
    """For each shock window, compute drop and recovery-half-life scalars.

    Returns
    -------
    scalars : one dict per shock window.
    sigma_p : the late-training sigma of the rolling-mean price (used for
              the relative-drop ratio).
    """
    r = _rolling_mean(prices, w=roll_window)
    tail_start = int((1.0 - sigma_window_frac) * len(r))
    sigma_p = float(np.std(r[tail_start:], ddof=1))

    out: List[Dict] = []
    for s_lo, s_hi in zip(shock_starts, shock_ends):
        # Pre-shock baseline: mean rolling-price over the [s_lo - roll_window, s_lo) window
        pre_lo = max(0, s_lo - roll_window)
        pre_mean = float(r[pre_lo:s_lo].mean()) if s_lo > pre_lo else float("nan")
        # In-window mean and trough
        if s_hi > s_lo:
            in_window = r[s_lo:s_hi]
            in_mean = float(in_window.mean())
            in_trough = float(in_window.min())
            in_trough_step = int(np.argmin(in_window) + s_lo)
        else:
            in_mean = float("nan")
            in_trough = float("nan")
            in_trough_step = -1
        mean_drop = float(pre_mean - in_mean) if not np.isnan(pre_mean) else float("nan")
        trough_drop = float(pre_mean - in_trough) if not np.isnan(pre_mean) else float("nan")
        relative_mean_drop = (mean_drop / sigma_p) if sigma_p > 0 else float("nan")
        relative_trough_drop = (trough_drop / sigma_p) if sigma_p > 0 else float("nan")
        # Recovery: first step >= s_hi at which rolling-mean returns to within
        # 1*sigma_p of pre_mean
        recovery_step = -1
        if not np.isnan(pre_mean) and sigma_p > 0 and s_hi < len(r):
            target = pre_mean - sigma_p
            for s in range(s_hi, len(r)):
                if r[s] >= target:
                    recovery_step = int(s)
                    break
        recovery_half_life = (recovery_step - s_hi) if recovery_step >= 0 else float("nan")
        window_width = s_hi - s_lo
        recovery_ratio = (
            (recovery_half_life / window_width)
            if (window_width > 0 and not np.isnan(recovery_half_life)) else float("nan")
        )

        out.append({
            "shock_start": int(s_lo),
            "shock_end": int(s_hi),
            "shock_width": int(window_width),
            "pre_shock_mean_price": round(float(pre_mean), 4),
            "in_window_mean_price": round(float(in_mean), 4),
            "in_window_trough_price": round(float(in_trough), 4),
            "in_window_trough_step": int(in_trough_step),
            "mean_drop_dollar": round(float(mean_drop), 4),
            "trough_drop_dollar": round(float(trough_drop), 4),
            "relative_mean_drop_in_sigmaP": round(float(relative_mean_drop), 4),
            "relative_trough_drop_in_sigmaP": round(float(relative_trough_drop), 4),
            "recovery_half_life_steps": (
                int(recovery_half_life)
                if not (isinstance(recovery_half_life, float) and np.isnan(recovery_half_life))
                else None
            ),
            "recovery_half_life_over_shock_width": (
                round(float(recovery_ratio), 4)
                if not np.isnan(recovery_ratio) else None
            ),
        })
    return out, sigma_p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=750)
    parser.add_argument("--base-seed", type=int, default=600)
    args = parser.parse_args()

    params = default_simulation_params()
    n_seeds = int(args.n_seeds)
    episodes = int(args.episodes)
    base_seed = int(args.base_seed)
    total_steps = episodes * params.multi_rl.steps_per_episode

    # Mirror the shock schedule of simulations.run_marl_stress_tests:
    # two negative-demand windows at roughly 1/3 and 2/3 of training.
    shock_starts = [int(0.33 * total_steps), int(0.66 * total_steps)]
    shock_width = max(50, total_steps // 10)
    shock_ends = [s + shock_width for s in shock_starts]
    delta_a = -10.0   # negative demand shock (matches simulations.py default)
    shock_schedule = [
        (int(s_lo), int(s_hi), float(delta_a))
        for s_lo, s_hi in zip(shock_starts, shock_ends)
    ]
    print(f"Shock schedule (total_steps={total_steps}, n_seeds={n_seeds}):")
    for s in shock_schedule:
        print(f"  shock window: steps [{s[0]} ; {s[1]}), delta_a={s[2]}")
    print()

    nash = cournot_equilibrium(
        params.players, params.demand, params.costs, params.capacities,
    )
    cartel = cartel_quotas(
        params.players, params.demand, params.costs, params.capacities,
    )
    nash_price = float(nash.price)
    cartel_price = float(cartel.quota_price)
    headline_params = replace(params.multi_rl, episodes=episodes)

    rows: List[Dict] = []
    sigma_rows: List[Dict] = []

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
            shock_schedule=shock_schedule,
        )
        prices = np.asarray(outcome.price_history, dtype=float)
        per_shock, sigma_p = _window_scalars(prices, shock_starts, shock_ends)
        sigma_rows.append({"seed": int(seed), "sigma_p_late": round(sigma_p, 4)})
        for k, ps in enumerate(per_shock):
            rows.append({"seed": int(seed), "shock_idx": int(k), **ps})
        elapsed_s = time.time() - ts
        last = per_shock[-1] if per_shock else {}
        last_drop = last.get("mean_drop_dollar", float("nan"))
        last_relative = last.get("relative_mean_drop_in_sigmaP", float("nan"))
        last_recovery = last.get("recovery_half_life_steps")
        print(f"  done in {elapsed_s:.1f}s.  "
              f"sigma_P_late = {sigma_p:.3f}  "
              f"shock-2: mean_drop = {last_drop:.2f} ({last_relative:.2f}*sigmaP) ; "
              f"recovery_HL = {last_recovery}")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    out_dir = os.path.join(ROOT, "outputs")
    df = pd.DataFrame(rows)
    sigma_df = pd.DataFrame(sigma_rows)
    out_path = os.path.join(out_dir, "marl_shock_window_scalars.csv")
    sigma_path = os.path.join(out_dir, "marl_shock_window_sigma.csv")
    df.to_csv(out_path, index=False)
    sigma_df.to_csv(sigma_path, index=False)
    print(f"Wrote: {out_path}")
    print(f"Wrote: {sigma_path}")

    # ---- Summary (aggregated over seeds and shock windows) -----------------
    num_cols = [
        "mean_drop_dollar", "trough_drop_dollar",
        "relative_mean_drop_in_sigmaP", "relative_trough_drop_in_sigmaP",
        "recovery_half_life_steps", "recovery_half_life_over_shock_width",
    ]
    summary_rows = []
    for stat_name, stat_fn in [
        ("median", lambda c: float(np.nanmedian(df[c].astype(float)))),
        ("IQR",    lambda c: float(np.nanpercentile(df[c].astype(float), 75)
                                 - np.nanpercentile(df[c].astype(float), 25))),
        ("mean",   lambda c: float(np.nanmean(df[c].astype(float)))),
        ("std",    lambda c: float(np.nanstd(df[c].astype(float), ddof=1))),
    ]:
        summary_rows.append({"stat": stat_name, **{c: round(stat_fn(c), 4) for c in num_cols}})
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "marl_shock_window_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")
    print()
    print("=" * 72)
    print("Shock window in-window scalars (aggregated over {} seeds * "
          "{} windows = {} obs)".format(n_seeds, len(shock_starts), len(rows)))
    print("=" * 72)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
