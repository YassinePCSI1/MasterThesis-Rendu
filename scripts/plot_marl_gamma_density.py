"""Companion figure to scripts/plot_marl_basins.py:
visualise how the per-seed distribution of the collusion index
deforms across the gamma (patience) sweep.

Top panel  : strip plot + boxplot, one column per gamma value
             (every dot is one seed, colour-coded by basin).
Bottom panel: Wilson 95% interval on P(CI >= 0.20) per gamma,
              with explicit annotations.

Requires the per-seed CSV produced by scripts/run_gamma_sweep_per_seed.py
(outputs/marl_gamma_sweep_per_seed.csv).

Output
------
outputs/marl_gamma_density.png
outputs/marl_gamma_proportions.csv
"""
from __future__ import annotations

import os
import sys
from math import sqrt
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
PER_SEED_CSV = os.path.join(OUT_DIR, "marl_gamma_sweep_per_seed.csv")
FIG_PATH = os.path.join(OUT_DIR, "marl_gamma_density.png")
SUMMARY_CSV = os.path.join(OUT_DIR, "marl_gamma_proportions.csv")


BASIN_ORDER = ["predatory", "nash_like", "partial_collusion", "supra_cartel"]
BASIN_COLOURS = {
    "predatory":          "#c0392b",
    "nash_like":          "#7f8c8d",
    "partial_collusion":  "#27ae60",
    "supra_cartel":       "#2980b9",
}
BASIN_LABEL = {
    "predatory":          "predatory  (CI < -0.10)",
    "nash_like":          "Nash-like   (-0.10 <= CI < 0.20)",
    "partial_collusion":  "partial collusion  (0.20 <= CI < 0.80)",
    "supra_cartel":       "supra-cartel  (CI >= 0.80)",
}


def classify_basin(ci: float) -> str:
    if ci < -0.10:
        return "predatory"
    if ci < 0.20:
        return "nash_like"
    if ci < 0.80:
        return "partial_collusion"
    return "supra_cartel"


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    if not os.path.exists(PER_SEED_CSV):
        print(f"ERROR: missing {PER_SEED_CSV}", file=sys.stderr)
        print("Run scripts/run_gamma_sweep_per_seed.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(PER_SEED_CSV)
    metric_cols = [c for c in ["greedy_collusion_index", "collusion_index"] if c in df.columns]
    if not metric_cols:
        print("ERROR: no CI column found in per-seed CSV", file=sys.stderr)
        sys.exit(1)
    metric = metric_cols[0]  # prefer greedy
    metric_label = ("greedy CI ($\\epsilon=0$)" if metric == "greedy_collusion_index"
                    else "tail-of-training CI")

    gammas = sorted(df["gamma"].unique())
    n_per_gamma = df.groupby("gamma").size().to_dict()
    n_seeds_max = int(max(n_per_gamma.values()))

    # ---- Per-gamma summary + Wilson on P(coop) ---------------------------
    rows = []
    for g in gammas:
        sub = df[df["gamma"] == g]
        vals = sub[metric].to_numpy()
        n = len(vals)
        coop_k = int(sum(1 for c in vals if c >= 0.20))
        p, lo, hi = wilson_interval(coop_k, n)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        half_mean = 1.96 * std / sqrt(n) if n > 1 else 0.0
        rows.append({
            "gamma": float(g),
            "n_seeds": n,
            "mean_CI": round(mean, 4),
            "std_CI": round(std, 4),
            "mean_CI_95_low": round(mean - half_mean, 4),
            "mean_CI_95_high": round(mean + half_mean, 4),
            "n_cooperative": coop_k,
            "p_cooperative": round(p, 4),
            "wilson_95_low": round(lo, 4),
            "wilson_95_high": round(hi, 4),
            "n_predatory":         int(sum(1 for c in vals if classify_basin(float(c)) == "predatory")),
            "n_nash_like":         int(sum(1 for c in vals if classify_basin(float(c)) == "nash_like")),
            "n_partial_collusion": int(sum(1 for c in vals if classify_basin(float(c)) == "partial_collusion")),
            "n_supra_cartel":      int(sum(1 for c in vals if classify_basin(float(c)) == "supra_cartel")),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print("Per-gamma summary (metric =", metric, ")")
    print(summary.to_string(index=False))
    print(f"\nWrote: {SUMMARY_CSV}")

    # ---- Figure ----------------------------------------------------------
    fig, (ax_strip, ax_prop) = plt.subplots(
        2, 1, figsize=(11, 8.5),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    # --- Top: strip + box per gamma -----------------------------------
    x_positions = np.arange(len(gammas))
    rng = np.random.default_rng(2026)
    for i, g in enumerate(gammas):
        sub = df[df["gamma"] == g]
        vals = sub[metric].to_numpy()
        basins = [classify_basin(float(v)) for v in vals]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        for k, basin in enumerate(BASIN_ORDER):
            mask = np.array([b == basin for b in basins])
            if mask.any():
                ax_strip.scatter(
                    x_positions[i] + jitter[mask], vals[mask],
                    s=40, color=BASIN_COLOURS[basin],
                    edgecolor="white", linewidth=0.6,
                    alpha=0.9, zorder=3,
                )
        # Boxplot overlay
        bp = ax_strip.boxplot(
            vals, positions=[x_positions[i]], widths=0.55,
            patch_artist=True, showfliers=False, zorder=2,
            boxprops=dict(facecolor="white", alpha=0.0, edgecolor="black", linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )

    ax_strip.axhline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.5)
    ax_strip.axhline(1.0, color="black", linestyle="--", lw=1.0, alpha=0.5)
    ax_strip.set_xticks(x_positions)
    ax_strip.set_xticklabels([f"{g:.2f}" for g in gammas])
    ax_strip.set_xlabel("Discount factor  $\\gamma$")
    ax_strip.set_ylabel(f"Per-seed converged {metric_label}")
    ax_strip.set_title(
        f"Per-seed dispersion of the collusion index across the $\\gamma$ sweep "
        f"({n_seeds_max} seeds per $\\gamma$)",
        fontsize=12,
    )
    ax_strip.set_ylim(-1.05, 2.05)
    ax_strip.grid(alpha=0.25, axis="y")

    # Annotate Nash and Cartel guides
    ax_strip.text(len(gammas) - 0.5, 0.04, "Nash benchmark",
                  fontsize=9, color="black", alpha=0.6, ha="right")
    ax_strip.text(len(gammas) - 0.5, 1.04, "Cartel benchmark",
                  fontsize=9, color="black", alpha=0.6, ha="right")

    legend_handles = [
        Patch(color=BASIN_COLOURS[k], label=BASIN_LABEL[k]) for k in BASIN_ORDER
    ]
    ax_strip.legend(
        handles=legend_handles, loc="upper left",
        fontsize=8.5, ncol=2, framealpha=0.85,
    )

    # --- Bottom: Wilson 95% on P(coop) per gamma ----------------------
    p_hat = summary["p_cooperative"].to_numpy()
    p_lo = summary["wilson_95_low"].to_numpy()
    p_hi = summary["wilson_95_high"].to_numpy()
    yerr = np.vstack([p_hat - p_lo, p_hi - p_hat])

    ax_prop.errorbar(
        x_positions, p_hat, yerr=yerr,
        fmt="o", color="#2c3e50", ecolor="#2c3e50",
        markersize=8, linewidth=2.0, capsize=4,
        elinewidth=1.6, label=r"Wilson 95% CI",
    )
    ax_prop.axhline(0.5, color="grey", linestyle=":", lw=1.0,
                    label="majority threshold (0.5)")
    for x, p, n in zip(x_positions, p_hat, summary["n_seeds"]):
        ax_prop.annotate(
            f"{p:.2f}\n(n={n})",
            xy=(x, p), xytext=(x + 0.12, p),
            fontsize=8.5, va="center",
        )
    ax_prop.set_xticks(x_positions)
    ax_prop.set_xticklabels([f"{g:.2f}" for g in gammas])
    ax_prop.set_xlabel("Discount factor  $\\gamma$")
    ax_prop.set_ylabel(r"$\Pr[\,\mathrm{CI} \geq 0.20\,]$")
    ax_prop.set_title(
        r"Probability of reaching a cooperative basin vs. $\gamma$  (Folk Theorem comparative static)",
        fontsize=11,
    )
    ax_prop.set_ylim(-0.05, 1.05)
    ax_prop.grid(alpha=0.25)
    ax_prop.legend(loc="lower right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=160, bbox_inches="tight")
    print(f"Wrote: {FIG_PATH}")


if __name__ == "__main__":
    main()
