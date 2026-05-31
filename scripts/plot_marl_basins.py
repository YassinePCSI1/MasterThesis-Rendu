"""Plot the per-seed distribution of the collusion index for the headline MARL
batch (triopoly and duopoly), to visualise the multi-basin nature of tabular
Q-learning in a multi-agent environment.

The argument: a single (mean +- std) statistic hides the fact that independent
seeds converge to qualitatively different equilibria (Nash-like, partial
collusion, supra-cartel).  This script produces the figure that makes the
multi-basin structure unambiguous to the reader, and reports a Wilson 95%
confidence interval on the *proportion* of seeds reaching a cooperative
basin (CI >= 0.20) -- a statistic that is strictly more informative than the
mean when the per-seed distribution is bimodal.

Outputs
-------
outputs/marl_collusion_basins.png   -- figure
outputs/marl_collusion_basins.csv   -- per-basin counts, proportions, Wilson CIs
"""
from __future__ import annotations

import os
import sys
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
TRIO_CSV = os.path.join(OUT_DIR, "multiagent_rl_robustness_triopoly_per_seed.csv")
DUO_CSV = os.path.join(OUT_DIR, "multiagent_rl_robustness_per_seed.csv")
FIG_PATH = os.path.join(OUT_DIR, "marl_collusion_basins.png")
SUMMARY_CSV = os.path.join(OUT_DIR, "marl_collusion_basins.csv")


# ---- Basin classification -------------------------------------------------

BASIN_ORDER = ["predatory", "nash_like", "partial_collusion", "supra_cartel"]
BASIN_COLOURS = {
    "predatory":          "#c0392b",
    "nash_like":          "#7f8c8d",
    "partial_collusion":  "#27ae60",
    "supra_cartel":       "#2980b9",
}
BASIN_LABEL = {
    "predatory":          "predatory  (CI < -0.10, below Nash)",
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


# ---- Wilson score interval for a binomial proportion ----------------------
# Robust at the extremes p -> 0 and p -> 1 where the normal approximation
# (mean +- 1.96 * sqrt(p(1-p)/n)) breaks down.  Reference: Wilson (1927).

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Return (p_hat, low, high) for a binomial proportion k/n, level 1 - alpha
    with z = Phi^{-1}(1 - alpha/2).  Default z = 1.96 -> 95% CI."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


# ---- I/O helpers ---------------------------------------------------------

def _summary(label: str, x: np.ndarray) -> str:
    n = len(x)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / sqrt(n) if n > 1 else 0.0
    return (
        f"{label:<14s} n={n:3d}  mean={mean:+.3f}  std={std:.3f}  "
        f"95% CI=[{mean - half:+.3f}, {mean + half:+.3f}]  "
        f"min={x.min():+.3f}  max={x.max():+.3f}"
    )


def _basin_counts(values: np.ndarray) -> Dict[str, int]:
    labels = [classify_basin(float(c)) for c in values]
    return {b: int(sum(1 for l in labels if l == b)) for b in BASIN_ORDER}


def _basin_table(label: str, values: np.ndarray) -> List[Dict[str, float]]:
    n = len(values)
    counts = _basin_counts(values)
    rows: List[Dict[str, float]] = []
    # Coop = partial_collusion OR supra_cartel
    coop_k = counts["partial_collusion"] + counts["supra_cartel"]
    p_coop, lo_coop, hi_coop = wilson_interval(coop_k, n)
    for basin in BASIN_ORDER:
        k = counts[basin]
        p, lo, hi = wilson_interval(k, n)
        rows.append({
            "regime": label,
            "basin": basin,
            "count": int(k),
            "proportion": round(p, 4),
            "wilson_95_low": round(lo, 4),
            "wilson_95_high": round(hi, 4),
        })
    rows.append({
        "regime": label,
        "basin": "cooperative_any (CI >= 0.20)",
        "count": int(coop_k),
        "proportion": round(p_coop, 4),
        "wilson_95_low": round(lo_coop, 4),
        "wilson_95_high": round(hi_coop, 4),
    })
    return rows


# ---- Plot ---------------------------------------------------------------

def _hist(ax: plt.Axes, data: np.ndarray, title: str) -> None:
    bins = np.linspace(-1.0, 2.0, 31)
    labels = [classify_basin(float(c)) for c in data]
    grouped = [
        data[np.array([l == k for l in labels])] for k in BASIN_ORDER
    ]
    ax.hist(
        grouped, bins=bins, stacked=True,
        color=[BASIN_COLOURS[k] for k in BASIN_ORDER],
        edgecolor="white", linewidth=0.7,
    )
    ax.axvline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
    ax.axvline(1.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
    ax.axvline(
        float(np.mean(data)), color="#e67e22", linestyle="-",
        lw=2.0, alpha=0.9,
        label=f"mean = {np.mean(data):+.2f}",
    )

    n = len(data)
    coop_k = int(sum(1 for c in data if classify_basin(float(c)) in
                     ("partial_collusion", "supra_cartel")))
    p, lo, hi = wilson_interval(coop_k, n)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-1.05, 2.05)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    ax.text(
        0.02, 0.97,
        (
            f"n = {n}\n"
            f"std = {np.std(data, ddof=1):.2f}\n"
            f"P(coop, CI>=0.20) = {p:.2f}\n"
            f"Wilson 95% = [{lo:.2f}, {hi:.2f}]"
        ),
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.8),
    )


def main() -> None:
    if not os.path.exists(TRIO_CSV):
        print(f"ERROR: missing {TRIO_CSV}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(DUO_CSV):
        print(f"ERROR: missing {DUO_CSV}", file=sys.stderr)
        sys.exit(1)

    trio = pd.read_csv(TRIO_CSV)
    duo = pd.read_csv(DUO_CSV)

    trio_tail = trio["collusion_index"].to_numpy()
    trio_greedy = trio["greedy_collusion_index"].to_numpy()
    duo_tail = duo["collusion_index"].to_numpy()
    duo_greedy = duo["greedy_collusion_index"].to_numpy()

    print("=" * 78)
    print("Per-seed distribution of the collusion index")
    print("=" * 78)
    print(_summary("trio tail", trio_tail))
    print(_summary("trio greedy", trio_greedy))
    print(_summary("duo  tail", duo_tail))
    print(_summary("duo  greedy", duo_greedy))
    print()

    # ---- Basin counts + Wilson on the greedy estimator -------------------
    print("Basin counts (greedy CI, headline 50-seed run)")
    print("-" * 78)
    rows = (
        _basin_table("triopoly_greedy", trio_greedy)
        + _basin_table("duopoly_greedy",  duo_greedy)
        + _basin_table("triopoly_tail",   trio_tail)
        + _basin_table("duopoly_tail",    duo_tail)
    )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote: {SUMMARY_CSV}")

    # ---- Figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey="row")
    _hist(axes[0, 0], trio_tail,
          "Triopoly (3 learners) - tail-of-training CI")
    _hist(axes[0, 1], trio_greedy,
          r"Triopoly (3 learners) - greedy rollout CI ($\epsilon$=0)")
    _hist(axes[1, 0], duo_tail,
          "Duopoly (OPEC+US learn, RUS myopic) - tail CI")
    _hist(axes[1, 1], duo_greedy,
          r"Duopoly (OPEC+US learn, RUS myopic) - greedy rollout CI ($\epsilon$=0)")

    for ax in axes[1, :]:
        ax.set_xlabel("Collusion index (0 = Nash, 1 = Cartel)")
    for ax in axes[:, 0]:
        ax.set_ylabel("# seeds")

    legend_handles = [
        Patch(color=BASIN_COLOURS[k], label=BASIN_LABEL[k]) for k in BASIN_ORDER
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=4, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Multi-agent Q-learning: per-seed distribution of the collusion index\n"
        "(50 independent seeds per regime, Option B headline run)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIG_PATH, dpi=160, bbox_inches="tight")
    print(f"Wrote: {FIG_PATH}")


if __name__ == "__main__":
    main()
