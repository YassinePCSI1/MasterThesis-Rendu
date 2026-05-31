"""Plotting utilities (matplotlib only)."""
from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .cournot_static import CournotResult

PALETTE = {
    "US": "#4c78a8",
    "OPEC": "#f58518",
    "RUS": "#54a24b",
    "nash": "#e45756",
    "cooperative": "#72b7b2",
    "stackelberg": "#b279a2",
}
PLAYER_COLORS = [PALETTE["US"], PALETTE["OPEC"], PALETTE["RUS"]]

# Regime colour scheme
REGIME_BG = {
    "cooperation": "#2d6a4f",   # dark green  (background alpha 0.10)
    "punishment":  "#c1121f",   # dark red
    "recovery":    "#1a6fb5",   # dark blue
    "transient":   "#8c6d3f",   # brown-ish (myopic ramp-up)
    "steady":      "#3a7d44",   # green (myopic near-Nash)
}


def _draw_regime_bands(
    axes: list,
    T: int,
    deviation_period: int,
    punishment_length: int,
) -> None:
    """Shade regime backgrounds and mark deviation point on a list of Axes.

    Phases
    ------
    Cooperation : t ∈ [0,  deviation_period)            → light green
    Punishment  : t ∈ [deviation_period, deviation_period + punishment_length]  → light red
    Recovery    : t ∈ [deviation_period + punishment_length + 1, T)  → light blue

    A dashed vertical line marks the deviation period.
    Text labels float at the top of the first axis.
    """
    from matplotlib.transforms import blended_transform_factory

    coop_end     = deviation_period - 0.5
    punish_end   = deviation_period + punishment_length + 0.5

    phases = [
        (-0.5,       coop_end,  REGIME_BG["cooperation"], "Cooperation phase"),
        (coop_end,   punish_end, REGIME_BG["punishment"],  "Punishment phase"),
        (punish_end, T - 0.5,   REGIME_BG["recovery"],    "Recovery"),
    ]

    for ax in axes:
        for x0, x1, color, _ in phases:
            ax.axvspan(x0, x1, alpha=0.10, color=color, zorder=0)
        ax.axvline(
            deviation_period, color=REGIME_BG["punishment"],
            linestyle="--", linewidth=1.8, alpha=0.90, zorder=6,
            label=f"Deviation  t={deviation_period}",
        )

    # Text labels on every axis (top, using blended transform)
    for ax in axes:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        # mid-x positions in data coords
        mids = [
            ((0 + coop_end) / 2,                 REGIME_BG["cooperation"], "Cooperation"),
            ((coop_end + punish_end) / 2,         REGIME_BG["punishment"],  "Punishment"),
            ((punish_end + (T - 0.5)) / 2,        REGIME_BG["recovery"],    "Recovery"),
        ]
        for xm, color, label in mids:
            if xm < 0 or xm > T:
                continue
            ax.text(
                xm, 0.975, label,
                ha="center", va="top",
                fontsize=7.5, fontweight="bold",
                color=color, alpha=0.85,
                transform=trans, clip_on=True,
                zorder=7,
            )


def _draw_convergence_bands(
    axes: list,
    prices: List[float],
    nash_price: float,
    tol: float = 1.0,
) -> None:
    """Shade transient and near-equilibrium zones for myopic convergence plots.

    Detects the first period where |P_t - P_nash| < tol and draws:
    - Transient zone  (tan background)   : t ∈ [0, convergence_period)
    - Steady-state zone (green background): t ∈ [convergence_period, T)
    A dotted vertical line marks the convergence period.
    """
    from matplotlib.transforms import blended_transform_factory

    T = len(prices)
    # find first period in steady state (allow small overshoot)
    conv_t = T
    for t, p in enumerate(prices):
        if abs(p - nash_price) < tol:
            conv_t = t
            break

    for ax in axes:
        ax.axvspan(-0.5, conv_t - 0.5, alpha=0.08, color=REGIME_BG["transient"], zorder=0)
        ax.axvspan(conv_t - 0.5, T - 0.5, alpha=0.08, color=REGIME_BG["steady"], zorder=0)
        ax.axvline(conv_t, color=REGIME_BG["steady"], linestyle=":",
                   linewidth=1.6, alpha=0.80, zorder=6,
                   label=f"Near-Nash  t={conv_t}")

    # Labels
    for ax in axes:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        mid_transient = conv_t / 2
        mid_steady    = conv_t + (T - conv_t) / 2
        for xm, color, label in [
            (mid_transient, REGIME_BG["transient"], "Transient"),
            (mid_steady,    REGIME_BG["steady"],    "Near Nash"),
        ]:
            if xm <= 0 or xm >= T:
                continue
            ax.text(
                xm, 0.975, label,
                ha="center", va="top",
                fontsize=7.5, fontweight="bold",
                color=color, alpha=0.85,
                transform=trans, clip_on=True,
                zorder=7,
            )


def plot_duopoly_vs_triopoly(duo: CournotResult, tri: CournotResult, path: str) -> None:
    labels = ["Duopoly", "Triopoly"]
    quantities = [duo.total_quantity, tri.total_quantity]
    prices = [duo.price, tri.price]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(labels, quantities, color="#4c78a8", alpha=0.8, label="Total Q")
    ax1.set_ylabel("Total Q (mbd)")

    ax2 = ax1.twinx()
    ax2.plot(labels, prices, color="#f58518", marker="o", label="Price")
    ax2.set_ylabel("Price ($/bbl)")

    ax1.set_title("Adding a strategic producer increases Q and lowers P")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_quantity_comparison(duo: CournotResult, tri: CournotResult, path: str) -> None:
    players = ["US", "OPEC", "RUS"]
    duo_vals = [duo.quantities.get(p, 0.0) for p in players]
    tri_vals = [tri.quantities.get(p, 0.0) for p in players]

    x = np.arange(len(players))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, duo_vals, width, label="Duopoly", color="#4c78a8", alpha=0.8)
    ax.bar(x + width / 2, tri_vals, width, label="Triopoly", color="#72b7b2", alpha=0.8)
    ax.set_xticks(x, players)
    ax.set_ylabel("Output (mbd)")
    ax.set_title("Output by player: duopoly vs triopoly")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_profit_comparison(duo: CournotResult, tri: CournotResult, path: str) -> None:
    players = ["US", "OPEC", "RUS"]
    duo_vals = [duo.profits.get(p, 0.0) for p in players]
    tri_vals = [tri.profits.get(p, 0.0) for p in players]

    x = np.arange(len(players))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, duo_vals, width, label="Duopoly", color="#f58518", alpha=0.8)
    ax.bar(x + width / 2, tri_vals, width, label="Triopoly", color="#e45756", alpha=0.8)
    ax.set_xticks(x, players)
    ax.set_ylabel("Profit (arbitrary units)")
    ax.set_title("Profit by player: duopoly vs triopoly")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_comparative_statics(results: List[Dict[str, float]], path: str) -> None:
    """Comparative statics: triopoly Cournot price and total output vs. demand
    intercept a.  Plotted as two side-by-side panels with discrete markers so
    the reader can see each sweep point individually (a dual-y-axis line plot
    is misleading because the two series happen to have similar slope shapes).
    """
    x = [r["param_value"] for r in results]
    prices = [r["price"] for r in results]
    quantities = [r["total_quantity"] for r in results]

    fig, (ax_p, ax_q) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_p.scatter(x, prices, color=PALETTE["OPEC"], marker="o", s=80,
                 edgecolor="black", linewidths=0.7, zorder=5,
                 label="Equilibrium price (simulated)")
    ax_p.set_xlabel("Demand intercept a")
    ax_p.set_ylabel("Equilibrium price ($/bbl)")
    ax_p.set_title("Price vs. demand intercept  (slope dP/da = 1/4)")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="best", fontsize=9)

    ax_q.scatter(x, quantities, color=PALETTE["US"], marker="s", s=80,
                 edgecolor="black", linewidths=0.7, zorder=5,
                 label="Total output Q (simulated)")
    ax_q.set_xlabel("Demand intercept a")
    ax_q.set_ylabel("Total Cournot output Q (mbd)")
    ax_q.set_title("Output vs. demand intercept  (slope dQ/da = 3/4)")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="best", fontsize=9)

    fig.suptitle("Comparative statics — triopoly Cournot equilibrium",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_time_series(prices: List[float],
                     quantities: List[Dict[str, float]],
                     path: str,
                     nash_price: float | None = None) -> None:
    """Myopic repeated-game dynamics with transient / near-Nash regime shading."""
    T = len(prices)
    ts = list(range(T))

    fig, (ax_q, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- top: per-player quantities ---
    for player, col in zip(quantities[0].keys(), PLAYER_COLORS):
        series = [q[player] for q in quantities]
        ax_q.plot(ts, series, color=col, linewidth=2, label=player)
    ax_q.set_ylabel("Output (mbd)")
    ax_q.legend(fontsize=9)
    ax_q.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- bottom: price ---
    ax_p.plot(ts, prices, color=PALETTE["nash"], linewidth=2.5, label="Market price")
    if nash_price is not None:
        ax_p.axhline(nash_price, color="red", linestyle="--", linewidth=1.4,
                     alpha=0.8, label=f"Nash P = {nash_price:.1f}")
    ax_p.set_ylabel("Price ($/bbl)")
    ax_p.set_xlabel("Period")
    ax_p.legend(fontsize=9)

    # Convergence shading on both panels
    if nash_price is not None:
        _draw_convergence_bands([ax_q, ax_p], prices, nash_price)

    fig.suptitle("Myopic Cournot Dynamics — Convergence to Nash Equilibrium",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_repeated_price_quantity(prices: List[float], quantities: List[float], path: str,
                                  nash_price: float | None = None,
                                  nash_Q: float | None = None) -> None:
    """Price and total output convergence with transient / near-Nash shading."""
    T = len(prices)
    ts = list(range(T))

    fig, (ax_p, ax_q) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_p.plot(ts, prices, color=PALETTE["nash"], linewidth=2.5, label="Market price")
    if nash_price is not None:
        ax_p.axhline(nash_price, color="red", linestyle="--", linewidth=1.4,
                     alpha=0.8, label=f"Nash P = {nash_price:.1f}")
    ax_p.set_ylabel("Price ($/bbl)")
    ax_p.legend(fontsize=9)

    ax_q.plot(ts, quantities, color=PALETTE["US"], linewidth=2.5, label="Total Q")
    if nash_Q is not None:
        ax_q.axhline(nash_Q, color="red", linestyle="--", linewidth=1.4,
                     alpha=0.8, label=f"Nash Q = {nash_Q:.1f}")
    ax_q.set_ylabel("Total Q (mbd)")
    ax_q.set_xlabel("Period")
    ax_q.legend(fontsize=9)
    ax_q.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if nash_price is not None:
        _draw_convergence_bands([ax_p, ax_q], prices, nash_price)

    fig.suptitle("Price & Quantity Dynamics — Myopic Best-Response",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_repeated_profits(
    profits: List[Dict[str, float]],
    path: str,
    nash_profits: Dict[str, float] | None = None,
    coop_profits: Dict[str, float] | None = None,
    nash_price: float | None = None,
    prices: List[float] | None = None,
) -> None:
    """Per-player profit dynamics with Nash/cooperative benchmarks and convergence shading."""
    from matplotlib.lines import Line2D

    players = list(profits[0].keys())
    color_cycle = [PALETTE.get(p, f"C{i}") for i, p in enumerate(players)]
    T = len(profits)
    ts = list(range(T))

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, player in enumerate(players):
        series = [p[player] for p in profits]
        ax.plot(ts, series, label=player, color=color_cycle[i], linewidth=2.2, zorder=5)

        if nash_profits and player in nash_profits:
            ax.axhline(
                nash_profits[player], color=color_cycle[i],
                linestyle="--", linewidth=1.3, alpha=0.65, zorder=4,
            )
        if coop_profits and player in coop_profits:
            ax.axhline(
                coop_profits[player], color=color_cycle[i],
                linestyle=":", linewidth=1.3, alpha=0.65, zorder=4,
            )

    extra = [
        Line2D([0], [0], linestyle="--", color="grey", lw=1.3, label="Nash benchmark"),
        Line2D([0], [0], linestyle=":", color="grey", lw=1.3, label="Cartel quota benchmark"),
    ]
    handles, lbs = ax.get_legend_handles_labels()
    ax.legend(handles=handles + (extra if nash_profits else []), fontsize=8, loc="lower right")
    ax.set_title("Profit Dynamics — Convergence to Nash Equilibrium", fontweight="bold")
    ax.set_xlabel("Period")
    ax.set_ylabel("Per-period profit (model units)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Convergence shading if price series is provided
    if nash_price is not None and prices is not None:
        _draw_convergence_bands([ax], prices, nash_price)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_repeated_dynamics_combined(
    prices: List[float],
    total_quantities: List[float],
    per_player_quantities: List[Dict[str, float]],
    per_player_profits: List[Dict[str, float]],
    path: str,
    nash_price: float | None = None,
    nash_total_q: float | None = None,
    nash_profits: Dict[str, float] | None = None,
    cartel_profits: Dict[str, float] | None = None,
) -> None:
    """Single 2x2 figure consolidating the four near-duplicate myopic
    convergence plots (per-player quantities, total quantity, price, and
    profits) that were previously emitted as four separate files.  Built
    once from the raw repeated-game simulation; the four standalone images
    (repeated_time_series, repeated_price_quantity, repeated_profit_time_
    series, repeated_nash_convergence) are kept on disk for backward
    compatibility but the thesis should cite this combined figure.
    """
    from matplotlib.lines import Line2D

    T = len(prices)
    ts = list(range(T))
    players = list(per_player_quantities[0].keys())
    color_cycle = [PALETTE.get(p, f"C{i}") for i, p in enumerate(players)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    ax_q_player, ax_q_total = axes[0]
    ax_p, ax_pi = axes[1]

    # --- Top-left: per-player quantities ---
    for i, p in enumerate(players):
        series = [q[p] for q in per_player_quantities]
        ax_q_player.plot(ts, series, color=color_cycle[i], linewidth=2.2,
                         label=p, zorder=5)
    ax_q_player.set_ylabel("Output (mbd)")
    ax_q_player.set_title("Per-player output trajectories")
    ax_q_player.legend(fontsize=9, loc="best")
    ax_q_player.grid(True, alpha=0.3)
    ax_q_player.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Top-right: industry total quantity ---
    ax_q_total.plot(ts, total_quantities, color=PALETTE["US"], linewidth=2.5,
                    label="Total Q", zorder=5)
    if nash_total_q is not None:
        ax_q_total.axhline(nash_total_q, color="red", linestyle="--",
                           linewidth=1.4, alpha=0.8,
                           label=f"Nash Q = {nash_total_q:.1f}")
    ax_q_total.set_ylabel("Total output (mbd)")
    ax_q_total.set_title("Industry-aggregate output")
    ax_q_total.legend(fontsize=9, loc="best")
    ax_q_total.grid(True, alpha=0.3)
    ax_q_total.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Bottom-left: market price ---
    ax_p.plot(ts, prices, color=PALETTE["nash"], linewidth=2.5,
              label="Market price", zorder=5)
    if nash_price is not None:
        ax_p.axhline(nash_price, color="red", linestyle="--", linewidth=1.4,
                     alpha=0.8, label=f"Nash P = {nash_price:.1f}")
    ax_p.set_ylabel("Price ($/bbl)")
    ax_p.set_xlabel("Period")
    ax_p.set_title("Market price")
    ax_p.legend(fontsize=9, loc="best")
    ax_p.grid(True, alpha=0.3)
    ax_p.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Bottom-right: per-player profits ---
    for i, p in enumerate(players):
        series = [pi[p] for pi in per_player_profits]
        ax_pi.plot(ts, series, color=color_cycle[i], linewidth=2.2,
                   label=p, zorder=5)
        if nash_profits and p in nash_profits:
            ax_pi.axhline(nash_profits[p], color=color_cycle[i],
                          linestyle="--", linewidth=1.1, alpha=0.6, zorder=3)
        if cartel_profits and p in cartel_profits:
            ax_pi.axhline(cartel_profits[p], color=color_cycle[i],
                          linestyle=":", linewidth=1.1, alpha=0.6, zorder=3)
    benchmark_legend = []
    if nash_profits:
        benchmark_legend.append(
            Line2D([0], [0], linestyle="--", color="grey", lw=1.2,
                   label="Nash benchmark")
        )
    if cartel_profits:
        benchmark_legend.append(
            Line2D([0], [0], linestyle=":", color="grey", lw=1.2,
                   label="Cartel quota benchmark")
        )
    handles, _ = ax_pi.get_legend_handles_labels()
    ax_pi.legend(handles=handles + benchmark_legend, fontsize=8, loc="best")
    ax_pi.set_ylabel("Per-period profit (model units)")
    ax_pi.set_xlabel("Period")
    ax_pi.set_title("Per-player profit trajectories")
    ax_pi.grid(True, alpha=0.3)
    ax_pi.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Convergence shading on every panel (uses the same price series for
    # the regime boundary so the four panels stay in sync).
    if nash_price is not None:
        _draw_convergence_bands(
            [ax_q_player, ax_q_total, ax_p, ax_pi], prices, nash_price,
        )

    fig.suptitle("Myopic Cournot dynamics — convergence to Nash equilibrium",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_punishment_regimes(
    punishment_outcome,
    deviation_period: int,
    punishment_length: int,
    nash_result,
    cartel_result,
    path: str,
) -> None:
    """Comprehensive punishment scenario plot with coloured regime backgrounds.

    Three panels (sharing x-axis):
    ┌─────────────────────────────────┐
    │  Panel 1: per-player quantities │
    ├─────────────────────────────────┤
    │  Panel 2: market price          │
    ├─────────────────────────────────┤
    │  Panel 3: per-player profits    │
    └─────────────────────────────────┘

    Background colours
    ------------------
    Green  → Cooperation phase  (t < deviation_period)
    Red    → Punishment phase   (deviation_period ≤ t ≤ deviation_period + punishment_length)
    Blue   → Recovery phase     (t > deviation_period + punishment_length)

    The dashed red vertical line marks the deviation period.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    players = list(punishment_outcome.quantities[0].keys())
    T = len(punishment_outcome.prices)
    ts = list(range(T))
    color_cycle = [PALETTE.get(p, f"C{i}") for i, p in enumerate(players)]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax_q, ax_p, ax_pi = axes

    # --- Panel 1: quantities ---
    for i, player in enumerate(players):
        series = [punishment_outcome.quantities[t][player] for t in range(T)]
        ax_q.plot(ts, series, color=color_cycle[i], linewidth=2.2, label=player, zorder=5)
        # Nash and cartel quota benchmarks
        if hasattr(nash_result, "quantities") and player in nash_result.quantities:
            ax_q.axhline(nash_result.quantities[player], color=color_cycle[i],
                         linestyle="--", linewidth=1.2, alpha=0.6, zorder=4)
        if hasattr(cartel_result, "quotas") and player in cartel_result.quotas:
            ax_q.axhline(cartel_result.quotas[player], color=color_cycle[i],
                         linestyle=":", linewidth=1.2, alpha=0.6, zorder=4)
    ax_q.set_ylabel("Output (mbd)", fontsize=10)
    ax_q.legend(fontsize=9, loc="upper right")

    # --- Panel 2: market price ---
    ax_p.plot(ts, punishment_outcome.prices, color=PALETTE["nash"],
              linewidth=2.5, label="Market price", zorder=5)
    if hasattr(nash_result, "price"):
        ax_p.axhline(nash_result.price, color="red", linestyle="--", linewidth=1.4,
                     alpha=0.7, label=f"Nash P = {nash_result.price:.1f}", zorder=4)
    if hasattr(cartel_result, "quota_price"):
        ax_p.axhline(cartel_result.quota_price, color=PALETTE["cooperative"],
                     linestyle=":", linewidth=1.4, alpha=0.7,
                     label=f"Cartel P = {cartel_result.quota_price:.1f}", zorder=4)
    ax_p.set_ylabel("Price ($/bbl)", fontsize=10)
    ax_p.legend(fontsize=9, loc="upper right")

    # --- Panel 3: profits ---
    for i, player in enumerate(players):
        series = [punishment_outcome.profits[t][player] for t in range(T)]
        ax_pi.plot(ts, series, color=color_cycle[i], linewidth=2.2, label=player, zorder=5)
        if hasattr(nash_result, "profits") and player in nash_result.profits:
            ax_pi.axhline(nash_result.profits[player], color=color_cycle[i],
                          linestyle="--", linewidth=1.2, alpha=0.6, zorder=4)
        if hasattr(cartel_result, "quota_profits") and player in cartel_result.quota_profits:
            ax_pi.axhline(cartel_result.quota_profits[player], color=color_cycle[i],
                          linestyle=":", linewidth=1.2, alpha=0.6, zorder=4)
    ax_pi.set_ylabel("Per-period profit", fontsize=10)
    ax_pi.set_xlabel("Period", fontsize=10)
    ax_pi.legend(fontsize=9, loc="upper right")
    ax_pi.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Draw regime bands on all three panels
    _draw_regime_bands(list(axes), T, deviation_period, punishment_length)

    # Shared legend for regime patches + line styles
    regime_patches = [
        Patch(facecolor=REGIME_BG["cooperation"], alpha=0.25, label="Cooperation phase"),
        Patch(facecolor=REGIME_BG["punishment"],  alpha=0.25, label="Punishment phase"),
        Patch(facecolor=REGIME_BG["recovery"],    alpha=0.25, label="Recovery phase"),
        Line2D([0], [0], linestyle="--", color=REGIME_BG["punishment"],
               linewidth=1.8, label=f"Deviation  t={deviation_period}"),
        Line2D([0], [0], linestyle="--", color="grey", linewidth=1.2, label="Nash benchmark"),
        Line2D([0], [0], linestyle=":",  color="grey", linewidth=1.2, label="Cartel quota"),
    ]
    fig.legend(handles=regime_patches,
               loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.suptitle(
        f"Tacit Cooperation with Deviation & Punishment\n"
        f"(deviation at t={deviation_period}, punishment length={punishment_length} periods)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curve(rewards: List[float], steps_per_episode: int, path: str) -> None:
    rewards = np.array(rewards)
    episode_rewards = rewards.reshape(-1, steps_per_episode).mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episode_rewards, color="#4c78a8")
    ax.set_title("Learning curve (average reward per episode)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_action_distribution(q_history: List[Dict[str, float]], player: str, path: str) -> None:
    actions = [q[player] for q in q_history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(actions, bins=20, color="#54a24b", alpha=0.8)
    ax.set_title(f"Action distribution for {player}")
    ax.set_xlabel("Output (mbd)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rl_rolling_outputs(q_history: List[Dict[str, float]], window: int, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    players = list(q_history[0].keys())
    for player in players:
        series = np.array([q[player] for q in q_history], dtype=float)
        if len(series) < window:
            rolling = series
        else:
            rolling = np.convolve(series, np.ones(window) / window, mode="valid")
        ax.plot(rolling, label=f"q_{player}")
    ax.set_title(f"RL rolling average outputs (window={window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Output (mbd)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-agent RL plots (Section 8b)
# ---------------------------------------------------------------------------

def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centred-trailing rolling mean, returning an array of the same length
    as the input (left-edge pads with the first valid mean)."""
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) <= window:
        return x
    rolled = np.convolve(x, np.ones(window) / window, mode="valid")
    pad = np.full(window - 1, rolled[0])
    return np.concatenate([pad, rolled])


def _stack_price_history(outcomes) -> "np.ndarray":
    """Stack price trajectories of a list of MultiAgentRLOutcome objects.

    Returns a 2-D array of shape ``(n_seeds, n_steps)`` truncated to the
    shortest trajectory length, so percentile operations are well-defined.
    """
    arrs = [np.asarray(o.price_history, dtype=float) for o in outcomes]
    n_steps = min(a.shape[0] for a in arrs)
    return np.stack([a[:n_steps] for a in arrs], axis=0)


def _stack_q_history(outcomes, player: str) -> "np.ndarray":
    """Stack ``q_history[player]`` across a list of outcomes."""
    arrs = [
        np.asarray([qd[player] for qd in o.q_history], dtype=float)
        for o in outcomes
    ]
    n_steps = min(a.shape[0] for a in arrs)
    return np.stack([a[:n_steps] for a in arrs], axis=0)


def plot_multiagent_price_convergence(
    outcome_or_outcomes,
    nash_price: float,
    cartel_price: float,
    path: str,
    *,
    rolling_window: int = 50,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
) -> None:
    """Multi-seed price-convergence plot with a 95% percentile band.

    Accepts either a single :class:`MultiAgentRLOutcome` (legacy callers) or
    a list of outcomes (recommended).  In the multi-seed case the figure
    shows the per-step **mean** price (rolling-mean smoothed) together with
    the per-step **2.5%–97.5% percentile band** across seeds — the proper
    visual analogue of a 95% empirical confidence interval for the price
    trajectory.  In the single-seed case we still draw the rolling-mean,
    but we annotate the figure to make clear that **no statistical claim**
    can be made from a single trajectory.
    """
    outcomes = (
        list(outcome_or_outcomes)
        if isinstance(outcome_or_outcomes, list) else [outcome_or_outcomes]
    )
    prices_mat = _stack_price_history(outcomes)  # (n_seeds, n_steps)
    n_seeds, n_steps = prices_mat.shape

    # Smooth each seed independently, then aggregate.
    rolling = np.stack(
        [_rolling_mean(prices_mat[s], window=rolling_window) for s in range(n_seeds)],
        axis=0,
    )
    mean_curve = rolling.mean(axis=0)
    if n_seeds > 1:
        lo = np.percentile(rolling, ci_low, axis=0)
        hi = np.percentile(rolling, ci_high, axis=0)
    else:
        lo = hi = mean_curve
    steps = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.fill_between(steps, nash_price, cartel_price, color=PALETTE["cooperative"],
                    alpha=0.10, label="Collusion corridor (Nash → Cartel)")
    if n_seeds > 1:
        ax.fill_between(steps, lo, hi, color=PALETTE["OPEC"], alpha=0.22,
                        label=f"95% band ({ci_low:.1f}–{ci_high:.1f} pct., n={n_seeds} seeds)")
    ax.plot(steps, mean_curve, color=PALETTE["OPEC"], linewidth=1.7,
            label=f"Mean rolling price (window = {rolling_window})")
    ax.axhline(nash_price, color=PALETTE["nash"], linestyle="--", linewidth=1.2,
               label=f"Nash benchmark ({nash_price:.1f} $/bbl)")
    ax.axhline(cartel_price, color=PALETTE["cooperative"], linestyle="--", linewidth=1.2,
               label=f"Cartel benchmark ({cartel_price:.1f} $/bbl)")

    cis = [o.collusion_index for o in outcomes]
    cps = [o.converged_price for o in outcomes]
    if n_seeds > 1:
        info_text = (
            f"Collusion index = {np.mean(cis):.2f} ± {np.std(cis, ddof=1):.2f}\n"
            f"Converged P = {np.mean(cps):.2f} ± {np.std(cps, ddof=1):.2f} $/bbl\n"
            f"n seeds = {n_seeds}"
        )
    else:
        info_text = (
            f"Collusion index = {cis[0]:.2f}\n"
            f"Converged P = {cps[0]:.2f} $/bbl\n"
            "single seed — no CI"
        )
    ax.text(
        0.985, 0.965, info_text,
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="grey", alpha=0.92),
    )

    ax.set_xlabel("Step (concatenated across episodes)")
    ax.set_ylabel("Market price ($/bbl)")
    title = (
        "Multi-Agent RL: Price Convergence — Mean ± 95% Band"
        if n_seeds > 1 else
        "Multi-Agent RL: Price Convergence (single seed — illustrative only)"
    )
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    y_top = max(cartel_price + 5, float(np.nanmax(hi)) + 2)
    y_bot = min(nash_price - 5, float(np.nanmin(lo)) - 2)
    ax.set_ylim(y_bot, y_top)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multiagent_outputs(
    outcome_or_outcomes,
    nash_q: Dict[str, float],
    cartel_q: Dict[str, float],
    players: List[str],
    path: str,
    *,
    rolling_window: int = 50,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
) -> None:
    """Per-player output trajectories with a 95% percentile band per player.

    Accepts a single outcome (legacy) or a list (recommended).  In the
    multi-seed case each player gets the **mean rolling output** as a solid
    line and a shaded 2.5%–97.5% band across seeds.  Non-learning players
    (myopic best-response) are still drawn, with a dotted line to mark them
    as rational — not learned — behaviour.
    """
    outcomes = (
        list(outcome_or_outcomes)
        if isinstance(outcome_or_outcomes, list) else [outcome_or_outcomes]
    )
    n_seeds = len(outcomes)
    fig, ax = plt.subplots(figsize=(10, 4.8))

    learners = outcomes[0].learning_players
    for p in players:
        mat = _stack_q_history(outcomes, p)
        rolled = np.stack([_rolling_mean(mat[s], window=rolling_window)
                           for s in range(n_seeds)], axis=0)
        mean = rolled.mean(axis=0)
        is_learner = p in learners
        ls = "-" if is_learner else ":"
        lbl = f"q_{p}" + ("" if is_learner else " (myopic)")
        color = PALETTE.get(p, "grey")
        if n_seeds > 1:
            lo = np.percentile(rolled, ci_low, axis=0)
            hi = np.percentile(rolled, ci_high, axis=0)
            ax.fill_between(np.arange(mean.shape[0]), lo, hi,
                            color=color, alpha=0.15)
        ax.plot(mean, color=color, linewidth=1.7 if is_learner else 1.2,
                linestyle=ls, label=lbl)
        if is_learner:
            ax.axhline(nash_q[p], color=color, linestyle="--",
                       linewidth=0.9, alpha=0.6,
                       label=f"q_{p} Nash ({nash_q[p]:.1f})")
            ax.axhline(cartel_q[p], color=color, linestyle=":",
                       linewidth=0.9, alpha=0.6,
                       label=f"q_{p} Cartel ({cartel_q[p]:.1f})")

    ax.set_xlabel("Step (concatenated across episodes)")
    ax.set_ylabel("Output (mbd)")
    title = (
        f"Multi-Agent RL: Player Outputs — Mean ± 95% Band (n={n_seeds} seeds)"
        if n_seeds > 1 else
        "Multi-Agent RL: Player Outputs (single seed — illustrative only)"
    )
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multiagent_vs_singleagent(
    single_outcome,
    multi_outcome,
    nash_price: float,
    cartel_price: float,
    path: str,
) -> None:
    """Bar comparison of converged price: Single-agent RL vs Multi-agent RL
    vs Nash and Cartel benchmarks.

    The single-agent outcome here is the existing OPEC-only Q-learner from
    ``rl_agent.py`` (RUS and US play deterministic best-responses).  Multi-agent
    additionally has US learning simultaneously, observing only the price.
    A higher converged price under multi-agent (relative to single-agent)
    is *the* empirical signature that learner reciprocity amplifies tacit
    collusion — a quantitative validation of the Folk-Theorem prediction.
    """
    # Single-agent converged price = mean of last 200 steps (matches existing report)
    sa_prices = np.asarray(single_outcome.price_history, dtype=float)
    sa_tail = sa_prices[-min(200, len(sa_prices)):]
    sa_price = float(sa_tail.mean())

    ma_price = float(multi_outcome.converged_price)

    labels = ["Nash\n(triopoly)", "Single-agent RL\n(OPEC only)",
              "Multi-agent RL\n(OPEC + US, price only)", "Cartel\n(cooperative)"]
    values = [nash_price, sa_price, ma_price, cartel_price]
    colors = [PALETTE["nash"], PALETTE["US"], PALETTE["OPEC"], PALETTE["cooperative"]]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(labels, values, color=colors, alpha=0.88, edgecolor="black", linewidth=0.6)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Compute and annotate uplift vs single-agent
    uplift = ma_price - sa_price
    denom = (cartel_price - nash_price)
    sa_collusion = (sa_price - nash_price) / denom if abs(denom) > 1e-9 else 0.0
    ma_collusion = (ma_price - nash_price) / denom if abs(denom) > 1e-9 else 0.0

    ax.set_ylabel("Converged market price ($/bbl)")
    ax.set_title("Tacit Collusion Amplification: Single- vs Multi-Agent RL\n"
                 f"Δ collusion index : {sa_collusion:.2f} → {ma_collusion:.2f}   "
                 f"(price uplift: {uplift:+.2f} $/bbl)",
                 fontweight="bold")
    ax.set_ylim(0, max(values) * 1.18)
    ax.axhline(nash_price, color=PALETTE["nash"], linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(cartel_price, color=PALETTE["cooperative"], linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multiagent_collusion_decomposition(
    outcome,
    nash_q: Dict[str, float],
    cartel_q: Dict[str, float],
    players: List[str],
    path: str,
) -> None:
    """Per-agent decomposition of the collusion outcome.

    For each learning agent, plot its converged output against its Nash and
    cartel benchmarks (horizontal markers), so the reader can see *which*
    agent is doing the quota-cutting.  This complements the aggregate
    collusion index by exposing asymmetries (e.g.\\ does OPEC restrict more
    than US, given its lower marginal cost?).
    """
    learners = list(outcome.learning_players)
    n = len(learners)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(7, 3 * n + 2), 5))
    x = np.arange(n)
    width = 0.22

    nash_vals = [nash_q[p] for p in learners]
    cartel_vals = [cartel_q[p] for p in learners]
    rl_vals = [outcome.converged_outputs[p] for p in learners]

    ax.bar(x - width, nash_vals, width, color=PALETTE["nash"], alpha=0.85,
           label="Nash benchmark", edgecolor="black", linewidth=0.5)
    ax.bar(x, rl_vals, width, color=PALETTE["OPEC"], alpha=0.95,
           label="Multi-agent RL converged", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, cartel_vals, width, color=PALETTE["cooperative"], alpha=0.85,
           label="Cartel benchmark", edgecolor="black", linewidth=0.5)

    for i, p in enumerate(learners):
        ax.text(x[i] - width, nash_vals[i] + 0.4, f"{nash_vals[i]:.1f}",
                ha="center", fontsize=8)
        ax.text(x[i], rl_vals[i] + 0.4, f"{rl_vals[i]:.1f}",
                ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i] + width, cartel_vals[i] + 0.4, f"{cartel_vals[i]:.1f}",
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(learners)
    ax.set_ylabel("Output (mbd)")
    ax.set_title("Multi-Agent RL: Per-Agent Quota Behaviour\n"
                 "Does each learner individually cut output toward the cartel target?",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(nash_vals), max(rl_vals), max(cartel_vals)) * 1.18)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multiagent_robustness_distribution(
    robustness_result: Dict,
    path: str,
    single_agent_collusion: float = 0.12,
) -> None:
    """Distribution of the collusion index across independent training seeds.

    A violin + boxplot combination shows whether the multi-agent collusion
    result is a robust statistical effect or an idiosyncratic single-seed
    artefact.  Reference lines:

    * ``collusion_index = 0``  : pure Nash benchmark.
    * ``collusion_index = 1``  : full cartel benchmark.
    * dashed red line          : single-agent RL collusion index (≈ 0.12),
                                 the level we have to beat to claim that
                                 multi-agent reciprocity amplifies collusion.

    The annotation reports the mean ± std and the 95% confidence interval
    of the mean (``mean ± 1.96 · std / √n``).
    """
    indices = np.asarray(robustness_result["collusion_indices"], dtype=float)
    mean_ci = robustness_result["mean_collusion_index"]
    std_ci = robustness_result["std_collusion_index"]
    ci_low = robustness_result["ci_95_low"]
    ci_high = robustness_result["ci_95_high"]
    n = robustness_result["n_seeds"]

    fig, ax = plt.subplots(figsize=(8, 5.2))

    parts = ax.violinplot(indices, positions=[0], widths=0.6, showmeans=False,
                          showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(PALETTE["OPEC"])
        body.set_edgecolor("black")
        body.set_alpha(0.45)

    bp = ax.boxplot(indices, positions=[0], widths=0.18, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                   markeredgecolor="black", markersize=7))
    for box in bp["boxes"]:
        box.set_facecolor(PALETTE["OPEC"])
        box.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)

    rng_jit = np.random.default_rng(42)
    jitter = rng_jit.uniform(-0.08, 0.08, size=len(indices))
    ax.scatter(jitter, indices, s=28, color="black", alpha=0.55, zorder=3,
               edgecolors="white", linewidths=0.6, label=f"individual seeds (n={n})")

    ax.axhline(0.0, color=PALETTE["nash"], linestyle="--", linewidth=1.1,
               label="Nash (collusion = 0)")
    ax.axhline(1.0, color=PALETTE["cooperative"], linestyle="--", linewidth=1.1,
               label="Cartel (collusion = 1)")
    ax.axhline(single_agent_collusion, color="#c1121f", linestyle=":", linewidth=1.6,
               label=f"Single-agent RL ({single_agent_collusion:.2f})")

    ax.text(
        0.97, 0.97,
        f"Mean = {mean_ci:.3f}\nStd  = {std_ci:.3f}\n"
        f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]\nn = {n} seeds",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="grey", alpha=0.92),
    )

    ax.set_xticks([0])
    ax.set_xticklabels(["Multi-Agent RL\n(OPEC + US)"])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel("Collusion index  (P − P_nash) / (P_cartel − P_nash)")
    ax.set_ylim(min(-0.15, float(indices.min()) - 0.05),
                max(1.05, float(indices.max()) + 0.05))
    ax.set_title(f"Multi-Agent RL Robustness: Collusion Index Distribution (n = {n} seeds)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multiagent_robustness_prices(
    robustness_result: Dict,
    nash_price: float,
    cartel_price: float,
    path: str,
) -> None:
    """Distribution of the converged market price across independent seeds.

    Boxplot on the price scale (dollars per barrel) makes the magnitude of
    the price uplift directly readable.  Nash and cartel benchmarks frame
    the plausible band; the annotation reports the sample mean ± std.
    """
    prices = np.asarray(robustness_result["converged_prices"], dtype=float)
    mean_p = robustness_result["mean_converged_price"]
    std_p = robustness_result["std_converged_price"]
    n = robustness_result["n_seeds"]

    fig, ax = plt.subplots(figsize=(8, 5.2))

    bp = ax.boxplot(prices, positions=[0], widths=0.35, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                   markeredgecolor="black", markersize=7))
    for box in bp["boxes"]:
        box.set_facecolor(PALETTE["OPEC"])
        box.set_alpha(0.80)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)

    rng_jit = np.random.default_rng(7)
    jitter = rng_jit.uniform(-0.10, 0.10, size=len(prices))
    ax.scatter(jitter, prices, s=28, color="black", alpha=0.55, zorder=3,
               edgecolors="white", linewidths=0.6, label=f"individual seeds (n={n})")

    ax.axhline(nash_price, color=PALETTE["nash"], linestyle="--", linewidth=1.1,
               label=f"Nash benchmark ({nash_price:.1f} $/bbl)")
    ax.axhline(cartel_price, color=PALETTE["cooperative"], linestyle="--", linewidth=1.1,
               label=f"Cartel benchmark ({cartel_price:.1f} $/bbl)")

    ax.text(
        0.97, 0.97,
        f"Mean = {mean_p:.2f} $/bbl\nStd  = {std_p:.2f} $/bbl\nn = {n} seeds",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="grey", alpha=0.92),
    )

    ax.set_xticks([0])
    ax.set_xticklabels(["Multi-Agent RL\n(converged price)"])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel("Converged market price ($/bbl)")
    margin = max(2.0, std_p * 1.5)
    ax.set_ylim(min(nash_price - 1.5, float(prices.min()) - margin),
                max(cartel_price + 1.5, float(prices.max()) + margin))
    ax.set_title(f"Converged Market Price Distribution Across Seeds (n = {n})",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Emergent punishment plots (Section 8b — Folk Theorem / Green-Porter signature)
# ---------------------------------------------------------------------------

def plot_punishment_detection(
    outcome,
    episodes: List,
    nash_price: float,
    cartel_price: float,
    path: str,
    window: int = 50,
) -> None:
    """Visualise emergent punishment episodes on the multi-agent trajectory.

    Two stacked panels share the x-axis (time step):

    * upper panel : rolling-mean market price with Nash and cartel benchmarks,
                    with each detected punishment episode shaded in red
                    (``axvspan`` from ``start_step`` to ``recovery_step``);
    * lower panel : rolling-mean output of every learning agent, with the
                    same red bands so the price drop and the output spike
                    can be visually correlated.

    The visual signature mirrors Figure 18 (Green-Porter regimes): a
    cooperative plateau, a sharp drop triggered by a defection, and a
    return towards the cooperative level after a punishment phase.
    """
    prices = np.asarray(outcome.price_history, dtype=float)
    rolling = _rolling_mean(prices, window=window)
    steps = np.arange(len(prices))

    fig, (ax_p, ax_q) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.12),
    )

    # --- Upper panel : price ---
    ax_p.fill_between(
        steps, nash_price, cartel_price,
        color=PALETTE["cooperative"], alpha=0.08, label="Collusion corridor",
    )
    ax_p.plot(steps, rolling, color="black", linewidth=1.4, label="Rolling-mean price")
    ax_p.axhline(nash_price, color=PALETTE["nash"], linestyle="--", linewidth=1.0,
                 label=f"Nash ({nash_price:.0f} $/bbl)")
    ax_p.axhline(cartel_price, color=PALETTE["cooperative"], linestyle="--",
                 linewidth=1.0, label=f"Cartel ({cartel_price:.0f} $/bbl)")

    label_added = False
    for ep in episodes:
        ax_p.axvspan(
            ep.start_step, ep.recovery_step,
            color=REGIME_BG["punishment"], alpha=0.18,
            label=("Punishment episode" if not label_added else None),
        )
        label_added = True
        ax_p.scatter([ep.trough_step], [ep.trough_price],
                     color=REGIME_BG["punishment"], s=22, zorder=5,
                     edgecolors="white", linewidths=0.6)

    ax_p.set_ylabel("Market price ($/bbl)")
    ax_p.set_title("Emergent Punishment Detection in Multi-Agent RL",
                   fontweight="bold")
    ax_p.legend(loc="lower right", fontsize=8, ncol=2)

    n_ep = len(episodes)
    if n_ep > 0:
        mean_dur = float(np.mean([ep.recovery_step - ep.start_step
                                   for ep in episodes]))
        mean_drop = float(np.mean([ep.price_drop for ep in episodes]))
        annot = (f"Episodes detected: {n_ep}\n"
                 f"Mean cycle length: {mean_dur:.0f} steps\n"
                 f"Mean drop: {mean_drop:.2f} $/bbl")
    else:
        annot = ("Episodes detected: 0\n"
                 "(no clear cooperation–punishment cycles)")
    ax_p.text(
        0.012, 0.97, annot,
        transform=ax_p.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="grey", alpha=0.92),
    )

    # --- Lower panel : learning-agent outputs ---
    for p in outcome.learning_players:
        series = np.asarray([qd[p] for qd in outcome.q_history], dtype=float)
        rolling_q = _rolling_mean(series, window=window)
        ax_q.plot(steps, rolling_q, color=PALETTE.get(p, "grey"),
                  linewidth=1.6, label=f"q_{p}")

    label_added = False
    for ep in episodes:
        ax_q.axvspan(
            ep.start_step, ep.recovery_step,
            color=REGIME_BG["punishment"], alpha=0.18,
            label=("Punishment episode" if not label_added else None),
        )
        label_added = True

    ax_q.set_xlabel("Training step")
    ax_q.set_ylabel("Output (mbd)")
    ax_q.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_punishment_anatomy(
    outcome,
    episodes: List,
    path: str,
    half_width: int = 30,
    window: int = 50,
) -> None:
    """Average shape of a punishment episode, centred on the trough (t = 0).

    Stacks the smoothed price trajectory of every detected episode in a
    ``[trough - half_width, trough + half_width]`` window, plots each as
    a faint line and the cross-episode mean as a thick black line.

    If fewer than 3 episodes are available, draws an explanatory message
    instead of a meaningless plot.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    if len(episodes) < 3:
        ax.set_axis_off()
        ax.text(
            0.5, 0.5,
            f"Insufficient punishment episodes detected\n"
            f"(found {len(episodes)}, need ≥ 3 to compute an average shape)\n\n"
            f"Interpretation: vanilla Q-learning under price-only\n"
            f"information did NOT spontaneously coordinate on\n"
            f"grim-trigger style punishment in this run.",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5",
                      edgecolor="grey"),
        )
        ax.set_title("Anatomy of Emergent Punishment Episodes", fontweight="bold")
        fig.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    prices = np.asarray(outcome.price_history, dtype=float)
    rolling = _rolling_mean(prices, window=window)
    n_steps = len(rolling)

    rel = np.arange(-half_width, half_width + 1)
    stacked = []
    for ep in episodes:
        lo = ep.trough_step - half_width
        hi = ep.trough_step + half_width + 1
        if lo < 0 or hi > n_steps:
            continue
        stacked.append(rolling[lo:hi])
    if len(stacked) < 3:
        ax.set_axis_off()
        ax.text(
            0.5, 0.5,
            "Insufficient punishment episodes within the trajectory\n"
            "(too close to start/end to extract a ±30-step window).",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5",
                      edgecolor="grey"),
        )
        ax.set_title("Anatomy of Emergent Punishment Episodes", fontweight="bold")
        fig.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    arr = np.vstack(stacked)
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_curve)

    for row in arr:
        ax.plot(rel, row, color=PALETTE["OPEC"], alpha=0.25, linewidth=1.0)

    ax.fill_between(rel, mean_curve - std_curve, mean_curve + std_curve,
                    color=PALETTE["OPEC"], alpha=0.18, label="±1 std")
    ax.plot(rel, mean_curve, color="black", linewidth=2.0,
            label=f"Mean over n={arr.shape[0]} episodes")

    ax.axvline(0, color=REGIME_BG["punishment"], linestyle="--", linewidth=1.0,
               label="Trough (t = 0)")
    ax.axhline(float(np.mean([ep.pre_price for ep in episodes])),
               color=REGIME_BG["cooperation"], linestyle=":", linewidth=1.0,
               label="Mean pre-drop price")

    ax.set_xlabel("Steps relative to trough")
    ax.set_ylabel("Market price ($/bbl)")
    ax.set_title("Anatomy of Emergent Punishment Episodes", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Learner-count comparison plots (Section 8b — 1 vs 2 vs 3 learners)
# ---------------------------------------------------------------------------

_LEARNER_REGIMES = ("single", "duopoly", "triopoly")
_LEARNER_LABELS = {
    "single":   "1 learner\n(OPEC only)",
    "duopoly":  "2 learners\n(OPEC + US)",
    "triopoly": "3 learners\n(all players)",
}
_LEARNER_BAR_COLORS = {
    "single":   PALETTE["nash"],
    "duopoly":  PALETTE["OPEC"],
    "triopoly": PALETTE["cooperative"],
}


def _trend_label(regimes_means: List[float]) -> str:
    """Short qualitative description of the monotonicity of three values."""
    a, b, c = regimes_means
    if c < a and c < b:
        return "↘ collusion harder with 3 learners"
    if c > b and b > a:
        return "↗ more learners ⇒ stronger collusion"
    if c > b and c > a:
        return "↗ removing the free-rider amplifies collusion"
    if b > a and b > c:
        return "∩ duopoly is the sweet spot"
    if b < a and b < c:
        return "∪ duopoly is the worst case"
    return "≈ effect of learner count is weak"


def plot_learner_comparison_collusion(
    comparison_result: Dict,
    path: str,
) -> None:
    """Bar chart of the mean collusion index for each learner-count regime.

    For every regime in ``("single", "duopoly", "triopoly")`` the bar height
    is the mean collusion index over the seeds and the error bar is ±1 std.
    A dashed horizontal line at ``collusion_index = 0`` marks the pure Nash
    benchmark; the headline answers the question "does adding learners help
    or hurt tacit collusion?".
    """
    regimes = [r for r in _LEARNER_REGIMES if r in comparison_result]
    means = [comparison_result[r]["mean_collusion_index"] for r in regimes]
    stds = [comparison_result[r]["std_collusion_index"] for r in regimes]
    n_seeds = comparison_result[regimes[0]]["n_seeds"]
    n_learners = [comparison_result[r]["n_learners"] for r in regimes]
    labels = [_LEARNER_LABELS[r] for r in regimes]
    colors = [_LEARNER_BAR_COLORS[r] for r in regimes]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    x = np.arange(len(regimes))
    bars = ax.bar(
        x, means, yerr=stds, capsize=6,
        color=colors, edgecolor="black", linewidth=0.8, alpha=0.85,
        error_kw=dict(elinewidth=1.2, ecolor="black"),
    )

    for bar, m, s, n in zip(bars, means, stds, n_learners):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (s if s > 0 else 0.0) + 0.012,
            f"{m:.3f}\n±{s:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0,
               label="Nash (collusion = 0)")
    ax.axhline(1.0, color=PALETTE["cooperative"], linestyle=":", linewidth=1.0,
               label="Cartel (collusion = 1)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Collusion index  (P − P_nash) / (P_cartel − P_nash)")
    ax.set_title(
        "Effect of Number of Learning Agents on Tacit Collusion",
        fontweight="bold",
    )
    ax.legend(loc="center left", fontsize=8)

    trend = _trend_label(means)
    ax.text(
        0.012, 0.97,
        f"n_seeds = {n_seeds} per regime\nTrend: {trend}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="grey", alpha=0.92),
    )

    y_top = max(means[i] + stds[i] for i in range(len(regimes))) + 0.20
    y_bot = min(0.0, min(means[i] - stds[i] for i in range(len(regimes))) - 0.05)
    ax.set_ylim(y_bot, max(0.6, y_top))

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learner_comparison_prices(
    comparison_result: Dict,
    nash_price: float,
    cartel_price: float,
    path: str,
) -> None:
    """Bar chart of the mean converged price for each learner-count regime.

    Same structure as :func:`plot_learner_comparison_collusion` but on the
    raw price scale ($/bbl), with the Nash and cartel benchmarks drawn as
    dashed reference lines.
    """
    regimes = [r for r in _LEARNER_REGIMES if r in comparison_result]
    means = [comparison_result[r]["mean_converged_price"] for r in regimes]
    stds = [comparison_result[r]["std_converged_price"] for r in regimes]
    n_seeds = comparison_result[regimes[0]]["n_seeds"]
    labels = [_LEARNER_LABELS[r] for r in regimes]
    colors = [_LEARNER_BAR_COLORS[r] for r in regimes]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    x = np.arange(len(regimes))
    bars = ax.bar(
        x, means, yerr=stds, capsize=6,
        color=colors, edgecolor="black", linewidth=0.8, alpha=0.85,
        error_kw=dict(elinewidth=1.2, ecolor="black"),
    )

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (s if s > 0 else 0.0) + 0.4,
            f"{m:.2f} $/bbl\n±{s:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(nash_price, color=PALETTE["nash"], linestyle="--", linewidth=1.1,
               label=f"Nash ({nash_price:.0f} $/bbl)")
    ax.axhline(cartel_price, color=PALETTE["cooperative"], linestyle="--",
               linewidth=1.1, label=f"Cartel ({cartel_price:.0f} $/bbl)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Converged market price ($/bbl)")
    ax.set_title(
        "Converged Market Price by Number of Learning Agents",
        fontweight="bold",
    )
    ax.legend(loc="center left", fontsize=8)

    trend = _trend_label(means)
    ax.text(
        0.012, 0.97,
        f"n_seeds = {n_seeds} per regime\nTrend: {trend}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="grey", alpha=0.92),
    )

    y_top = max(cartel_price + 6.0,
                max(means[i] + stds[i] for i in range(len(regimes))) + 5.0)
    y_bot = min(nash_price - 3.0,
                min(means[i] - stds[i] for i in range(len(regimes))) - 2.0)
    ax.set_ylim(y_bot, y_top)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Audit Part III stress-test plots (γ sweep, shocks, deviation, capacities,
# Stackelberg-MARL, policy heatmaps)
# ---------------------------------------------------------------------------

def plot_marl_gamma_sweep(rows: List[Dict[str, float]], path: str) -> None:
    """Mean collusion index (and 95% CI band) as a function of γ.

    The Folk-Theorem-style prediction is that algorithmic cooperation
    becomes feasible only for sufficiently patient agents.  This panel
    visualises where the empirical learning system crosses from the Nash
    basin (CI ≈ 0) towards the cartel basin (CI ≈ 1).
    """
    gammas = [r["gamma"] for r in rows]
    means = [r["mean_collusion_index"] for r in rows]
    lo = [r["ci_95_low"] for r in rows]
    hi = [r["ci_95_high"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.fill_between(gammas, lo, hi, color=PALETTE["OPEC"], alpha=0.18,
                    label="95% CI of mean")
    ax.plot(gammas, means, "o-", color=PALETTE["OPEC"], linewidth=2,
            markersize=8, label="Mean collusion index")
    ax.axhline(0.0, color=PALETTE["nash"], linestyle="--", linewidth=1.1,
               label="Nash (CI = 0)")
    ax.axhline(1.0, color=PALETTE["cooperative"], linestyle="--", linewidth=1.1,
               label="Cartel (CI = 1)")
    ax.set_xlabel(r"Discount factor $\gamma$")
    ax.set_ylabel("Mean collusion index (over seeds)")
    ax.set_title(
        "Multi-Agent RL: Sensitivity to Patience (γ Sweep)",
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marl_shock_response(
    outcome,
    shock_schedule: List,
    nash_price: float,
    cartel_price: float,
    path: str,
) -> None:
    """Plot the headline shock-experiment trajectory with shaded shock windows."""
    prices = np.asarray(outcome.price_history, dtype=float)
    rolling = _rolling_mean(prices, window=50)
    steps = np.arange(len(prices))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(steps, nash_price, cartel_price, color=PALETTE["cooperative"],
                    alpha=0.10, label="Collusion corridor (Nash → Cartel)")
    ax.plot(steps, rolling, color=PALETTE["OPEC"], linewidth=1.7,
            label="Rolling price (window=50)")
    ax.axhline(nash_price, color=PALETTE["nash"], linestyle="--",
               linewidth=1.1, label=f"Nash ({nash_price:.1f})")
    ax.axhline(cartel_price, color=PALETTE["cooperative"], linestyle="--",
               linewidth=1.1, label=f"Cartel ({cartel_price:.1f})")
    for i, (s_start, s_end, delta_a) in enumerate(shock_schedule):
        ax.axvspan(s_start, s_end, color="#8c1c1c", alpha=0.12,
                   label="Demand shock" if i == 0 else None)
    ax.set_xlabel("Global training step")
    ax.set_ylabel("Market price ($/bbl)")
    ax.set_title(
        "Multi-Agent RL Under Demand Shocks During Training",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marl_forced_deviation(
    outcome_or_outcomes,
    deviator: str,
    deviation_start: int,
    deviation_duration: int,
    nash_price: float,
    cartel_price: float,
    pre_mean: float,
    during_mean: float,
    post_mean: float,
    path: str,
    *,
    rolling_window: int = 20,
    zoom_window: int = 500,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
) -> None:
    """Forced-deviation response with a 95% percentile band, zoomed around
    the deviation window so the reaction of *every* player is legible.

    The figure has two stacked panels:
      * **top** — mean rolling market price ± 95% percentile band across seeds,
        with the forced-deviation window shaded.  The horizontal axis is
        clipped to ``[deviation_start − zoom_window, deviation_start +
        deviation_duration + zoom_window]`` so the visual scale matches the
        size of the perturbation; the full training trajectory is shown only
        when a single seed is provided.
      * **bottom** — mean rolling output per player ± 95% band, same zoom.
        Solid lines for the deviator; dashed for the other learners.  The
        plot intentionally exposes whether *non-deviator* learners react
        within the window (algorithmic punishment) or stay flat.
    """
    outcomes = (
        list(outcome_or_outcomes)
        if isinstance(outcome_or_outcomes, list) else [outcome_or_outcomes]
    )
    n_seeds = len(outcomes)

    prices = _stack_price_history(outcomes)
    rolled_p = np.stack(
        [_rolling_mean(prices[s], window=rolling_window) for s in range(n_seeds)],
        axis=0,
    )
    p_mean = rolled_p.mean(axis=0)
    if n_seeds > 1:
        p_lo = np.percentile(rolled_p, ci_low, axis=0)
        p_hi = np.percentile(rolled_p, ci_high, axis=0)
    else:
        p_lo = p_hi = p_mean

    n_steps = p_mean.shape[0]
    steps = np.arange(n_steps)
    if n_seeds > 1:
        x_lo = max(0, deviation_start - zoom_window)
        x_hi = min(n_steps - 1, deviation_start + deviation_duration + zoom_window)
    else:
        x_lo, x_hi = 0, n_steps - 1

    fig, (ax_price, ax_q) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)

    ax_price.fill_between(steps, nash_price, cartel_price,
                          color=PALETTE["cooperative"], alpha=0.10,
                          label="Collusion corridor")
    if n_seeds > 1:
        ax_price.fill_between(steps, p_lo, p_hi, color=PALETTE["OPEC"],
                              alpha=0.22,
                              label=f"95% band (n={n_seeds} seeds)")
    ax_price.plot(steps, p_mean, color=PALETTE["OPEC"], linewidth=1.6,
                  label=f"Mean rolling price (w={rolling_window})")
    ax_price.axhline(nash_price, color=PALETTE["nash"], linestyle="--",
                     linewidth=1.0, label=f"Nash ({nash_price:.1f})")
    ax_price.axhline(cartel_price, color=PALETTE["cooperative"], linestyle="--",
                     linewidth=1.0, label=f"Cartel ({cartel_price:.1f})")
    ax_price.axvspan(deviation_start, deviation_start + deviation_duration,
                     color="#8c1c1c", alpha=0.13,
                     label=f"Forced deviation by {deviator}")
    info = (
        f"Mean price (pre / during / post): "
        f"{pre_mean:.1f} / {during_mean:.1f} / {post_mean:.1f}\n"
        f"n seeds = {n_seeds}, deviation = {deviation_duration} steps"
    )
    ax_price.text(
        0.012, 0.97, info,
        transform=ax_price.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="grey", alpha=0.92),
    )
    ax_price.set_ylabel("Market price ($/bbl)")
    title = (
        f"Forced Deviation Response — Deviator = {deviator} (Mean ± 95% Band)"
        if n_seeds > 1 else
        f"Forced Deviation Response — Deviator = {deviator} (single seed)"
    )
    ax_price.set_title(title, fontweight="bold")
    ax_price.legend(loc="lower right", fontsize=8, ncol=2)

    learners = outcomes[0].learning_players
    all_pl = list(outcomes[0].q_history[0].keys())
    for p in all_pl:
        mat = _stack_q_history(outcomes, p)
        rolled = np.stack(
            [_rolling_mean(mat[s], window=rolling_window) for s in range(n_seeds)],
            axis=0,
        )
        q_mean = rolled.mean(axis=0)
        color = PALETTE.get(p, "grey")
        is_dev = (p == deviator)
        is_learner = p in learners
        ls = "-" if is_dev else ("--" if is_learner else ":")
        lbl = (
            f"q_{p}" + (" — deviator" if is_dev
                       else (" (learner)" if is_learner else " (myopic)"))
        )
        if n_seeds > 1:
            q_lo = np.percentile(rolled, ci_low, axis=0)
            q_hi = np.percentile(rolled, ci_high, axis=0)
            ax_q.fill_between(np.arange(q_mean.shape[0]), q_lo, q_hi,
                              color=color, alpha=0.13)
        ax_q.plot(q_mean, color=color, linewidth=1.6 if is_dev else 1.3,
                  linestyle=ls, label=lbl)
    ax_q.axvspan(deviation_start, deviation_start + deviation_duration,
                 color="#8c1c1c", alpha=0.13)
    ax_q.set_xlabel("Global training step")
    ax_q.set_ylabel("Output (mbd)")
    ax_q.legend(loc="upper right", fontsize=8, ncol=2)

    ax_price.set_xlim(x_lo, x_hi)
    ax_q.set_xlim(x_lo, x_hi)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marl_capacity_comparison(
    summary_uncon: Dict[str, float],
    summary_con: Dict[str, float],
    path: str,
) -> None:
    """Side-by-side bars: collusion index and converged price, with/without
    capacity constraints."""
    labels = ["Unconstrained", "Constrained"]
    ci_means = [summary_uncon["mean_collusion_index"], summary_con["mean_collusion_index"]]
    ci_stds = [summary_uncon["std_collusion_index"], summary_con["std_collusion_index"]]
    pr_means = [summary_uncon["mean_converged_price"], summary_con["mean_converged_price"]]
    pr_stds = [summary_uncon["std_converged_price"], summary_con["std_converged_price"]]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax1.bar(x, ci_means, yerr=ci_stds, color=[PALETTE["nash"], PALETTE["cooperative"]],
            edgecolor="black", linewidth=0.7, capsize=6)
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean collusion index")
    ax1.set_title("Collusion Index vs. Capacities", fontweight="bold")

    ax2.bar(x, pr_means, yerr=pr_stds, color=[PALETTE["nash"], PALETTE["cooperative"]],
            edgecolor="black", linewidth=0.7, capsize=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean converged price ($/bbl)")
    ax2.set_title("Converged Price vs. Capacities", fontweight="bold")

    fig.suptitle(
        "Multi-Agent RL: Effect of Capacity Constraints",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marl_stackelberg_comparison(
    rows: List[Dict[str, float]],
    path: str,
) -> None:
    """Two-panel comparison of static Stackelberg vs MARL single-learner.

    Left panel: leader **quantity** (commitment benchmark).
    Right panel: **clearing price** (market-outcome benchmark).

    Both benchmarks matter: a discounted leader in a repeated game can
    diverge from the static $q_L^\\ast$ while still producing a clearing
    price close to $P^\\ast$.  Showing both quantities side-by-side
    prevents the misreading that "the MARL leader fails to recover
    Stackelberg" when only the commitment quantity is off.
    """
    leaders = [r["leader"] for r in rows]
    static_q = [r["static_leader_q"] for r in rows]
    marl_q = [r["marl_leader_q_mean"] for r in rows]
    marl_q_std = [r["marl_leader_q_std"] for r in rows]
    static_p = [r.get("static_leader_price", float("nan")) for r in rows]
    marl_p = [r["marl_converged_price_mean"] for r in rows]
    marl_p_std = [r["marl_converged_price_std"] for r in rows]
    x = np.arange(len(leaders))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.bar(x - width / 2, static_q, width, color=PALETTE["stackelberg"],
           edgecolor="black", linewidth=0.7, label="Static $q_L^\\ast$ (§3)")
    ax.bar(x + width / 2, marl_q, width, color=PALETTE["OPEC"],
           edgecolor="black", linewidth=0.7,
           yerr=marl_q_std, capsize=5, label="MARL $\\bar q_L$ ± std")
    ax.set_xticks(x); ax.set_xticklabels(leaders)
    ax.set_xlabel("Leader (sole learner)")
    ax.set_ylabel("Leader output (mbd)")
    ax.set_title("Commitment benchmark — quantity", fontweight="bold")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.bar(x - width / 2, static_p, width, color=PALETTE["stackelberg"],
           edgecolor="black", linewidth=0.7, label="Static $P^\\ast$ (§3)")
    ax.bar(x + width / 2, marl_p, width, color=PALETTE["OPEC"],
           edgecolor="black", linewidth=0.7,
           yerr=marl_p_std, capsize=5, label="MARL $\\bar P$ ± std")
    ax.set_xticks(x); ax.set_xticklabels(leaders)
    ax.set_xlabel("Leader (sole learner)")
    ax.set_ylabel("Clearing price ($/bbl)")
    ax.set_title("Market-outcome benchmark — price", fontweight="bold")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Stackelberg Equilibrium vs. Decentralised MARL Leader",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marl_policy_heatmaps(
    matrices: Dict[str, np.ndarray],
    price_centres: np.ndarray,
    q_centres_per_player: Dict[str, np.ndarray],
    path: str,
) -> None:
    """One panel per learning agent: argmax-action quantity as a function of
    (observed price, own previous output).  Shows the *shape* of the
    learned reciprocity rule — typically: low past prices → reduce output."""
    players = list(matrices.keys())
    n = len(players)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.6), squeeze=False)
    for i, p in enumerate(players):
        ax = axes[0][i]
        mat = matrices[p]
        q_centres = q_centres_per_player[p]
        im = ax.imshow(
            mat[::-1, :],  # high price at top
            aspect="auto", cmap="viridis",
            extent=(q_centres.min(), q_centres.max(),
                    price_centres.min(), price_centres.max()),
            origin="lower",
        )
        ax.set_xlabel(f"Own previous output q_{p} (mbd)")
        ax.set_ylabel("Observed market price ($/bbl)")
        ax.set_title(f"Learned policy: argmax_a Q({p})", fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(f"Chosen q_{p} (mbd)")

    fig.suptitle(
        "Multi-Agent RL: Policy Heatmaps (argmax-action per state)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stackelberg plots
# ---------------------------------------------------------------------------

def plot_stackelberg_comparison(rows: List[Dict], players: List[str], path: str) -> None:
    """Bar chart: quantities and price across leader configurations."""
    labels = [r["leader"] for r in rows]
    x = np.arange(len(labels))
    width = 0.18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    offsets = np.linspace(-(len(players) - 1) / 2, (len(players) - 1) / 2, len(players)) * width
    for i, p in enumerate(players):
        vals = [r.get(f"q_{p}", 0.0) for r in rows]
        col = PALETTE.get(p, f"C{i}")
        ax1.bar(x + offsets[i], vals, width, label=p, color=col, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("Output (mbd)")
    ax1.set_title("Production by player under each leadership regime")
    ax1.legend(loc="upper right", fontsize=8)

    prices = [r["P"] for r in rows]
    welfares = [r["total_welfare"] for r in rows]
    color_price = "#f58518"
    color_welfare = "#4c78a8"
    bars = ax2.bar(x, prices, color=color_price, alpha=0.7, label="Price ($/bbl)")
    ax2b = ax2.twinx()
    ax2b.plot(x, welfares, "o--", color=color_welfare, label="Total welfare", linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("Price ($/bbl)", color=color_price)
    ax2b.set_ylabel("Total welfare", color=color_welfare)
    ax2.set_title("Price and welfare by leadership regime")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=8)

    fig.suptitle("Stackelberg vs Cournot Nash: market structure comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Market power plots
# ---------------------------------------------------------------------------

def plot_market_power(df: "pd.DataFrame", path: str) -> None:
    """Grouped bar charts for HHI and Lerner indices across market structures."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = df["model"].tolist()
    x = np.arange(len(models))

    # HHI
    ax = axes[0]
    bars = ax.bar(x, df["HHI"], color="#b279a2", alpha=0.8)
    ax.axhline(2500, color="red", linestyle="--", linewidth=1, label="HHI = 2500 (highly concentrated)")
    ax.axhline(1500, color="orange", linestyle="--", linewidth=1, label="HHI = 1500 (moderate)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("HHI")
    ax.set_title("Herfindahl-Hirschman Index")
    ax.legend(fontsize=6)

    # Lerner indices
    ax = axes[1]
    lerner_cols = [c for c in df.columns if c.startswith("lerner_")]
    player_names = [c.replace("lerner_", "") for c in lerner_cols]
    bar_width = 0.25
    offsets = np.linspace(-(len(lerner_cols) - 1) / 2, (len(lerner_cols) - 1) / 2, len(lerner_cols)) * bar_width
    for i, (col, pname) in enumerate(zip(lerner_cols, player_names)):
        if col in df.columns:
            ax.bar(x + offsets[i], df[col].fillna(0), bar_width,
                   label=pname, color=PALETTE.get(pname, f"C{i}"), alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Lerner Index")
    ax.set_title("Lerner Index by player")
    ax.legend(fontsize=8)

    # Market share
    ax = axes[2]
    share_cols = [c for c in df.columns if c.startswith("share_")]
    share_names = [c.replace("share_", "") for c in share_cols]
    bottom = np.zeros(len(models))
    colors = [PALETTE.get(n, f"C{i}") for i, n in enumerate(share_names)]
    for col, pname, col_color in zip(share_cols, share_names, colors):
        if col in df.columns:
            vals = df[col].fillna(0).values
            ax.bar(x, vals, bottom=bottom, label=pname, color=col_color, alpha=0.85)
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Market share (%)")
    ax.set_title("Market share by player (%)")
    ax.legend(fontsize=8)

    fig.suptitle("Market Power Analysis across Market Structures", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shapley value plots
# ---------------------------------------------------------------------------

def plot_shapley_values(shap_result, path: str) -> None:
    """Horizontal bar chart of Shapley values with characteristic function table."""
    from .coalition import ShapleyResult
    assert isinstance(shap_result, ShapleyResult)

    players = list(shap_result.values.keys())
    values = [shap_result.values[p] for p in players]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [PALETTE.get(p, "gray") for p in players]
    bars = ax1.barh(players, values, color=colors, alpha=0.85)
    ax1.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax1.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f}", va="center", fontsize=9)
    ax1.set_xlabel("Shapley value (profit units)")
    ax1.set_title("Shapley Value Allocation\n(fair marginal contribution)")
    stable_str = "Core stable: YES" if shap_result.core_stable else "Core stable: NO"
    ax1.set_title(ax1.get_title() + f"\n{stable_str}", fontsize=10)

    # Characteristic function bar chart
    char = shap_result.characteristic
    labels = []
    char_vals = []
    for k in sorted(char.keys(), key=lambda t: (len(t), t)):
        if not k:
            continue
        labels.append("+".join(k))
        char_vals.append(char[k])
    x2 = np.arange(len(labels))
    ax2.bar(x2, char_vals, color="#72b7b2", alpha=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Coalition value v(S)")
    ax2.set_title("Characteristic Function v(S)\nfor all coalitions")

    fig.suptitle("Cooperative Game Theory: Shapley Values & Coalition Stability",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Folk theorem plots
# ---------------------------------------------------------------------------

def plot_folk_theorem(folk_result, path: str) -> None:
    """Visual summary of Folk theorem critical discount factors."""
    from .coalition import FolkTheoremResult
    assert isinstance(folk_result, FolkTheoremResult)

    players = list(folk_result.delta_star.keys())
    delta_stars = [folk_result.delta_star[p] for p in players]
    delta_actual = folk_result.delta_actual

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [PALETTE.get(p, "gray") for p in players]
    bars = ax1.bar(players, delta_stars, color=colors, alpha=0.8)
    ax1.axhline(delta_actual, color="red", linestyle="--", linewidth=2,
                label=f"δ actual = {delta_actual:.2f}")
    for bar, val in zip(bars, delta_stars):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Critical discount factor δ*")
    ax1.set_title("Grim-Trigger IC Constraint\nδ* per player")
    ax1.legend(fontsize=9)
    sustainable_text = "✓ Cooperation SUSTAINABLE" if folk_result.cooperation_sustainable else "✗ Cooperation NOT SUSTAINABLE"
    ax1.text(0.5, 0.05, sustainable_text, ha="center", va="bottom",
             transform=ax1.transAxes, fontsize=10,
             color="green" if folk_result.cooperation_sustainable else "red",
             fontweight="bold")

    # Profit comparison bars
    profit_types = ["pi_deviation", "pi_cooperative", "pi_nash"]
    profit_data = {
        "Deviation": [folk_result.pi_deviation[p] for p in players],
        "Cooperative": [folk_result.pi_cooperative[p] for p in players],
        "Nash": [folk_result.pi_nash[p] for p in players],
    }
    x = np.arange(len(players))
    width = 0.25
    offsets = [-width, 0, width]
    profit_colors = ["#e45756", "#72b7b2", "#b279a2"]
    for i, (label, vals) in enumerate(profit_data.items()):
        ax2.bar(x + offsets[i], vals, width, label=label, color=profit_colors[i], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(players)
    ax2.set_ylabel("Per-period profit")
    ax2.set_title("Profit comparison:\nDeviation vs Cooperative vs Nash")
    ax2.legend(fontsize=8)

    fig.suptitle("Folk Theorem: Sustainability of Tacit Collusion in Oil Markets",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stochastic plots
# ---------------------------------------------------------------------------

def plot_stochastic_price_bands(stoch_outcome, path: str) -> None:
    """Fan chart of Monte Carlo price paths with confidence bands."""
    T = stoch_outcome.price_mean.shape[0]
    t = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Price fan chart
    ax1.fill_between(t, stoch_outcome.price_p10, stoch_outcome.price_p90,
                     alpha=0.15, color="#4c78a8", label="10-90 pct band")
    ax1.fill_between(t, stoch_outcome.price_p25, stoch_outcome.price_p75,
                     alpha=0.30, color="#4c78a8", label="25-75 pct band")
    ax1.plot(t, stoch_outcome.price_mean, color="#4c78a8", linewidth=2.5, label="Mean price")

    # Plot a few sample paths in grey
    n_sample = min(30, stoch_outcome.price_paths.shape[0])
    for i in range(n_sample):
        ax1.plot(t, stoch_outcome.price_paths[i], color="grey", alpha=0.08, linewidth=0.7)

    ax1.set_xlabel("Period")
    ax1.set_ylabel("Price ($/bbl)")
    ax1.set_title(f"Oil Price Distribution (σ={stoch_outcome.sigma}, ρ={stoch_outcome.rho})\n"
                  f"Monte Carlo, n={stoch_outcome.price_paths.shape[0]} paths")
    ax1.legend(fontsize=8)

    # Terminal price distribution histogram
    terminal_prices = stoch_outcome.price_paths[:, -1]
    ax2.hist(terminal_prices, bins=40, color="#4c78a8", alpha=0.8, edgecolor="white")
    ax2.axvline(terminal_prices.mean(), color="#f58518", linewidth=2,
                label=f"Mean = {terminal_prices.mean():.1f}")
    ax2.axvline(np.percentile(terminal_prices, 10), color="red", linewidth=1.5,
                linestyle="--", label=f"P10 = {np.percentile(terminal_prices, 10):.1f}")
    ax2.axvline(np.percentile(terminal_prices, 90), color="green", linewidth=1.5,
                linestyle="--", label=f"P90 = {np.percentile(terminal_prices, 90):.1f}")
    ax2.set_xlabel("Terminal price ($/bbl)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of terminal period price")
    ax2.legend(fontsize=8)

    fig.suptitle("Stochastic Demand Uncertainty in the Oil Market (AR-1 shocks)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# New plots: myopic convergence with Nash overlay
# ---------------------------------------------------------------------------

def plot_myopic_with_nash_convergence(
    myopic_outcome,
    nash_result,
    coop_result,
    path: str,
) -> None:
    """Myopic convergence with Transient / Near-Nash regime shading.

    Two panels (quantities left, price right).  Convergence zone boundary
    is detected automatically as the first period where |P_t − P_Nash| < 1 $/bbl.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    players = list(myopic_outcome.quantities[0].keys())
    T = len(myopic_outcome.prices)
    ts = list(range(T))

    fig, (ax_q, ax_p) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: per-player quantities ---
    for i, p in enumerate(players):
        col = PALETTE.get(p, f"C{i}")
        series = [myopic_outcome.quantities[t][p] for t in range(T)]
        ax_q.plot(ts, series, color=col, linewidth=2.2, label=p, zorder=5)
        ax_q.axhline(nash_result.quantities[p], color=col, linestyle="--",
                     linewidth=1.3, alpha=0.65, zorder=4)
        if hasattr(coop_result, "target_outputs"):
            ax_q.axhline(coop_result.target_outputs[p], color=col, linestyle=":",
                         linewidth=1.3, alpha=0.65, zorder=4)

    extra_q = [
        Line2D([0], [0], linestyle="--", color="grey", lw=1.3, label="Nash equilibrium"),
        Line2D([0], [0], linestyle=":",  color="grey", lw=1.3, label="Cartel quota"),
    ]
    handles_q, _ = ax_q.get_legend_handles_labels()
    ax_q.legend(handles=handles_q[:len(players)] + extra_q, fontsize=8)
    ax_q.set_xlabel("Period")
    ax_q.set_ylabel("Output (mbd)")
    ax_q.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Right: market price ---
    ax_p.plot(ts, myopic_outcome.prices, color=PALETTE["nash"],
              linewidth=2.5, label="Myopic price", zorder=5)
    ax_p.axhline(nash_result.price, color="red", linestyle="--", linewidth=1.5,
                 alpha=0.8, label=f"Nash P = {nash_result.price:.1f}", zorder=4)
    if hasattr(coop_result, "target_price"):
        ax_p.axhline(coop_result.target_price, color=PALETTE["cooperative"],
                     linestyle=":", linewidth=1.5, alpha=0.8,
                     label=f"Cartel P = {coop_result.target_price:.1f}", zorder=4)
    ax_p.set_xlabel("Period")
    ax_p.set_ylabel("Price ($/bbl)")
    ax_p.legend(fontsize=9)
    ax_p.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Convergence shading on both panels
    _draw_convergence_bands([ax_q, ax_p], myopic_outcome.prices, nash_result.price)

    # Regime legend
    regime_patches = [
        Patch(facecolor=REGIME_BG["transient"], alpha=0.20, label="Transient phase"),
        Patch(facecolor=REGIME_BG["steady"],    alpha=0.20, label="Near-Nash steady state"),
        Line2D([0], [0], linestyle=":", color=REGIME_BG["steady"],
               lw=1.6, label="Convergence point"),
    ]
    fig.legend(handles=regime_patches, loc="lower center", ncol=3,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.suptitle("Myopic Cournot Dynamics: Convergence to Nash Equilibrium",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# New plots: inertia (lambda) sensitivity
# ---------------------------------------------------------------------------

def plot_lambda_sensitivity(
    lambda_results: Dict[float, object],
    nash_result,
    path: str,
    n_players: int = 3,
) -> None:
    """Show how inertia λ affects convergence dynamics.

    Panel 1 — Eigenvalue |1−2λ| with monotone/oscillatory regime shading.
              For n players, the aggregate eigenvalue is μ = 1 − nλ/(n−1)
              but for the simple 2b denominator Cournot it simplifies to
              μ = 1 − 2λ when n = 3.  Convergence rate = |μ|.
    Panel 2 — Convergence speed (periods to ±2 % of Nash) with regime bands.
    Panel 3 — Price trajectories coloured by regime (blue=monotone, red=oscillatory).
    """
    lambdas = sorted(lambda_results.keys())
    nash_price = nash_result.price
    lam_star = 2.0 / (n_players + 1)

    eigenvalues = [1.0 - 2.0 * lam for lam in lambdas]
    abs_eigenvalues = [abs(ev) for ev in eigenvalues]

    conv_periods = []
    for lam in lambdas:
        prices = np.array(lambda_results[lam].prices)
        within = np.where(np.abs(prices - nash_price) / nash_price < 0.02)[0]
        conv_periods.append(int(within[0]) if len(within) > 0 else len(prices))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    mono_color = "#2166ac"
    osc_color = "#b2182b"
    crit_color = "#4daf4a"

    # ---- Panel 1: eigenvalue landscape ----
    ax = axes[0]
    ax.axvspan(-0.02, lam_star, alpha=0.08, color=mono_color)
    ax.axvspan(lam_star, 1.02, alpha=0.08, color=osc_color)

    ax.plot(lambdas, abs_eigenvalues, "o-", color="#333333", linewidth=2, markersize=7,
            zorder=3, label=r"|$\mu$| = |1 - 2$\lambda$|")
    ax.axvline(lam_star, color=crit_color, linewidth=2, linestyle="-",
               label=f"λ* = {lam_star:.2f} (instant convergence)")
    ax.axvline(0.2, color="red", linewidth=1.5, linestyle="--",
               label="λ = 0.2 (baseline)")

    ax.text(lam_star / 2, 0.95, "Monotone\n(slow approach)",
            ha="center", va="top", fontsize=9, color=mono_color, fontweight="bold")
    ax.text((1.0 + lam_star) / 2, 0.95, "Oscillatory\n(overshoot)",
            ha="center", va="top", fontsize=9, color=osc_color, fontweight="bold")

    ax.set_xlabel("Inertia λ", fontsize=10)
    ax.set_ylabel("Eigenvalue  |μ|", fontsize=10)
    ax.set_title(r"Convergence Rate = |1 $-$ 2$\lambda$|", fontweight="bold",
                  fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="center right")

    # ---- Panel 2: convergence periods with regime bands ----
    ax = axes[1]
    ax.axvspan(-0.02, lam_star, alpha=0.08, color=mono_color)
    ax.axvspan(lam_star, 1.02, alpha=0.08, color=osc_color)

    colors_pts = [mono_color if lam < lam_star - 0.01
                  else crit_color if abs(lam - lam_star) < 0.01
                  else osc_color for lam in lambdas]
    ax.plot(lambdas, conv_periods, "-", color="#999999", linewidth=1.5, zorder=1)
    for i, lam in enumerate(lambdas):
        ax.plot(lam, conv_periods[i], "o", color=colors_pts[i],
                markersize=9, zorder=3, markeredgecolor="white", markeredgewidth=1.2)
    ax.axvline(lam_star, color=crit_color, linewidth=2, linestyle="-")
    ax.axvline(0.2, color="red", linewidth=1.5, linestyle="--")

    ax.annotate(f"λ*={lam_star:.1f}\n0 periods",
                xy=(lam_star, 0), xytext=(lam_star + 0.12, 15),
                fontsize=8, fontweight="bold", color=crit_color,
                arrowprops=dict(arrowstyle="->", color=crit_color, lw=1.5))

    ax.set_xlabel("Inertia λ", fontsize=10)
    ax.set_ylabel("Periods to converge (±2 % of Nash)", fontsize=10)
    ax.set_title(r"Convergence Speed (symmetric by |1$-$2$\lambda$|)",
                  fontweight="bold", fontsize=10)
    ax.set_xlim(-0.02, 1.02)

    # ---- Panel 3: price trajectories coloured by regime ----
    ax = axes[2]
    ax.axhline(nash_price, color="red", linestyle="--", linewidth=1.5, alpha=0.6,
               label=f"Nash P = {nash_price:.0f}")

    for lam in lambdas:
        outcome = lambda_results[lam]
        is_baseline = abs(lam - 0.2) < 0.01
        if lam < lam_star - 0.01:
            c = mono_color
            style = "-"
        elif abs(lam - lam_star) < 0.01:
            c = crit_color
            style = "-"
        elif lam > 0.99:
            c = "#e7298a"
            style = "-"
        else:
            c = osc_color
            style = "-"
        lw = 2.8 if is_baseline else (2.0 if abs(lam - lam_star) < 0.01 else 1.1)
        alpha = 1.0 if is_baseline or abs(lam - lam_star) < 0.01 else 0.55
        ax.plot(outcome.prices, color=c, linewidth=lw, alpha=alpha,
                linestyle=style, label=f"λ={lam:.1f}")

    ax.set_xlabel("Period", fontsize=10)
    ax.set_ylabel("Price ($/bbl)", fontsize=10)
    ax.set_title("Price Paths (blue=monotone, red=oscillatory)", fontweight="bold",
                  fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    fig.suptitle(
        r"Inertia ($\lambda$) Sensitivity — Critical $\lambda^*$ = 2/(n+1)"
        f" = {lam_star:.2f} for n={n_players} players",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# New plots: RL hyperparameter sweeps
# ---------------------------------------------------------------------------

def plot_rl_hyperparameter_sweep(
    alpha_results: Dict[float, float],
    gamma_results: Dict[float, float],
    epsilon_results: Dict[float, float],
    baseline: Dict[str, float],
    path: str,
) -> None:
    """Three-panel plot: avg reward vs alpha, gamma, epsilon.

    Justifies the chosen hyperparameters by showing:
    - Alpha: optimal learning rate region
    - Gamma: value of patience (long vs short horizon)
    - Epsilon: exploration-exploitation trade-off

    The baseline values are highlighted with a vertical red dashed line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    configs = [
        (axes[0], alpha_results, "α (learning rate)",  baseline.get("alpha", 0.15),  "#4c78a8"),
        (axes[1], gamma_results, "γ (discount factor)", baseline.get("gamma", 0.95), "#f58518"),
        (axes[2], epsilon_results, "ε (exploration)",   baseline.get("epsilon", 0.1), "#54a24b"),
    ]

    for ax, results, xlabel, baseline_val, color in configs:
        xs = sorted(results.keys())
        ys = [results[x] for x in xs]
        ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=8)
        ax.axvline(baseline_val, color="red", linestyle="--", linewidth=1.8,
                   label=f"Baseline = {baseline_val}")
        best_x = xs[int(np.argmax(ys))]
        best_y = max(ys)
        ax.scatter([best_x], [best_y], s=120, color="gold", zorder=5,
                   edgecolors="black", linewidth=1, label=f"Best = {best_x}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Avg reward (last episodes)")
        ax.set_title(f"Sensitivity to {xlabel.split('(')[0].strip()}", fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Q-Learning Hyperparameter Sensitivity Analysis\n"
                 "(OPEC agent, avg reward over last training episodes)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# New plots: jump-diffusion stochastic
# ---------------------------------------------------------------------------

def plot_stochastic_comparison(
    stoch_ar1,
    stoch_jump,
    path: str,
) -> None:
    """Side-by-side comparison of AR(1) vs jump-diffusion price paths.

    Shows that adding Poisson supply shocks produces the spikes and crashes
    characteristic of real Brent crude oil price time series.
    """
    T = stoch_ar1.price_mean.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, stoch, title, color in [
        (axes[0], stoch_ar1,  "AR(1) demand shocks only",  "#4c78a8"),
        (axes[1], stoch_jump, "AR(1) + Poisson supply jumps", "#e45756"),
    ]:
        n_show = min(40, stoch.price_paths.shape[0])
        for i in range(n_show):
            ax.plot(t, stoch.price_paths[i], color="grey", alpha=0.07, linewidth=0.6)
        ax.fill_between(t, stoch.price_p10, stoch.price_p90,
                        alpha=0.15, color=color, label="P10-P90 band")
        ax.fill_between(t, stoch.price_p25, stoch.price_p75,
                        alpha=0.30, color=color, label="P25-P75 band")
        ax.plot(t, stoch.price_mean, color=color, linewidth=2.5, label="Mean")
        ax.set_xlabel("Period")
        ax.set_ylabel("Price ($/bbl)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Stochastic Oil Price: AR(1) vs Jump-Diffusion\n"
                 "(Jump-diffusion replicates real-world supply shocks)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# New plots: Evolutionary Game Dynamics
# ---------------------------------------------------------------------------

def plot_evo_phase_diagram_2strategy(evo_outcome, path: str) -> None:
    """Phase portrait for the 2-strategy (C vs D) evolutionary game.

    Panel 1 — 1-D phase portrait: ẋ_C as a function of x_C.
              Flow arrows show direction of evolution.
              Fixed points marked with circles (filled = stable, open = unstable).

    Panel 2 — Time series of strategy share x_C for multiple initial conditions.
              Shows whether cooperation can be sustained from various starting points.
    """
    from .evolutionary import EvoOutcome

    payoff = evo_outcome.payoff_2
    x_grid = evo_outcome.phase_x
    dx_grid = evo_outcome.phase_dx
    t = evo_outcome.t_grid

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel 1: Phase portrait ---
    ax = axes[0]
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(x_grid, dx_grid, color="#4c78a8", linewidth=2.5, label="ẋ_C (replicator RHS)")
    ax.fill_between(x_grid, dx_grid, 0, where=(dx_grid > 0),
                    alpha=0.15, color="#54a24b", label="C spreads (ẋ > 0)")
    ax.fill_between(x_grid, dx_grid, 0, where=(dx_grid < 0),
                    alpha=0.15, color="#e45756", label="D spreads (ẋ < 0)")

    # Mark fixed points
    x_star = evo_outcome.interior_eq_2
    for x_fp, label, marker, color_fp in [
        (0.0, "x=0: All Defect",   "s", "#e45756"),
        (1.0, "x=1: All Cooperate","D", "#54a24b"),
    ]:
        dx_fp = float(np.interp(x_fp, x_grid, dx_grid))
        ax.scatter([x_fp], [dx_fp], s=100, color=color_fp, zorder=5, marker=marker, label=label)

    if x_star is not None:
        dx_xs = float(np.interp(x_star, x_grid, dx_grid))
        ax.scatter([x_star], [dx_xs], s=120, color="black", zorder=6,
                   marker="^", label=f"Interior x*={x_star:.3f}")

    # Add ESS annotations
    for i, lbl in enumerate(evo_outcome.ess_labels_2):
        ax.annotate(lbl, xy=(0.02, 0.97 - i*0.08), xycoords="axes fraction",
                    fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("x_C (fraction cooperating)", fontsize=11)
    ax.set_ylabel("ẋ_C (rate of change)", fontsize=11)
    ax.set_title("1-D Phase Portrait: Replicator Dynamics", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    # --- Panel 2: Time series ---
    ax = axes[1]
    n_traj = len(evo_outcome.trajectories_2)
    cmap = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_traj))
    for i, traj in enumerate(evo_outcome.trajectories_2):
        ax.plot(t, traj[:, 0], color=cmap[i], alpha=0.8, linewidth=1.5)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="x=0.5")
    if x_star is not None:
        ax.axhline(x_star, color="black", linestyle="--", linewidth=1.5,
                   label=f"Interior x*={x_star:.3f}")
    ax.axhline(1.0, color="#54a24b", linestyle="--", linewidth=1.3, label="All Cooperate")
    ax.axhline(0.0, color="#e45756", linestyle="--", linewidth=1.3, label="All Defect")
    ax.set_xlabel("Evolutionary time", fontsize=11)
    ax.set_ylabel("x_C (fraction cooperating)", fontsize=11)
    ax.set_title("Trajectories from Multiple Initial Conditions\n(color: low x_C → red, high x_C → green)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"Evolutionary Game Dynamics: {evo_outcome.focal_player} — Cooperate vs Defect\n"
        f"Payoffs: π_dev={payoff.profit_labels.get('pi_dev','?')}, "
        f"π_coop={payoff.profit_labels.get('pi_coop','?')}, "
        f"π_nash={payoff.profit_labels.get('pi_nash','?')}, "
        f"π_sucker={payoff.profit_labels.get('pi_sucker','?')}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _to_bary(points: np.ndarray) -> np.ndarray:
    """Convert simplex coordinates (n, 3) to 2-D Cartesian for triangular plot.

    Equilateral triangle vertices: C=(0,0), D=(1,0), P=(0.5, √3/2).
    """
    C_xy = np.array([0.0, 0.0])
    D_xy = np.array([1.0, 0.0])
    P_xy = np.array([0.5, np.sqrt(3) / 2])
    return (points[:, 0:1] * C_xy + points[:, 1:2] * D_xy + points[:, 2:3] * P_xy)


def plot_evo_phase_diagram_3strategy(evo_outcome, path: str) -> None:
    """Triangular phase diagram for the 3-strategy (C, D, P) evolutionary game.

    Uses barycentric (ternary) coordinates on an equilateral triangle.
    Trajectories show the evolutionary paths from diverse starting points.
    Vector field shows the local direction of evolution at grid points.
    """
    from .evolutionary import EvoOutcome, _replicator_rhs

    payoff = evo_outcome.payoff_3
    T_pts = len(evo_outcome.t_grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Triangle vertices in Cartesian
    C_xy = np.array([0.0, 0.0])
    D_xy = np.array([1.0, 0.0])
    P_xy = np.array([0.5, np.sqrt(3) / 2])
    vertices = np.array([C_xy, D_xy, P_xy])
    labels_v = ["C\n(Cooperate)", "D\n(Defect)", "P\n(Punish)"]
    label_offsets = [np.array([-0.12, -0.07]), np.array([0.02, -0.07]), np.array([-0.05, 0.04])]

    for ax_idx, (ax, show_vector) in enumerate(zip(axes, [True, False])):
        # Draw triangle border
        tri_patch = plt.Polygon(vertices, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(tri_patch)

        # Label corners
        for xy, lbl, off in zip(vertices, labels_v, label_offsets):
            ax.annotate(lbl, xy=xy + off, fontsize=11, ha="center", fontweight="bold")

        if show_vector:
            # Vector field: sample interior grid points
            n_grid = 12
            arrow_pts = []
            for i in range(1, n_grid):
                for j in range(1, n_grid - i):
                    k = n_grid - i - j
                    x0 = np.array([i, j, k], dtype=float) / n_grid
                    arrow_pts.append(x0)
            arrow_pts_arr = np.array(arrow_pts)
            xy_pts = _to_bary(arrow_pts_arr)

            for idx, (x0, xy_pt) in enumerate(zip(arrow_pts, xy_pts)):
                dx = _replicator_rhs(0.0, x0, payoff.A)
                # Convert dx to barycentric direction
                xy_dx = _to_bary((x0 + dx * 0.05)[np.newaxis, :])[0] - xy_pt
                norm = np.linalg.norm(xy_dx)
                if norm > 1e-6:
                    ax.annotate("", xy=xy_pt + xy_dx / norm * 0.04, xytext=xy_pt,
                                arrowprops=dict(arrowstyle="->", color="grey",
                                                lw=0.8, mutation_scale=8))
            ax.set_title("Phase Diagram with Vector Field", fontweight="bold")
        else:
            ax.set_title("Trajectories from Multiple Initial Conditions", fontweight="bold")

        # Plot trajectories
        n_traj = len(evo_outcome.trajectories_3)
        cmap_t = plt.cm.plasma(np.linspace(0.1, 0.9, n_traj))
        for i, traj in enumerate(evo_outcome.trajectories_3):
            xy_traj = _to_bary(traj)
            ax.plot(xy_traj[:, 0], xy_traj[:, 1], color=cmap_t[i], alpha=0.55, linewidth=1.2)
            # Start marker
            ax.scatter(xy_traj[0, 0], xy_traj[0, 1], s=25, color=cmap_t[i], zorder=4, alpha=0.7)
            # End marker with arrow
            ax.annotate("", xy=xy_traj[-1], xytext=xy_traj[-2],
                        arrowprops=dict(arrowstyle="->", color=cmap_t[i],
                                        lw=1.2, mutation_scale=10))

        # Mark corner equilibria (potential ESS)
        for xy, col in zip(vertices, ["#54a24b", "#e45756", "#b279a2"]):
            ax.scatter(*xy, s=80, color=col, zorder=6, edgecolors="black", linewidth=1.2)

        ax.set_xlim(-0.18, 1.15)
        ax.set_ylim(-0.12, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        f"Evolutionary Dynamics (3 Strategies): {evo_outcome.focal_player}\n"
        f"Payoffs (C,C)={payoff.profit_labels.get('pi_coop','?')}  "
        f"(D,C)={payoff.profit_labels.get('pi_dev','?')}  "
        f"(D,D)={payoff.profit_labels.get('pi_nash','?')}  "
        f"(P,D)={payoff.profit_labels.get('pi_P_vs_D','?')}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_evo_payoff_matrix(evo_outcome, path: str) -> None:
    """Heatmap of the 2x2 and 3x3 payoff matrices.

    Allows the reader to immediately see the Prisoner's Dilemma structure
    and how the punishment strategy reshapes the incentives.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, payoff, title in [
        (axes[0], evo_outcome.payoff_2, "2-Strategy Game (C vs D)"),
        (axes[1], evo_outcome.payoff_3, "3-Strategy Game (C, D, P)"),
    ]:
        n = len(payoff.strategies)
        im = ax.imshow(payoff.A, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(payoff.strategies, fontsize=10)
        ax.set_yticklabels(payoff.strategies, fontsize=10)
        ax.set_xlabel("Opponent strategy", fontsize=10)
        ax.set_ylabel("Focal strategy", fontsize=10)
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Profit (model units)")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{payoff.A[i, j]:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if payoff.A[i, j] < payoff.A.mean() else "black")

    fig.suptitle(
        f"Evolutionary Game Payoff Matrices — {evo_outcome.focal_player}\n"
        "Row = focal player strategy; Column = opponent strategy",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_evo_punishment_sweep(sweep_result, t_grid: np.ndarray, path: str) -> None:
    """Punishment multiplier sensitivity: how severity reshapes evolutionary dynamics.

    Panel 1 — Stacked area: final strategy shares vs punishment multiplier.
    Panel 2 — Heatmap: 3x3 payoff entries as a function of the multiplier.
    Panel 3 — Mean population payoff trajectory for selected multipliers.
    """
    from .evolutionary import PunishmentSweepResult, STRATEGIES_3

    mults = sweep_result.multipliers
    conv = np.array(sweep_result.convergence)  # (n_mult, 3)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # --- Panel 1: Stacked area of final strategy shares ---
    ax = axes[0]
    colors = ["#54a24b", "#e45756", "#b279a2"]
    labels = ["Cooperate (C)", "Defect (D)", "Punish (P)"]
    ax.stackplot(mults, conv[:, 0], conv[:, 1], conv[:, 2],
                 labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("Punishment multiplier (× Nash output)", fontsize=10)
    ax.set_ylabel("Final strategy share", fontsize=10)
    ax.set_title("Equilibrium Strategy Mix\nvs Punishment Severity", fontweight="bold")
    ax.legend(fontsize=8, loc="center right")
    ax.set_xlim(mults[0], mults[-1])
    ax.set_ylim(0, 1)

    for i, m in enumerate(mults):
        dom = sweep_result.dominant_strategy[i]
        if i == 0 or dom != sweep_result.dominant_strategy[i - 1]:
            ax.axvline(m, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    # --- Panel 2: Selected payoff entries across multipliers ---
    ax = axes[1]
    entry_labels = [
        ("P vs C", 2, 0), ("P vs D", 2, 1), ("P vs P", 2, 2),
        ("C vs P", 0, 2), ("D vs P", 1, 2),
    ]
    entry_colors = ["#b279a2", "#9467bd", "#8c564b", "#54a24b", "#e45756"]
    for (lbl, r, c), col in zip(entry_labels, entry_colors):
        vals = [float(sweep_result.payoff_matrices[k][r, c]) for k in range(len(mults))]
        ax.plot(mults, vals, "o-", label=lbl, color=col, linewidth=2, markersize=5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Punishment multiplier (× Nash output)", fontsize=10)
    ax.set_ylabel("Payoff (profit units)", fontsize=10)
    ax.set_title("Punishment-Related Payoffs\nvs Multiplier", fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2)

    # --- Panel 3: Mean population payoff over time for selected multipliers ---
    ax = axes[2]
    sel_indices = [0, len(mults) // 3, 2 * len(mults) // 3, len(mults) - 1]
    sel_indices = sorted(set(min(idx, len(mults) - 1) for idx in sel_indices))
    cmap_sel = plt.cm.viridis(np.linspace(0.15, 0.85, len(sel_indices)))

    for color_i, idx in enumerate(sel_indices):
        mp = sweep_result.mean_payoff_trajectory[idx]
        t_plot = t_grid[:len(mp)]
        ax.plot(t_plot, mp, color=cmap_sel[color_i], linewidth=2,
                label=f"m={mults[idx]:.2f} → {sweep_result.dominant_strategy[idx]}")

    ax.set_xlabel("Evolutionary time", fontsize=10)
    ax.set_ylabel("Mean population payoff", fontsize=10)
    ax.set_title("Population Fitness Over Time\n(selected multipliers)", fontweight="bold")
    ax.legend(fontsize=8)

    model_tag = "Conditional (TFT-like)" if sweep_result.conditional else "Unconditional"
    fig.suptitle(
        f"Punishment Severity Sensitivity ({model_tag}):\n"
        "How Aggressive Must OPEC's Retaliation Be?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_demand_shock_scenarios(df: "pd.DataFrame", players: List[str], path: str) -> None:
    """Show price and profit responses across a range of demand shocks."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    shock_sizes = df["shock_size"].values
    prices_after = df["price_after"].values
    prices_before = df["price_before"].values[0]

    ax = axes[0]
    ax.axhline(prices_before, color="grey", linestyle="--", linewidth=1.2, label="Baseline price")
    ax.plot(shock_sizes, prices_after, "o-", color="#4c78a8", linewidth=2, markersize=7, label="Post-shock price")
    ax.fill_between(shock_sizes,
                    [p * 0.95 for p in prices_after],
                    [p * 1.05 for p in prices_after],
                    alpha=0.15, color="#4c78a8")
    ax.set_xlabel("Demand shock Δa ($/bbl)")
    ax.set_ylabel("Equilibrium price ($/bbl)")
    ax.set_title("Price response to demand shocks")
    ax.legend(fontsize=8)

    ax = axes[1]
    pct_changes = df["price_change_pct"].values
    colors_bar = ["#e45756" if v < 0 else "#54a24b" for v in pct_changes]
    ax.bar(range(len(pct_changes)), pct_changes, color=colors_bar, alpha=0.8)
    ax.set_xticks(range(len(shock_sizes)))
    ax.set_xticklabels([f"{s:+.0f}" for s in shock_sizes], fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Demand shock Δa")
    ax.set_ylabel("Price change (%)")
    ax.set_title("Price elasticity to demand shocks")

    ax = axes[2]
    colors_p = [PALETTE.get(p, f"C{i}") for i, p in enumerate(players)]
    for i, p in enumerate(players):
        col = f"profit_after_{p}"
        if col in df.columns:
            ax.plot(shock_sizes, df[col].values, "o-",
                    color=colors_p[i], label=p, linewidth=2, markersize=6)
    ax.set_xlabel("Demand shock Δa")
    ax.set_ylabel("Profit ($/day units)")
    ax.set_title("Profit by player under demand shocks")
    ax.legend(fontsize=8)

    fig.suptitle("Demand Shock Scenario Analysis: Oil Market Resilience",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Green-Porter plots
# ---------------------------------------------------------------------------

_GP_COOP = "#2d6a4f"
_GP_PUNISH = "#ae2012"
_GP_TRIGGER = "#9b2226"


def plot_green_porter_regimes(
    gp_outcome,
    path: str,
    representative_path: int = 0,
) -> None:
    """Visualise a single Green-Porter MC path with regime shading.

    Panel 1 — Price trajectory with trigger line and regime backgrounds.
    Panel 2 — Zoom on a false-trigger episode.
    Panel 3 — Per-player quantities switching between cartel and Nash.
    """
    # Auto-select a path with at least one punishment trigger for illustration
    idx = representative_path
    for i in range(gp_outcome.regime_paths.shape[0]):
        if np.any(gp_outcome.regime_paths[i] == 1):
            idx = i
            break
    prices = gp_outcome.price_paths[idx]
    regimes = gp_outcome.regime_paths[idx]
    T = len(prices)
    p_bar = gp_outcome.trigger_price
    p_nash = gp_outcome.nash_price
    p_coop = gp_outcome.coop_price
    players = gp_outcome.players

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    # --- Panel 1: full path ---
    ax = axes[0]
    _shade_gp_regimes(ax, regimes, T)
    ax.plot(prices, color="#333333", linewidth=1.2, label="Observed price")
    ax.axhline(p_coop, color=_GP_COOP, linestyle="--", linewidth=1.2,
               label=f"Cartel price = {p_coop:.1f}")
    ax.axhline(p_nash, color="#4c78a8", linestyle="--", linewidth=1.2,
               label=f"Nash price = {p_nash:.1f}")
    ax.axhline(p_bar, color=_GP_TRIGGER, linestyle=":", linewidth=1.8,
               label=f"Trigger $\\bar{{p}}$ = {p_bar:.1f}")
    ax.set_ylabel("Price ($/bbl)")
    ax.set_title("Green-Porter: price path with trigger-price punishment",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    # --- Panel 2: zoom on first false trigger ---
    ax2 = axes[1]
    trigger_periods = np.where(
        (regimes[:-1] == 0) & (regimes[1:] == 1)
    )[0]
    if len(trigger_periods) > 0:
        tp = trigger_periods[0]
        lo = max(0, tp - 5)
        hi = min(T, tp + 25)
        zoom_prices = prices[lo:hi]
        zoom_regimes = regimes[lo:hi]
        zoom_t = np.arange(lo, hi)
        _shade_gp_regimes(ax2, zoom_regimes, len(zoom_prices), t_offset=lo)
        ax2.plot(zoom_t, zoom_prices, "o-", color="#333333", linewidth=1.5,
                 markersize=4)
        ax2.axhline(p_bar, color=_GP_TRIGGER, linestyle=":", linewidth=1.8)
        ax2.axhline(p_coop, color=_GP_COOP, linestyle="--", linewidth=1)
        ax2.axhline(p_nash, color="#4c78a8", linestyle="--", linewidth=1)
        ax2.axvline(tp, color="black", linestyle="-", linewidth=1.5, alpha=0.5)
        ax2.annotate("Trigger fired\n(demand shock, not deviation)",
                     xy=(tp, prices[tp]), xytext=(tp + 3, prices[tp] + 5),
                     fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", lw=1.5))
        ax2.set_ylabel("Price ($/bbl)")
        ax2.set_title(f"Zoom: false trigger at t={tp} (shock-induced price war)",
                      fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No trigger in this path", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=12)
        ax2.set_title("Zoom (no trigger)", fontweight="bold")

    # --- Panel 3: per-player quantities ---
    ax3 = axes[2]
    _shade_gp_regimes(ax3, regimes, T)
    for i, p in enumerate(players):
        q = gp_outcome.player_quantity_paths[p][idx]
        ax3.plot(q, color=PLAYER_COLORS[i % len(PLAYER_COLORS)],
                 linewidth=1.3, label=p)
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Quantity (mbd)")
    ax3.set_title("Per-player output: cartel quotas vs Nash quantities",
                  fontweight="bold")
    ax3.legend(fontsize=9)

    fig.suptitle("Green-Porter (1984): Repeated Game Under Imperfect Monitoring",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _shade_gp_regimes(ax, regimes, length, t_offset=0):
    """Shade background green (cooperation) / red (punishment)."""
    t = 0
    while t < length:
        r = regimes[t]
        t_start = t
        while t < length and regimes[t] == r:
            t += 1
        color = _GP_COOP if r == 0 else _GP_PUNISH
        ax.axvspan(t_start + t_offset, t + t_offset, alpha=0.10, color=color)


def plot_cooperation_survival(
    gp_ar1,
    gp_jump,
    path: str,
) -> None:
    """Fraction of MC paths in cooperation over time (AR1 vs jump-diffusion)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    T = len(gp_ar1.coop_fraction)
    ax.plot(gp_ar1.coop_fraction, color="#2166ac", linewidth=2,
            label="AR(1) only")
    if gp_jump is not None:
        ax.plot(gp_jump.coop_fraction, color="#b2182b", linewidth=2,
                label="AR(1) + Jumps")
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.4)
    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel("Fraction of MC paths in cooperation", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Cooperation Survival Under Stochastic Demand "
                 "(Green-Porter trigger-price mechanism)",
                 fontweight="bold")

    ar1_ss = float(gp_ar1.coop_fraction[-20:].mean())
    ax.axhline(ar1_ss, color="#2166ac", linestyle="--", alpha=0.5)
    ax.text(T * 0.75, ar1_ss + 0.03, f"AR(1) steady-state: {ar1_ss:.0%}",
            fontsize=9, color="#2166ac")
    if gp_jump is not None:
        jump_ss = float(gp_jump.coop_fraction[-20:].mean())
        ax.axhline(jump_ss, color="#b2182b", linestyle="--", alpha=0.5)
        ax.text(T * 0.75, jump_ss - 0.05, f"Jump steady-state: {jump_ss:.0%}",
                fontsize=9, color="#b2182b")

    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delta_star_comparison(
    det_result,
    stoch_ar1_result,
    stoch_jump_result,
    delta_actual: float,
    path: str,
) -> None:
    """Bar chart comparing deterministic vs stochastic δ*."""
    fig, ax = plt.subplots(figsize=(9, 5))

    players = list(det_result.delta_star.keys())
    x = np.arange(len(players))
    w = 0.22

    det_vals = [det_result.delta_star[p] for p in players]
    ar1_val = stoch_ar1_result.delta_star_stochastic
    jump_val = stoch_jump_result.delta_star_stochastic if stoch_jump_result else ar1_val

    ax.bar(x - w, det_vals, w, label="Deterministic", color="#4c78a8", zorder=3)
    ax.bar(x, [ar1_val] * len(players), w, label="Stochastic (AR1)",
           color="#f58518", zorder=3)
    ax.bar(x + w, [jump_val] * len(players), w, label="Stochastic (AR1+Jumps)",
           color="#e45756", zorder=3)

    ax.axhline(delta_actual, color="black", linestyle="--", linewidth=2,
               label=f"$\\delta_{{actual}}$ = {delta_actual}", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(players)
    ax.set_ylabel(r"Critical discount factor $\delta^*$", fontsize=11)
    ax.set_title(r"Folk Theorem $\delta^*$: Deterministic vs Stochastic "
                 "(Green-Porter)", fontweight="bold")
    ax.set_ylim(0, min(1.05, delta_actual + 0.15))
    ax.legend(fontsize=9)

    alpha_ar1 = stoch_ar1_result.alpha
    alpha_j = stoch_jump_result.alpha if stoch_jump_result else alpha_ar1
    ax.text(0.02, 0.97,
            f"False-trigger prob. $\\alpha$: AR1={alpha_ar1:.1%}, "
            f"Jump={alpha_j:.1%}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence_after_shock(
    conv_data,
    path: str,
) -> None:
    """Mean conditional price path after a large negative shock."""
    if conv_data is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    h = conv_data.horizon
    t = np.arange(h)

    ax.fill_between(t,
                    conv_data.mean_price_path - conv_data.std_price_path,
                    conv_data.mean_price_path + conv_data.std_price_path,
                    alpha=0.2, color="#4c78a8")
    ax.plot(t, conv_data.mean_price_path, "o-", color="#4c78a8", linewidth=2,
            markersize=4, label="Mean price after shock")
    ax.axhline(conv_data.coop_price, color=_GP_COOP, linestyle="--",
               linewidth=1.5, label=f"Cartel price = {conv_data.coop_price:.1f}")
    ax.axhline(conv_data.nash_price, color="#e45756", linestyle="--",
               linewidth=1.5, label=f"Nash price = {conv_data.nash_price:.1f}")

    hl = conv_data.half_life_periods
    if hl < h:
        ax.axvline(hl, color="grey", linestyle=":", linewidth=1.5)
        ax.text(hl + 0.3, conv_data.mean_price_path[0] * 0.95,
                f"Half-life = {hl:.1f} periods", fontsize=10, fontweight="bold")

    ax.set_xlabel("Periods after shock", fontsize=11)
    ax.set_ylabel("Price ($/bbl)", fontsize=11)
    ax.set_title(f"Convergence After Large Shock "
                 f"(n={conv_data.n_events} events, threshold = "
                 f"{conv_data.coop_price - conv_data.mean_price_path[0]:.0f} $/bbl)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# Section A — Bertrand competition plots
# ===========================================================================

def plot_bertrand_nash_equilibrium(result, path: str) -> None:
    """Three-panel bar chart: Bertrand-Nash prices, quantities and profits per player."""
    players = list(result.prices.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [
        ("Equilibrium price ($/bbl)", [result.prices[p] for p in players]),
        ("Equilibrium quantity",       [result.quantities[p] for p in players]),
        ("Equilibrium profit",         [result.profits[p] for p in players]),
    ]
    for ax, (title, vals) in zip(axes, panels):
        colors = [PALETTE.get(p, "grey") for p in players]
        bars = ax.bar(players, vals, color=colors, edgecolor="black", linewidth=0.7)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:,.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Bertrand-Nash equilibrium  (sigma = {result.sigma:.2f})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bertrand_vs_cournot(comparison, players, path: str) -> None:
    """Cournot-Nash vs Bertrand-Nash on price, total Q, profits and consumer surplus."""
    cournot = comparison.cournot_nash
    bertrand = comparison.bertrand_nash
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.bar(["Cournot", "Bertrand"], [cournot["price"], bertrand.average_price],
           color=[PALETTE["nash"], PALETTE["stackelberg"]],
           edgecolor="black", linewidth=0.7)
    ax.set_title("Equilibrium price")
    ax.set_ylabel("$/bbl")
    for i, v in enumerate([cournot["price"], bertrand.average_price]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(["Cournot", "Bertrand"], [cournot["total_q"], bertrand.total_quantity],
           color=[PALETTE["nash"], PALETTE["stackelberg"]],
           edgecolor="black", linewidth=0.7)
    ax.set_title("Total quantity")
    ax.set_ylabel("mbd")
    for i, v in enumerate([cournot["total_q"], bertrand.total_quantity]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    x = np.arange(len(players))
    w = 0.38
    cournot_profits = [cournot[f"profit_{p}"] for p in players]
    bertrand_profits = [bertrand.profits[p] for p in players]
    ax.bar(x - w / 2, cournot_profits, w, color=PALETTE["nash"],
           edgecolor="black", linewidth=0.6, label="Cournot")
    ax.bar(x + w / 2, bertrand_profits, w, color=PALETTE["stackelberg"],
           edgecolor="black", linewidth=0.6, label="Bertrand")
    ax.set_xticks(x)
    ax.set_xticklabels(players)
    ax.set_title("Per-player profit")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(["Cournot", "Bertrand"],
           [cournot["consumer_surplus"], bertrand.consumer_surplus],
           color=[PALETTE["nash"], PALETTE["stackelberg"]],
           edgecolor="black", linewidth=0.7)
    ax.set_title("Consumer surplus")
    ax.set_ylabel("welfare units")
    for i, v in enumerate([cournot["consumer_surplus"], bertrand.consumer_surplus]):
        ax.text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Cournot vs Bertrand at sigma = {bertrand.sigma:.2f}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bertrand_sigma_sweep(sweep, players, path: str) -> None:
    """3-panel: Bertrand-Nash price, profit and quantity vs sigma per player."""
    sigmas = sweep.sigma_values
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, ylabel in zip(
        axes, ["prices", "profits", "quantities"],
        ["Price ($/bbl)", "Profit", "Quantity"],
    ):
        for p in players:
            series = [getattr(r, key)[p] for r in sweep.nash_results]
            ax.plot(sigmas, series, marker="o", color=PALETTE.get(p, "grey"),
                    linewidth=1.6, label=p)
        ax.set_xlabel("sigma  (substitutability)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("Equilibrium prices")
    axes[1].set_title("Equilibrium profits")
    axes[2].set_title("Equilibrium quantities")
    fig.suptitle(
        "Bertrand-Nash vs sigma  (sigma=0 -> monopolies, sigma->1 -> MC pricing)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bertrand_cooperation_gap(sweep, players, path: str) -> None:
    """Nash vs cooperative prices and joint profits across sigma — collusion premium."""
    sigmas = sweep.sigma_values
    nash_prices = [r.average_price for r in sweep.nash_results]
    coop_prices = [r.average_price for r in sweep.coop_results]
    nash_total = [sum(r.profits.values()) for r in sweep.nash_results]
    coop_total = [sum(r.profits.values()) for r in sweep.coop_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(sigmas, nash_prices, marker="o", color=PALETTE["nash"], label="Bertrand-Nash")
    ax.plot(sigmas, coop_prices, marker="s", color=PALETTE["cooperative"], label="Cooperative")
    ax.fill_between(sigmas, nash_prices, coop_prices, alpha=0.18,
                    color=PALETTE["cooperative"], label="Collusion premium")
    ax.set_xlabel("sigma  (substitutability)")
    ax.set_ylabel("Average price ($/bbl)")
    ax.set_title("Average price: Nash vs cooperative")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(sigmas, nash_total, marker="o", color=PALETTE["nash"], label="Bertrand-Nash")
    ax.plot(sigmas, coop_total, marker="s", color=PALETTE["cooperative"], label="Cooperative")
    ax.set_xlabel("sigma  (substitutability)")
    ax.set_ylabel("Total industry profit")
    ax.set_title("Joint profit: Nash vs cooperative")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Bertrand collusion premium  (gap shrinks as sigma -> 1)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Section B — Capacity-constraint analysis plots
# ===========================================================================

def plot_capacity_constrained_vs_unconstrained(comparison, players, path: str) -> None:
    """Side-by-side grouped bars: Nash price, total Q, profits, cartel,
    constrained vs unconstrained."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    nash_u = comparison.unconstrained["nash"]
    nash_c = comparison.constrained["nash"]
    cartel_u = comparison.unconstrained["cartel"]
    cartel_c = comparison.constrained["cartel"]

    ax = axes[0, 0]
    ax.bar(["Unconstrained", "Constrained"],
           [nash_u.price, nash_c.price],
           color=[PALETTE["nash"], PALETTE["stackelberg"]],
           edgecolor="black", linewidth=0.7)
    for i, v in enumerate([nash_u.price, nash_c.price]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("$/bbl")
    ax.set_title("Nash price")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(["Unconstrained", "Constrained"],
           [nash_u.total_quantity, nash_c.total_quantity],
           color=[PALETTE["nash"], PALETTE["stackelberg"]],
           edgecolor="black", linewidth=0.7)
    for i, v in enumerate([nash_u.total_quantity, nash_c.total_quantity]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("mbd")
    ax.set_title("Nash total quantity")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    x = np.arange(len(players))
    w = 0.38
    ax.bar(x - w / 2, [nash_u.profits[p] for p in players], w,
           color=PALETTE["nash"], edgecolor="black", linewidth=0.6,
           label="Unconstrained")
    ax.bar(x + w / 2, [nash_c.profits[p] for p in players], w,
           color=PALETTE["stackelberg"], edgecolor="black", linewidth=0.6,
           label="Constrained")
    ax.set_xticks(x)
    ax.set_xticklabels(players)
    ax.set_ylabel("Profit")
    ax.set_title("Nash profits per player")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(["U-Nash", "U-Cartel", "C-Nash", "C-Cartel"],
           [sum(nash_u.profits.values()), sum(cartel_u.quota_profits.values()),
            sum(nash_c.profits.values()), sum(cartel_c.quota_profits.values())],
           color=[PALETTE["nash"], PALETTE["cooperative"],
                  PALETTE["stackelberg"], PALETTE["OPEC"]],
           edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Total industry profit")
    ax.set_title("Industry profit: Nash vs Cartel")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Capacity constraints — constrained vs unconstrained",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_opec_sweep(sweep_df, path: str) -> None:
    """4-panel sweep of OPEC capacity → Nash price, OPEC profit, delta*, HHI."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cap = sweep_df["opec_cap"].values
    panels = [
        (axes[0, 0], "nash_price", "$/bbl",       "Nash price",                 PALETTE["nash"]),
        (axes[0, 1], "opec_profit", "Profit",      "OPEC Nash profit",           PALETTE["OPEC"]),
        (axes[1, 0], "delta_star",  "delta*",      "Folk-theorem delta* (binding)", PALETTE["cooperative"]),
        (axes[1, 1], "hhi",         "HHI",         "Market concentration (HHI)", PALETTE["stackelberg"]),
    ]
    for ax, col, ylabel, title, color in panels:
        ax.plot(cap, sweep_df[col], marker="o", color=color, linewidth=1.8)
        ax.set_xlabel("OPEC capacity (mbd)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("OPEC capacity sweep — comparative statics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_binding_analysis(comparison, players, path: str) -> None:
    """Stacked bar: used vs slack capacity at Nash and at cartel quotas."""
    caps = comparison.constrained["caps_map"]
    nash_c = comparison.constrained["nash"]
    cartel_c = comparison.constrained["cartel"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, q_map in zip(
        axes, ["Nash", "Cartel quotas"],
        [nash_c.quantities, cartel_c.quotas],
    ):
        used = [q_map[p] for p in players]
        slack = [max(0.0, caps[p] - q_map[p]) for p in players]
        ax.bar(players, used, color=PALETTE["nash"], edgecolor="black",
               linewidth=0.6, label="Used")
        ax.bar(players, slack, bottom=used, color="#dddddd",
               edgecolor="black", linewidth=0.6, label="Slack")
        for i, p in enumerate(players):
            ax.text(i, caps[p], f"cap={caps[p]:.0f}", ha="center", va="bottom",
                    fontsize=9, color="black")
        ax.set_ylabel("Output (mbd)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Binding-constraint analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_folk_theorem(comparison, players, path: str) -> None:
    """delta* per player: constrained vs unconstrained."""
    folk_u = comparison.unconstrained["folk"]
    folk_c = comparison.constrained["folk"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(players))
    w = 0.38
    ax.bar(x - w / 2, [folk_u.delta_star[p] for p in players], w,
           color=PALETTE["nash"], edgecolor="black", linewidth=0.6,
           label="Unconstrained")
    ax.bar(x + w / 2, [folk_c.delta_star[p] for p in players], w,
           color=PALETTE["cooperative"], edgecolor="black", linewidth=0.6,
           label="Constrained")
    ax.axhline(0.95, linestyle="--", color="black", linewidth=1.0,
               label="Calibrated delta = 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(players)
    ax.set_ylabel("delta*  (critical discount factor)")
    ax.set_title("Folk-theorem delta*: capacity vs no capacity",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Section C — Welfare and deadweight-loss plots
# ===========================================================================

def plot_welfare_decomposition(decompositions, path: str) -> None:
    """Stacked bar: CS + PS + DWL across market structures."""
    labels = [d.structure_label for d in decompositions]
    cs = np.array([d.consumer_surplus for d in decompositions])
    ps = np.array([d.producer_surplus for d in decompositions])
    dwl = np.array([d.deadweight_loss for d in decompositions])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, cs, color="#4c78a8", edgecolor="black",
           linewidth=0.6, label="Consumer surplus")
    ax.bar(labels, ps, bottom=cs, color="#54a24b", edgecolor="black",
           linewidth=0.6, label="Producer surplus")
    ax.bar(labels, dwl, bottom=cs + ps, color="#e45756", edgecolor="black",
           linewidth=0.6, label="Deadweight loss")
    for i, d in enumerate(decompositions):
        ax.text(i, cs[i] + ps[i] + dwl[i],
                f"W={d.total_welfare:,.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Welfare (illustrative units)")
    ax.set_title("Welfare decomposition across market structures",
                 fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_welfare_dwl_comparison(decompositions, path: str) -> None:
    """Horizontal bar ranking market structures by deadweight loss."""
    labels = [d.structure_label for d in decompositions]
    dwl = [d.deadweight_loss for d in decompositions]
    order = np.argsort(dwl)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sorted_labels = [labels[i] for i in order]
    sorted_dwl = [dwl[i] for i in order]
    bars = ax.barh(sorted_labels, sorted_dwl,
                   color=[PALETTE["cooperative"] if v == 0 else PALETTE["nash"]
                          for v in sorted_dwl],
                   edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, sorted_dwl):
        ax.text(v, b.get_y() + b.get_height() / 2,
                f"  {v:,.1f}", va="center", fontsize=9)
    ax.set_xlabel("Deadweight loss (welfare units)")
    ax.set_title("Deadweight loss ranking", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_welfare_carbon_tax(carbon_results, path: str) -> None:
    """3-panel: prices, DWL and collusion premium % vs carbon tax."""
    taxes = [c.tax for c in carbon_results]
    nash_p = [c.nash.price for c in carbon_results]
    cartel_p = [c.cartel.quota_price for c in carbon_results]
    dwl_n = [c.dwl_nash for c in carbon_results]
    dwl_c = [c.dwl_cartel for c in carbon_results]
    premium = [c.collusion_premium_pct for c in carbon_results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axes[0]
    ax.plot(taxes, nash_p, marker="o", color=PALETTE["nash"], label="Nash")
    ax.plot(taxes, cartel_p, marker="s", color=PALETTE["cooperative"], label="Cartel")
    ax.set_xlabel("Carbon tax ($/bbl)")
    ax.set_ylabel("Price ($/bbl)")
    ax.set_title("Equilibrium price vs tax")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(taxes, dwl_n, marker="o", color=PALETTE["nash"], label="DWL — Nash")
    ax.plot(taxes, dwl_c, marker="s", color=PALETTE["cooperative"], label="DWL — Cartel")
    ax.set_xlabel("Carbon tax ($/bbl)")
    ax.set_ylabel("Deadweight loss")
    ax.set_title("Deadweight loss vs tax")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(taxes, premium, marker="o", color=PALETTE["OPEC"])
    ax.set_xlabel("Carbon tax ($/bbl)")
    ax.set_ylabel("Collusion premium (%)")
    ax.set_title("Collusion premium vs tax")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Carbon-tax interaction with collusion incentives",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_welfare_surplus_distribution(decompositions, players, path: str) -> None:
    """Stacked bar: total welfare split between consumers and each producer.

    Consumers are drawn in a neutral purple to keep them visually distinct
    from US (blue), OPEC (orange), and RUS (green) producer stacks; the
    original palette used the same blue for both Consumers and Producer US,
    which made the two strata indistinguishable in the stacked bars.
    """
    CONSUMER_COLOR = "#5e3c99"  # distinct from any producer colour
    labels = [d.structure_label for d in decompositions]
    cs = np.array([d.consumer_surplus for d in decompositions])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros_like(cs, dtype=float)
    ax.bar(labels, cs, bottom=bottom, color=CONSUMER_COLOR,
           edgecolor="black", linewidth=0.6, label="Consumers")
    bottom = bottom + cs
    for p, color in zip(players, [PALETTE.get(p, "grey") for p in players]):
        prof = np.array([d.profits_by_player.get(p, 0.0) for d in decompositions])
        ax.bar(labels, prof, bottom=bottom, color=color,
               edgecolor="black", linewidth=0.6, label=f"Producer {p}")
        bottom = bottom + prof
    ax.set_ylabel("Surplus (illustrative units)")
    ax.set_title("Surplus distribution by stakeholder", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Section D — N-player sensitivity plots
# ===========================================================================

def plot_n_player_price_quantity(sweep, path: str) -> None:
    """2-panel: Nash price (with cartel benchmark) and total Q vs n."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    n = sweep.n_values
    ax = axes[0]
    ax.plot(n, sweep.nash_prices, marker="o", color=PALETTE["nash"], label="Nash")
    ax.plot(n, sweep.cartel_prices, marker="s", color=PALETTE["cooperative"],
            linestyle="--", label="Cartel")
    ax.set_xlabel("Number of players n")
    ax.set_ylabel("Price ($/bbl)")
    ax.set_title("Nash and cartel prices vs n")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(n, sweep.nash_total_quantities, marker="o", color=PALETTE["nash"])
    ax.set_xlabel("Number of players n")
    ax.set_ylabel("Total quantity (mbd)")
    ax.set_title("Total Nash output vs n")
    ax.grid(True, alpha=0.3)
    fig.suptitle("N-player sensitivity — price and quantity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_n_player_cooperation(sweep, path: str) -> None:
    """2-panel: delta* (binding) and HHI vs n."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    n = sweep.n_values
    ax = axes[0]
    ax.plot(n, sweep.delta_star_binding, marker="o",
            color=PALETTE["cooperative"], linewidth=1.8)
    ax.axhline(0.95, linestyle="--", color="black", linewidth=1.0,
               label="Calibrated delta = 0.95")
    ax.set_xlabel("Number of players n")
    ax.set_ylabel("delta* (binding)")
    ax.set_title("Folk-theorem threshold vs n")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(n, sweep.hhi_values, marker="o", color=PALETTE["stackelberg"],
            linewidth=1.8)
    ax.set_xlabel("Number of players n")
    ax.set_ylabel("HHI")
    ax.set_title("Market concentration vs n")
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        "N-player sensitivity — cooperation sustainability and concentration",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_n_player_opec_power(sweep, path: str) -> None:
    """3-panel: OPEC profit, Shapley value, market share vs n."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    n = sweep.n_values
    ax = axes[0]
    ax.plot(n, sweep.opec_profits_nash, marker="o", color=PALETTE["nash"], label="Nash")
    ax.plot(n, sweep.opec_profits_cartel, marker="s",
            color=PALETTE["cooperative"], label="Cartel")
    ax.set_xlabel("n")
    ax.set_ylabel("OPEC profit")
    ax.set_title("OPEC profits vs n")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(n, sweep.opec_shapley, marker="o", color=PALETTE["OPEC"], linewidth=1.8)
    ax.set_xlabel("n")
    ax.set_ylabel("OPEC Shapley value")
    ax.set_title("OPEC bargaining power vs n")
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    ax.plot(n, sweep.opec_market_shares, marker="o",
            color=PALETTE["OPEC"], linewidth=1.8)
    ax.set_xlabel("n")
    ax.set_ylabel("Market share (Nash)")
    ax.set_title("OPEC market share vs n")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Dilution of OPEC's power as the market fragments",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Section E — Correlated-equilibrium plots
# ===========================================================================

def plot_correlated_eq_comparison(ce_compare, players, path: str) -> None:
    """Grouped bar: price, Q and per-player profits across Nash, CEs and Cartel."""
    nash = ce_compare.nash
    cartel = ce_compare.cartel
    ce = ce_compare.ce_results

    structures = ["Nash"] + list(ce.keys()) + ["Cartel"]
    prices = [nash.price] + [ce[k].expected_price for k in ce] + [cartel.quota_price]
    qs = (
        [nash.total_quantity]
        + [sum(ce[k].expected_quantities.values()) for k in ce]
        + [cartel.total_output]
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.bar(structures, prices,
           color=[PALETTE["nash"]] + [PALETTE["stackelberg"]] * len(ce)
                  + [PALETTE["cooperative"]],
           edgecolor="black", linewidth=0.6)
    for i, v in enumerate(prices):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Price ($/bbl)")
    ax.set_title("Equilibrium price")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(structures, qs,
           color=[PALETTE["nash"]] + [PALETTE["stackelberg"]] * len(ce)
                  + [PALETTE["cooperative"]],
           edgecolor="black", linewidth=0.6)
    for i, v in enumerate(qs):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mbd")
    ax.set_title("Total quantity")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    x = np.arange(len(players))
    w = 0.8 / len(structures)
    for i, s in enumerate(structures):
        if s == "Nash":
            vals = [nash.profits[p] for p in players]
            color = PALETTE["nash"]
        elif s == "Cartel":
            vals = [cartel.quota_profits[p] for p in players]
            color = PALETTE["cooperative"]
        else:
            vals = [ce[s].expected_profits[p] for p in players]
            color = PALETTE["stackelberg"]
        ax.bar(x + (i - (len(structures) - 1) / 2) * w, vals, w,
               color=color, edgecolor="black", linewidth=0.5,
               label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(players)
    ax.set_ylabel("Profit")
    ax.set_title("Profits per player")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    support_sizes = [ce[k].support_size for k in ce]
    ax.bar(list(ce.keys()), support_sizes, color=PALETTE["stackelberg"],
           edgecolor="black", linewidth=0.6)
    for i, v in enumerate(support_sizes):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# action profiles with p>0")
    ax.set_title("CE support size")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Correlated equilibrium vs Nash and Cartel", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cartel_consumer_surplus_proxy(cartel) -> float:
    """Linear-demand CS approximation at the cartel quota."""
    Q = cartel.total_output
    P = cartel.quota_price
    a = P + Q
    return 0.5 * (a - P) * Q


def plot_correlated_eq_welfare(ce_compare, path: str) -> None:
    """Stacked bar of CS+PS for Nash, each CE, and Cartel."""
    nash = ce_compare.nash
    cartel = ce_compare.cartel
    ce = ce_compare.ce_results
    labels = ["Nash"] + list(ce.keys()) + ["Cartel"]
    cs = np.asarray(
        [nash.consumer_surplus]
        + [ce[k].expected_consumer_surplus for k in ce]
        + [_cartel_consumer_surplus_proxy(cartel)]
    )
    ps = np.asarray(
        [sum(nash.profits.values())]
        + [sum(ce[k].expected_profits.values()) for k in ce]
        + [sum(cartel.quota_profits.values())]
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, cs, color="#4c78a8", edgecolor="black",
           linewidth=0.6, label="Consumer surplus")
    ax.bar(labels, ps, bottom=cs, color="#54a24b", edgecolor="black",
           linewidth=0.6, label="Producer surplus")
    for i in range(len(labels)):
        ax.text(i, cs[i] + ps[i],
                f"W={cs[i] + ps[i]:,.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Welfare")
    ax.set_title("Welfare across Nash, correlated equilibria, and cartel",
                 fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ce_marginal_q_opec_q_us(ce_result, players, n_grid):
    """Return the marginal CE distribution over (q_OPEC, q_US) as a 2-D array,
    with q_OPEC indexing rows (origin lower) and q_US indexing columns.
    Other players are integrated out.
    """
    joint = np.asarray(ce_result.prob_distribution).reshape((n_grid,) * len(players))
    if "OPEC" in players and "US" in players:
        ax_opec = players.index("OPEC")
        ax_us = players.index("US")
    else:
        ax_opec, ax_us = 0, 1
    other_axes = tuple(i for i in range(len(players)) if i not in (ax_opec, ax_us))
    marg = joint.sum(axis=other_axes) if other_axes else joint
    if ax_opec > ax_us:
        marg = marg.T
    return marg


def plot_correlated_eq_support(ce_result, players, action_grid, path: str) -> None:
    """Heatmap of the CE marginal probability over (q_OPEC, q_US) for a single
    objective.  Retained for backward compatibility; the multi-objective
    version :func:`plot_correlated_eq_support_all` is preferred for the
    thesis report."""
    n_grid = len(action_grid)
    if len(players) < 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(ce_result.prob_distribution)),
               ce_result.prob_distribution,
               color=PALETTE["stackelberg"], edgecolor="black", linewidth=0.4)
        ax.set_title("CE probability distribution")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    marg = _ce_marginal_q_opec_q_us(ce_result, players, n_grid)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(marg, origin="lower", cmap="viridis",
                   extent=[action_grid[0], action_grid[-1],
                           action_grid[0], action_grid[-1]],
                   aspect="auto")
    fig.colorbar(im, ax=ax, label="Probability")
    ax.set_xlabel("q_US (mbd)")
    ax.set_ylabel("q_OPEC (mbd)")
    ax.set_title(
        f"CE recommendation distribution  ({ce_result.objective})\n"
        f"marginal over (q_OPEC, q_US); q_RUS integrated out",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlated_eq_support_all(comparison, path: str) -> None:
    """Side-by-side heatmaps of the CE marginal probability over (q_OPEC, q_US)
    for every solved CE objective (typically max_welfare, max_joint_profit,
    max_min_profit).  Each cell of the heatmap is one bin of the discretised
    action grid; the colour intensity gives the probability that the mediator
    recommends that (q_US, q_OPEC) pair.  q_RUS is integrated out because the
    full 3-D distribution cannot be displayed as a flat heatmap.
    """
    objectives = list(comparison.ce_results.keys())
    n = len(objectives)
    players = comparison.players
    action_grid = comparison.action_grid
    n_grid = len(action_grid)
    # The action grid is linspace(0, q_max, n_grid); each pixel of the
    # heatmap should be centred on its grid value, not on the bin edge,
    # so the red expected-value marker lines up with the highlighted cell.
    half_bin = (action_grid[-1] - action_grid[0]) / (n_grid - 1) / 2

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.4))
    if n == 1:
        axes = [axes]
    for ax, obj in zip(axes, objectives):
        ce = comparison.ce_results[obj]
        marg = _ce_marginal_q_opec_q_us(ce, players, n_grid)
        im = ax.imshow(marg, origin="lower", cmap="viridis",
                       extent=[action_grid[0] - half_bin,
                               action_grid[-1] + half_bin,
                               action_grid[0] - half_bin,
                               action_grid[-1] + half_bin],
                       aspect="auto", vmin=0.0, vmax=1.0)
        # Mark the expected (q_US, q_OPEC) with a red cross
        exp_q_us = ce.expected_quantities.get("US", float("nan"))
        exp_q_opec = ce.expected_quantities.get("OPEC", float("nan"))
        ax.scatter([exp_q_us], [exp_q_opec], marker="x", color="red", s=120,
                   linewidths=2.2, zorder=5,
                   label=f"E[q] = ({exp_q_us:.1f}, {exp_q_opec:.1f})")
        ax.set_xlabel("q_US (mbd)")
        if ax is axes[0]:
            ax.set_ylabel("q_OPEC (mbd)")
        ax.set_title(
            f"{obj}\nsupport size = {ce.support_size}, "
            f"E[P] = {ce.expected_price:.2f}, "
            f"E[W] = {ce.expected_total_welfare:.0f}",
            fontsize=10, fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        fig.colorbar(im, ax=ax, label="Probability", fraction=0.046, pad=0.04)

    fig.suptitle(
        "Correlated-equilibrium recommendation distributions  "
        "(marginal over q_OPEC, q_US; q_RUS integrated out)",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Section F — Empirical-validation plots
# ===========================================================================

def plot_empirical_price_wars(validation_result, path: str) -> None:
    """Paired bar (historical vs model) of price drops for each historical episode."""
    episodes = validation_result.episodes
    labels = [e["episode"] for e in episodes]
    hist_drop = [e["historical_drop"] for e in episodes]
    model_drop = [e["model_drop"] for e in episodes]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, hist_drop, w, color=PALETTE["nash"],
           edgecolor="black", linewidth=0.6, label="Historical drop")
    ax.bar(x + w / 2, model_drop, w, color=PALETTE["cooperative"],
           edgecolor="black", linewidth=0.6, label="Model drop")
    for i, (h, m) in enumerate(zip(hist_drop, model_drop)):
        ax.text(i - w / 2, h, f"{h:.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Price drop ($/bbl)")
    ax.set_title("Empirical vs model price-war depth", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_empirical_mechanism_match(validation_result, path: str) -> None:
    """Heatmap of (direction, magnitude, mechanism, overall) match per episode."""
    episodes = validation_result.episodes
    labels = [e["episode"] for e in episodes]
    cols = ["direction", "magnitude_30pct", "mechanism", "overall"]
    grid = np.array([[1.0 if e[k] else 0.0 for k in cols] for e in episodes])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Direction", "Magnitude (+/-30%)", "Mechanism", "Overall"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(cols)):
            ax.text(j, i, "Y" if grid[i, j] else "N",
                    ha="center", va="center", fontsize=14, color="black")
    ax.set_title(
        f"Empirical mechanism match — overall fit score = "
        f"{validation_result.overall_fit_score:.2f}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_empirical_model_vs_history(validation_result, path: str) -> None:
    """Scatter: x = historical price, y = model prediction; 45° line."""
    pts_hist, pts_model, labels = [], [], []
    for e in validation_result.episodes:
        pts_hist.extend([e["historical_pre"], e["historical_trough"]])
        pts_model.extend([e["model_pre"], e["model_trough"]])
        labels.extend([f"{e['episode']} pre", f"{e['episode']} trough"])
    pts_hist = np.array(pts_hist)
    pts_model = np.array(pts_model)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pts_hist, pts_model, s=60, color=PALETTE["OPEC"],
               edgecolors="black", linewidth=0.6, zorder=3)
    lo = min(pts_hist.min(), pts_model.min()) - 5
    hi = max(pts_hist.max(), pts_model.max()) + 5
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0,
            label="45 degree line (perfect prediction)")
    for x, y, lbl in zip(pts_hist, pts_model, labels):
        ax.annotate(lbl, (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("Historical price ($/bbl)")
    ax.set_ylabel("Model price ($/bbl)")
    ax.set_title("Model vs history", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
