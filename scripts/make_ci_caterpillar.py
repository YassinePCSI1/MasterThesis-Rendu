"""Generate the collusion-index caterpillar (forest) plot for Section 4.

One row per experimental regime: point estimate of the collusion index
with its 95% confidence interval, anchored to the two reference lines
CI=0 (static Nash) and CI=1 (joint-profit cartel). The figure gives the
reader a single visual map of where every Section-4 experiment lands.

Run from the repo root:  python scripts/make_ci_caterpillar.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (label, mean, lo, hi, group) -- values sourced verbatim from Section 4.
# lo/hi = None means no interval reported (point estimate only).
ROWS = [
    ("Single learner (OPEC)",        -0.19, -0.24, -0.13, "learners"),
    ("Duopoly (OPEC, US)",           -0.21, -0.24, -0.18, "learners"),
    ("Triopoly -- tail estimator",    0.61,  0.57,  0.67, "learners"),
    ("Triopoly -- greedy (headline)", 0.73,  0.62,  0.84, "learners"),
    ("Patience $\\gamma=0.50$",       0.27,  0.24,  0.29, "patience"),
    ("Patience $\\gamma=0.95$",       0.745, 0.67,  0.82, "patience"),
    ("Demand shocks (post-shock)",    0.96,  0.88,  1.04, "robust"),
    ("Coarse grid ($2.7\\times$)",    0.56,  None,  None, "robust"),
]

GROUP_COLOR = {
    "learners": "#1f4e79",
    "patience": "#2e7d32",
    "robust":   "#b8860b",
}

fig, ax = plt.subplots(figsize=(7.4, 4.0))

# Reference bands / lines.
ax.axvspan(-0.45, 0.0, color="#d62728", alpha=0.05)
ax.axvspan(0.0, 1.0, color="#2ca02c", alpha=0.05)
ax.axvline(0.0, color="#444444", lw=1.2, ls="--")
ax.axvline(1.0, color="#444444", lw=1.2, ls="--")
ax.text(0.0, len(ROWS) - 0.35, "  Nash (0)", color="#444444",
        fontsize=9, ha="left", va="bottom")
ax.text(1.0, len(ROWS) - 0.35, "Cartel (1)  ", color="#444444",
        fontsize=9, ha="right", va="bottom")

labels = []
for i, (label, mean, lo, hi, group) in enumerate(reversed(ROWS)):
    color = GROUP_COLOR[group]
    if lo is not None and hi is not None:
        ax.plot([lo, hi], [i, i], color=color, lw=2.0, solid_capstyle="round")
        ax.plot([lo, lo], [i - 0.12, i + 0.12], color=color, lw=1.5)
        ax.plot([hi, hi], [i - 0.12, i + 0.12], color=color, lw=1.5)
    marker = "o" if lo is not None else "D"
    ax.plot(mean, i, marker=marker, color=color, ms=8,
            mec="white", mew=1.0, zorder=5)
    ax.annotate(f"{mean:+.2f}", (mean, i), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color=color)
    labels.append(label)

ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_ylim(-0.6, len(ROWS) - 0.2)
ax.set_xlim(-0.45, 1.15)
ax.set_xlabel("Collusion index $\\mathrm{CI}$ "
              "(0 = competitive Nash, 1 = joint-profit cartel)", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", color="#cccccc", lw=0.5, alpha=0.6)

legend_handles = [
    Line2D([0], [0], color=GROUP_COLOR["learners"], lw=2, marker="o",
           label="Number of learners"),
    Line2D([0], [0], color=GROUP_COLOR["patience"], lw=2, marker="o",
           label="Patience sweep"),
    Line2D([0], [0], color=GROUP_COLOR["robust"], lw=2, marker="o",
           label="Robustness"),
    Line2D([0], [0], color="gray", lw=0, marker="D",
           label="point estimate (no CI reported)"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, framealpha=0.9)

fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "Rendu_final" / "figures" / "ci_caterpillar.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
