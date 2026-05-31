"""One-shot patch: merge baseline_report_core.md into report.generate_report."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_PY = ROOT / "src" / "report.py"

NEW_HEADER_AND_SETUP = r'''
    header_lines: list[str] = [
        "# Game Theory in Oil Producing Countries",
        "## Quantitative Model Report",
        "",
        "> *This report is auto-generated from the simulation outputs.*  ",
        "> *All values are illustrative model outputs calibrated for interpretability.*",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
        "*Core report (original **MasterThesis-main** baseline, §§1–11):*",
        "1. [Model Overview & Calibration](#1-model-overview--calibration)",
        "2. [The Benchmark: Static Cournot Equilibrium](#2-the-benchmark-static-cournot-equilibrium)",
        "3. [Static Market Structure: Market Power & First-Mover Advantage](#3-static-market-structure-market-power--first-mover-advantage)",
        "4. [From Static to Dynamic: Repeated Game Dynamics](#4-from-static-to-dynamic-repeated-game-dynamics)",
        "5. [Can Cooperation Be Sustained? Cartel, Punishment & Folk Theorem](#5-can-cooperation-be-sustained-cartel-punishment--folk-theorem)",
        "6. [Fair Sharing: Coalition Formation & Shapley Values](#6-fair-sharing-coalition-formation--shapley-values)",
        "7. [Robustness to Uncertainty: Stochastic Demand](#7-robustness-to-uncertainty-stochastic-demand)",
        "8. [Learning Without a Model: Reinforcement Learning](#8-learning-without-a-model-reinforcement-learning)",
        "9. [Population Dynamics: Evolutionary Game Theory](#9-population-dynamics-evolutionary-game-theory)",
        "10. [Cross-Model Synthesis & Conclusions](#10-cross-model-synthesis--conclusions)",
        "11. [Policy Implications](#11-policy-implications)",
        "",
        "*Extensions (added analyses, §§12–20):*",
        "12. [Price Competition: Bertrand Model](#12-price-competition-bertrand-model)",
        "13. [Capacity Constraints](#13-capacity-constraints)",
        "14. [Welfare & Deadweight Loss Analysis](#14-welfare--deadweight-loss-analysis)",
        "15. [N-Player Sensitivity: Market Fragmentation](#15-n-player-sensitivity-market-fragmentation)",
        "16. [Correlated Equilibrium: Can a Mediator Help?](#16-correlated-equilibrium-can-a-mediator-help)",
        "17. [Empirical Validation Against Historical Price Wars](#17-empirical-validation-against-historical-price-wars)",
        "18. [Multi-Agent Reinforcement Learning Under Imperfect Monitoring](#18-multi-agent-reinforcement-learning-under-imperfect-monitoring)",
        "19. [Extended Cross-Model Synthesis & Conclusions](#19-extended-cross-model-synthesis--conclusions)",
        "20. [Extended Policy Implications](#20-extended-policy-implications)",
        "",
        "---",
        "",
    ]

    _baseline_path = Path(__file__).with_name("baseline_report_core.md")
    baseline_lines = _baseline_path.read_text(encoding="utf-8").split("\n")

    bridge_lines: list[str] = [
        "",
        "---",
        "",
        "> **Extensions (§§12–20).**  Everything from here downward *adds* new models and ",
        "an updated synthesis to the baseline above.  **Sections 1–11 are preserved verbatim** ",
        "from the archived *MasterThesis-main* report.  §§19–20 extend—not replace—the ",
        "synthesis and policy discussion in §§10–11.",
        "",
        "---",
        "",
    ]

    # CSVs required by extension sections (core §§1–9 live in the baseline markdown file).
    df_static = _read(f"{output_dir}/static_equilibrium.csv")
    df_stack = _read(f"{output_dir}/stackelberg_comparison.csv")
    df_rl_bench = _read(f"{output_dir}/rl_benchmark_comparison.csv")

    ext_lines: list[str] = []
'''.lstrip("\n")

APPENDIX_SPLIT_OLD = '''        "  rather than on the algorithms themselves.",
        "",
        "---",
        "",
        "## Appendix: Output Files",
        "",
    ]
    for name, path_val in artifacts.items():
        lines.append(f"- **{name}**: `{os.path.basename(path_val)}`")'''

APPENDIX_SPLIT_NEW = '''        "  rather than on the algorithms themselves.",
        "",
        "---",
        "",
    ]
    _merged_extensions = _renumber_extension_headings(ext_lines)
    lines: list[str] = header_lines + baseline_lines + bridge_lines + _merged_extensions

    lines += [
        "## Appendix: Output Files",
        "",
    ]
    for name, path_val in artifacts.items():
        lines.append(f"- **{name}**: `{os.path.basename(path_val)}`")'''


def main() -> None:
    text = REPORT_PY.read_text(encoding="utf-8")

    if "ext_lines: list[str] = []" in text:
        print("Already patched:", REPORT_PY)
        return

    lines = text.split("\n")

    i0 = next(i for i, line in enumerate(lines) if line.strip() == "lines: list[str] = []")
    i_bertrand = next(
        i for i, line in enumerate(lines) if "# ──────────────────────────── 10. Bertrand" in line
    )

    new_header = NEW_HEADER_AND_SETUP.split("\n")
    lines = lines[:i0] + new_header + lines[i_bertrand:]

    body = "\n".join(lines)
    if APPENDIX_SPLIT_OLD not in body:
        raise SystemExit("appendix split anchor not found — report.py layout changed?")
    body = body.replace(APPENDIX_SPLIT_OLD, APPENDIX_SPLIT_NEW, 1)

    lines = body.split("\n")
    i_glue = next(
        i
        for i, line in enumerate(lines)
        if "_merged_extensions = _renumber_extension_headings(ext_lines)" in line
    )
    i_bertrand = next(
        i for i, line in enumerate(lines) if "# ──────────────────────────── 10. Bertrand" in line
    )

    for i in range(i_bertrand, i_glue):
        line = lines[i]
        if line.startswith("    lines +="):
            lines[i] = line.replace("    lines +=", "    ext_lines +=", 1)
        elif line.startswith("            lines.append("):
            lines[i] = line.replace("            lines.append(", "            ext_lines.append(", 1)

    out = "\n".join(lines)
    REPORT_PY.write_text(out + ("\n" if not out.endswith("\n") else ""), encoding="utf-8")
    print("patched", REPORT_PY)


if __name__ == "__main__":
    main()
