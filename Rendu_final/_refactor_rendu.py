"""Fix Rendu_final thesis structure: section hierarchy, backmatter floats."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PART3 = ROOT / "part_III_IV_body.tex"
THESIS_BM = ROOT.parent / "thesis" / "backmatter_tables.tex"
OUT_BM = ROOT / "backmatter_tables.tex"

ENV_RE = re.compile(
    r"\\begin\{(figure|table)\}\*?(?:\[[^\]]*\])?.*?\\end\{\1\}\*?",
    re.DOTALL,
)


def fix_section_hierarchy(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_marl = False
    in_falsification = False
    in_robustness = False

    for line in lines:
        if line.startswith("\\section{Multi-Agent"):
            in_marl = True
            in_falsification = in_robustness = False
            out.append(line)
            continue
        if line.startswith("\\section{Cross-Model") or line.startswith(
            "\\section{Policy Implications}"
        ):
            in_marl = in_falsification = in_robustness = False
            out.append(line)
            continue

        if in_marl and line.startswith("\\section{"):
            line = line.replace("\\section{", "\\subsection{", 1)
            if "Falsification" in line:
                line = line.replace(
                    "Falsification of Folk-Theorem retaliation",
                    "Falsification of explicit Folk-Theorem retaliation",
                )
                in_falsification = True
                in_robustness = False
            elif "Robustness battery" in line:
                in_robustness = True
                in_falsification = False
            else:
                in_falsification = in_robustness = False
            out.append(line)
            continue

        if (in_falsification or in_robustness) and line.startswith("\\subsection{"):
            line = line.replace("\\subsection{", "\\subsubsection{", 1)

        out.append(line)

    return "\n".join(out) + "\n"


def fix_ci_notation(text: str) -> str:
    """Standardise collusion-index notation: \\CI already includes the overline."""
    text = text.replace("\\overline{\\CI}^{\\text{greedy}}", "\\CI^{\\text{greedy}}")
    text = text.replace("$\\overline{\\CI} =", "$\\CI =")
    text = text.replace("($\\overline{\\CI} =", "($\\CI =")
    return text


def fix_section_references(text: str) -> str:
    """Align legacy Part labels with section numbering."""
    replacements = [
        (
            "The analytical frameworks of Parts~I and~II share a common",
            "The analytical frameworks of Sections~2 and~3 share a common",
        ),
        (
            "The unifying metric across Parts~I--III is the collusion index",
            "The unifying metric across Sections~2--4 is the collusion index",
        ),
        (
            "The sixteen analytical frameworks of Parts~I--II and the\n"
            "experimental evidence of Part~III converge toward a coherent",
            "The sixteen analytical frameworks of Sections~2 and~3 and the\n"
            "experimental evidence of Section~4 converge toward a coherent",
        ),
        (
            "The analytical pipeline of Parts~I--II established that the",
            "The analytical pipeline of Sections~2 and~3 established that the",
        ),
        (
            "The empirical pipeline of Part~III\nclosed the question",
            "The empirical pipeline of Section~4\nclosed the question",
        ),
        (
            "self-enforcing equilibrium. The empirical pipeline of Part~III\n"
            "closed the question",
            "self-enforcing equilibrium. The empirical pipeline of Section~4\n"
            "closed the question",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def fix_em_dashes(text: str) -> str:
    """Replace em dashes (---) in body prose with commas, colons, or parentheses."""
    replacements = [
        ("common knowledge of primitives---demand intercept, cost vector,\nrivals' strategies---or governed",
         "common knowledge of primitives (demand intercept, cost vector,\nrivals' strategies) or governed"),
        ("can \\emph{model-free} algorithmic agents---which receive the\nrealised market price as their sole feedback signal and never\nobserve rivals' individual production---spontaneously converge",
         "can \\emph{model-free} algorithmic agents, which receive the\nrealised market price as their sole feedback signal and never\nobserve rivals' individual production, spontaneously converge"),
        ("\\citeauthor{calvano2020} mechanism---documented in differentiated\nBertrand competition---survives the translation",
         "\\citeauthor{calvano2020} mechanism, documented in differentiated\nBertrand competition, survives the translation"),
        ("structure---the same Green--Porter environment studied analytically\nin Section~3.5---here confronts model-free learners",
         "structure: the same Green--Porter environment studied analytically\nin Section~3.5. Here, model-free learners"),
        ("is the discount factor---the algorithmic counterpart of the",
         "is the discount factor, the algorithmic counterpart of the"),
        ("cooperative cartel quota---a range wide enough to accommodate both",
         "cooperative cartel quota, a range wide enough to accommodate both"),
        ("below zero---encountered in the single-learner and duopoly regimes\ndocumented below---identify predatory pricing",
         "below zero (encountered in the single-learner and duopoly regimes\ndocumented below) identify predatory pricing"),
        ("baseline---uniform draws on the $15$-level action grid---yields",
         "baseline (uniform draws on the $15$-level action grid) yields"),
        ("does not converge to a unique equilibrium---it\nconverges",
         "does not converge to a unique equilibrium: it\nconverges"),
        ("score interval---the standard\ncorrection for normal-approximation breakdown at extreme\nproportions---to $\\Pr",
         "score interval, the standard\ncorrection for normal-approximation breakdown at extreme\nproportions, applied to $\\Pr"),
        ("only two learning agents---OPEC\nand the US---while Russia plays",
         "only two learning agents (OPEC\nand the US) while Russia plays"),
        ("cooperative basin entirely---a finding examined in further detail",
         "cooperative basin entirely; a finding examined in further detail"),
        ("$+2.00\\,\\text{mbd}$---roughly $0.13$ standard deviations of\ntheir pre-window output---an order of magnitude smaller",
         "$+2.00\\,\\text{mbd}$, roughly $0.13$ standard deviations of\ntheir pre-window output, an order of magnitude smaller"),
        ("$n_{\\text{seeds}} = 5$). The mean-drop criterion is met---the\nmedian drop of $11.57\\,\\$\\text{/bbl}$ corresponds to\napproximately $3.9\\sigma_P$---but the frequency",
         "$n_{\\text{seeds}} = 5$). The mean-drop criterion is met: the\nmedian drop of $11.57\\,\\$\\text{/bbl}$ corresponds to\napproximately $3.9\\sigma_P$; but the frequency"),
        ("is a \\emph{static} fixed point---a joint-output tuple selected",
         "is a \\emph{static} fixed point: a joint-output tuple selected"),
        ("not strengthen---it attenuates.",
         "not strengthen; it attenuates."),
        ("no threat of deviation-triggered punishment---closer in spirit",
         "no threat of deviation-triggered punishment, closer in spirit"),
        ("The policy implication---developed in\nSection~\\ref{sec:policy}---is that detection screens",
         "The policy implication, developed in\nSection~\\ref{sec:policy}, is that detection screens"),
        ("converged $\\CI$---computed against the unconstrained benchmarks\nto preserve cross-row comparability---yields $95\\%$ CIs",
         "converged $\\CI$, computed against the unconstrained benchmarks\nto preserve cross-row comparability, yields $95\\%$ CIs"),
        ("agents converge above that elevated benchmark---indicating that",
         "agents converge above that elevated benchmark, indicating that"),
        ("$0.70$ units---over four pooled standard errors.",
         "$0.70$ units, over four pooled standard errors."),
        ("approximately $50.70\\,\\text{mbd}$---$27\\%$\nabove the Nash level---and thereby drives",
         "approximately $50.70\\,\\text{mbd}$, $27\\%$\nabove the Nash level, and thereby drives"),
        ("\\emph{non-monotone} in the number of learners---predatory for\n$n_{\\text{learners}} \\in \\{1,\\,2\\}$, cooperative for\n$n_{\\text{learners}} = 3$---a departure from the standard",
         "\\emph{non-monotone} in the number of learners: predatory for\n$n_{\\text{learners}} \\in \\{1,\\,2\\}$, cooperative for\n$n_{\\text{learners}} = 3$; a departure from the standard"),
        ("agent's greedy quantity---encoded by colour---against the",
         "agent's greedy quantity, encoded by colour, against the"),
        ("premium---the ``green-cartel'' channel.",
         "premium: the ``green-cartel'' channel."),
        ("        around explicit time-series patterns---reaction-function\n        tests, punishment-cycle detectors, retaliation-window\n        analyses---will systematically fail against this class",
         "        around explicit time-series patterns (reaction-function\n        tests, punishment-cycle detectors, retaliation-window\n        analyses) will systematically fail against this class"),
        ("coordination? The answer---a path-dependent Markov-perfect\nlock-in rather than a Folk-Theorem retaliation\nequilibrium---is not merely a taxonomic refinement.",
         "coordination? The answer, a path-dependent Markov-perfect\nlock-in rather than a Folk-Theorem retaliation\nequilibrium, is not merely a taxonomic refinement."),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def clean_float_captions(block: str) -> str:
    """Replace em-dash separators in captions with colons; keep table dashes."""
    return re.sub(
        r"(\\caption\{[^}]*?) --- ([^}]*\})",
        r"\1: \2",
        block,
        flags=re.DOTALL,
    )


def extract_floats(text: str) -> tuple[str, list[str]]:
    floats: list[str] = []

    def repl(m: re.Match[str]) -> str:
        block = m.group(0)
        block = re.sub(r"\\begin\{(figure|table)\}\[[^\]]*\]", r"\\begin{\1}[H]", block)
        block = block.replace("[b]", "[H]")
        block = clean_float_captions(block)
        floats.append(block.strip())
        return ""

    cleaned = ENV_RE.sub(repl, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, floats


def extract_table(label: str, source: str) -> str:
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", source, re.DOTALL):
        if f"\\label{{{label}}}" in m.group(0):
            return m.group(0)
    raise ValueError(f"missing table {label}")


def extract_first_n_figures(figures_section: str, n: int = 38) -> str:
    blocks: list[str] = []
    for m in re.finditer(
        r"\\begin\{figure\}.*?\\end\{figure\}", figures_section, re.DOTALL
    ):
        block = m.group(0)
        block = re.sub(
            r"\\begin\{figure\}\[[^\]]*\]", r"\\begin{figure}[H]", block
        )
        blocks.append(block.strip())
        if len(blocks) == n:
            break
    if len(blocks) < n:
        raise ValueError(f"expected {n} figures, found {len(blocks)}")
    return "\n\n".join(blocks)


def suppress_list_entry(float_tex: str) -> str:
    """Use \\caption* so appendix floats do not appear in LOT/LOF."""
    return float_tex.replace("\\caption{", "\\caption*{", 1)


def make_appendix_a_table() -> str:
    """Multi-page longtable for Appendix A (full calibration table + note)."""
    return r"""
\section{Calibrated parameters of the linear-Cournot triopoly}
\label{app:calibration}

\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.19\linewidth} c c
  >{\raggedright\arraybackslash}p{0.46\linewidth}@{}}
\caption*{Calibrated parameters of the linear-Cournot triopoly, grouped
by analytical block. Each row reports the parameter, its symbol, its
calibrated value, and the rationale and source linking the value to the
empirical literature or to a deliberate modelling choice.}
\label{tab:calibration} \\
\toprule
\textbf{Parameter} & \textbf{Symbol} & \textbf{Value} & \textbf{Rationale and source} \\
\midrule
\endfirsthead
\caption*{(continued) Calibrated parameters of the linear-Cournot triopoly} \\
\toprule
\textbf{Parameter} & \textbf{Symbol} & \textbf{Value} & \textbf{Rationale and source} \\
\midrule
\endhead
\midrule
\multicolumn{4}{r}{\small(continued on next page)} \\
\endfoot
\bottomrule
\endlastfoot
\multicolumn{4}{l}{\emph{A. Demand}}\\
\midrule
Demand intercept & $a$ & \$140/bbl & Choke price at upper end of IEA
(2019) WEO long-run scenarios; implies monopoly $P$ of \$80, within
Saudi fiscal break-even range (IMF Article IV). \\
Demand slope & $b$ & $1.0$ & Normalisation. Implies own-price elasticity
of approx.\ $-0.75$ at triopoly Nash, mid-range of EIA Short-Term
Energy Outlook ($-0.4$ to $-1.0$) and \citet{hamilton2003}. \\
\midrule
\multicolumn{4}{l}{\emph{B. Marginal costs (short-run, well-level, per barrel)}}\\
\midrule
OPEC marginal cost & $c_{\text{OPEC}}$ & \$20 & Gulf wellhead lifting
($\sim$\$5-\$10, \citealp{bp2022}) plus transport, infrastructure and
fiscal overhead. Consistent with Aramco Q1 2026 disclosed cost.
\emph{Not} the fiscal break-even of \$80. \\
Russia marginal cost & $c_{\text{RUS}}$ & \$35 & Siberian lifting
($\sim$\$15-\$20, \citealp{imf2021}), Urals export differential
($\sim$\$10-\$15), plus $\sim$\$5 sanctions premium. \\
US shale cost & $c_{\text{US}}$ & \$45 & Well-level break-even
(Dallas Fed Energy Survey, 2020). Firm-level Permian break-even
(\$61-\$62 in Q1 2025 survey) embeds corporate overhead, outside
short-run Cournot best-response margin. \\
\midrule
\multicolumn{4}{l}{\emph{C. Capacities (mbd)}}\\
\midrule
OPEC capacity & $\mathrm{cap}_{\text{OPEC}}$ & $40$ & Sum of sustainable
capacity of OPEC core (OPEC MOMR, 2024-2026 avg). Set at unconstrained
Nash output to make binding at Nash and slack at cartel. \\
Russia capacity & $\mathrm{cap}_{\text{RUS}}$ & $35$ & Upper bound on
non-OPEC OPEC+ bloc expansion under existing infrastructure
(\citealp[p.~??]{henderson2015}; \citealp{eia2025}). \\
US capacity & $\mathrm{cap}_{\text{US}}$ & $30$ & Upper bound for US
shale consistent with Permian Wolfcamp tier estimates (Smith, 2016;
EIA Permian assessment). \\
OPEC capacity sweep & & $25$-$60$ & Range spans 2022-23 tight regime
and 2020 slack regime; brackets all empirically observed levels. \\
\midrule
\multicolumn{4}{l}{\emph{D. Time and patience}}\\
\midrule
Discount factor & $\delta$ & $0.95$ & Quarterly frequency matching
OPEC+ meeting cadence. Annualised $\sim$19\%, deliberately elevated to
proxy political and fiscal impatience. Comfortably above binding
$\delta_{\text{OPEC}}^\ast = 0.529$. \\
Horizon & $T$ & $50$ qtrs & Long enough for transients to die, short
enough to avoid reserves-depletion and energy-transition dynamics that
the static cost vector cannot capture. \\
Inertia & $\lambda$ & $0.2$ & Quarterly partial-adjustment speed;
half-life $\sim$3 quarters, matching \citet{almoguera2011}
regime-switch adjustment estimates. \\
\midrule
\multicolumn{4}{l}{\emph{E. Robustness and welfare}}\\
\midrule
Bertrand substitutability & $\sigma$ & $0.6$ & Calibrated against
observed Brent/WTI/Urals/sour spreads; centre of implied $0.5$-$0.7$
substitution range. \\
Carbon tax sweep & $\tau$ & \$0-50 & Spans $0$ to $\sim$\$$116$/tCO$_2$,
covering EU ETS settlement levels and the OECD-IMF social-cost-of-carbon
2030 range (at $0.43$~tCO$_2$/bbl). \\
Competitive benchmark & $P_c$ & \$20 ($=c_{\min}$) & Aggressive
upper-bound convention consistent with Lombardi and Van Robays (2011) and
Hochman and Zilberman (2015). Weighted-average benchmark would halve DWL. \\
\midrule
\multicolumn{4}{l}{\emph{F. Stochastic demand}}\\
\midrule
$AR(1)$ persistence & $\rho$ & $0.6$ & Quarterly autocorrelation of
detrended global oil-demand growth, 1990-2024 (IEA OMR data, authors'
calculation). \\
$AR(1)$ std deviation & $\sigma_a$ & $8$ & $\sim$$6\%$ of intercept;
matches \citet{hamilton2003}, \citet{kilian2009}. \\
Monte Carlo paths & $N_{\text{mc}}$ & $300$ & MC standard error on
mean price $\sim$$0.46$, well below \$1/bbl economic threshold.
Doubling changes no result by $>0.1$. \\
Simulation horizon & & $200$ qtrs & Long enough for steady-state
cooperation and punishment fractions to converge. \\
GP trigger price & $\bar{p}$ & \$68 & Midpoint of Nash (\$60) and
cartel (\$80). \citet{porter1983} and \citet{almoguera2011} convention;
monotone robustness over range. \\
Punishment length & $T_p$ & $10$ qtrs & Empirical envelope of 1985-86
(8 qtrs) and 2020 (5 weeks plus quota uncertainty) episodes. \\
\midrule
\multicolumn{4}{l}{\emph{G. Correlated equilibrium}}\\
\midrule
Grid resolution & & $10$ levels & $1{,}000$ LP variables; 20-level grid
(8{,}000 vars) yields same qualitative answers in spot-check runs. \\
Objectives & & $3$ & Max-welfare (utilitarian), max-joint-profit
(cartel), max-min (Rawlsian). \\
\midrule
\multicolumn{4}{l}{\emph{H. Evolutionary game}}\\
\midrule
PD payoffs & & $(2025, 1800, 1600, 1500)$ & Derived from Cournot
calibration: $\pi_{\text{dev}}$, $\pi_{\text{coop}}$, $\pi_{\text{nash}}$,
$\pi_{\text{sucker}}$. Not chosen independently. \\
Punishment multiplier & & $1.0$-$2.0$ & Spans no punishment, 2020
Saudi $50\%$ production surge, and doubling consistent with installed
capacity. \\
\end{longtable}

\medskip
\noindent\emph{Note:} parameter values reflect the joint constraint that
(a)~each number is empirically anchored or explicitly normalised,
(b)~the binding constraints in Folk Theorem and Green--Porter analyses
are derived rather than imposed, and (c)~the model remains analytically
tractable across the sixteen frameworks. Sensitivity analyses, reported
in the relevant figures, confirm that qualitative conclusions are robust
to perturbations within the empirically defensible range of each
parameter.
"""


def build_backmatter(part3_floats: list[str]) -> str:
    thesis_bm = THESIS_BM.read_text()
    cal = make_appendix_a_table()
    guide = suppress_list_entry(extract_table("tab:reading-guide", thesis_bm))
    gp = extract_table("tab:green-porter", thesis_bm)

    figures_block = thesis_bm.split("\\section*{Figures}")[1]
    figures_block = figures_block.split("% --- Appendix (literature")[0]
    figures_38 = extract_first_n_figures(figures_block, 38)

    part3_tables: list[str] = []
    part3_figs: list[str] = []
    for block in part3_floats:
        if block.startswith("\\begin{table}"):
            part3_tables.append(block)
        else:
            part3_figs.append(block)

    lit_start = thesis_bm.find("% --- Appendix (literature")
    if lit_start == -1:
        raise ValueError("literature appendix block not found in thesis backmatter")
    lit = thesis_bm[lit_start:]
    lit = lit.replace("\\appendix\n", "", 1)
    lit = lit.replace("\\caption{", "\\caption*{")
    lit = re.sub(r"\\clearpage\s*", "", lit, count=1)
    lit = re.sub(r"(\\section\{)", r"\\clearpage\n\1", lit)

    return f"""% Auto-generated by _refactor_rendu.py
\\clearpage
\\section*{{Figures}}
\\addcontentsline{{toc}}{{section}}{{Figures}}
\\setcounter{{figure}}{{0}}

{figures_38}

{chr(10).join(part3_figs)}

\\clearpage
\\section*{{Tables}}
\\addcontentsline{{toc}}{{section}}{{Tables}}
\\setcounter{{table}}{{0}}

{gp}

{chr(10).join(part3_tables)}

\\clearpage
\\appendix

{cal}

\\clearpage
\\section{{Reading guide: sixteen-step analytical pipeline}}
\\label{{app:reading-guide}}

{guide}

{lit}
"""


def main() -> None:
    source = ROOT / "part_III_IV.tex"
    if not source.exists():
        source = PART3
    text = source.read_text()
    if "\\chapter{Multi-Agent" in text:
        start = text.find("\\chapter{Multi-Agent")
        end = text.find("% ============================================================\n%  BIBLIOGRAPHY")
        if end == -1:
            end = text.find("\\bibliographystyle")
        text = text[start:end].rstrip()
        text = text.replace("\\chapter{", "\\section{")
        header = "% Extracted from part_III_IV.tex for inclusion in main.tex\n\\setcounter{section}{3}\n\n"
        text = header + text + "\n"

    text = fix_section_hierarchy(text)
    text = fix_em_dashes(text)
    text = fix_ci_notation(text)
    text = fix_section_references(text)
    text, floats = extract_floats(text)
    PART3.write_text(text)
    OUT_BM.write_text(build_backmatter(floats))
    print(f"Updated {PART3.name}: removed {len(floats)} floats")
    print(f"Wrote {OUT_BM.name}")


if __name__ == "__main__":
    main()
