"""One-shot restructuring of outputs/report.tex.

New structure (centred on: 'does model-free AI converge to game-theoretic equilibria?'):

Part I  — Foundations: Static Market Environment
  §1  Model Overview & Calibration  (unchanged content; reading-guide table refreshed)
  §2  Static Cournot Equilibrium
  §3  Stackelberg & Market Power
  §4  Capacity Constraints           (moved from old §11)
  §5  Welfare & Deadweight Loss      (moved from old §12)
  §6  Bertrand Price Competition     (moved from old §10; framed as robustness check)

Part II — Strategic Dynamics: Theoretical Anchor
  §7  Repeated Game Dynamics         (old §4)
  §8  Cartel, Punishment & Folk Theorem  (old §5)
  §9  Correlated Equilibrium         (old §14)
  §10 Shapley Values                 (old §6)
  §11 Stochastic Demand & Green-Porter (old §7)
  §12 Evolutionary Dynamics (bridge to AI)  (old §9)

Part III — Model-Free AI in the Game-Theoretic Environment
  §13 Single-Agent RL                (old §8)
  §14 Multi-Agent RL                 (old §16)

Part IV — Synthesis
  §15 Cross-Model Synthesis: Does AI Converge to Theory?  (rewritten)
  §16 Policy Implications            (pruned)

Removed: §13 N-Player Sensitivity, §15 Empirical Validation (already deleted).
"""
from __future__ import annotations
import re
from pathlib import Path

REPORT_PATH = Path("outputs/report.tex")
text = REPORT_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Step 1. Split file into: preamble, body, footer (Figures + Appendix)
# ---------------------------------------------------------------------------
m_body_start = text.index("\\clearpage\n% ============================================================================\n\\part*{PART I")
m_figs = text.index("\\section*{Figures}")
m_figs_block_start = text.rfind("\\clearpage", 0, m_figs)

preamble = text[:m_body_start]
body = text[m_body_start:m_figs_block_start]
figures_appendix = text[m_figs_block_start:]

# ---------------------------------------------------------------------------
# Step 2. Parse body into sections.
# A "block" starts at \part*{...} or \section{...}; we collect title + content
# until the next \part* / \section / end of body.
# ---------------------------------------------------------------------------
SECTION_RE = re.compile(r"^(\\(?:part\*|section)\{[^}]+\})", re.MULTILINE)
matches = list(SECTION_RE.finditer(body))

blocks: list[tuple[str, str]] = []   # list of (header_line, content_after_header)
for i, m in enumerate(matches):
    start = m.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
    full = body[start:end]
    # First line is the header
    nl = full.index("\n")
    header = full[:nl]
    content = full[nl + 1:]
    blocks.append((header, content))

# Index blocks by a tag for clarity
def find_block(title_keyword: str) -> tuple[str, str]:
    for h, c in blocks:
        if title_keyword in h:
            return h, c
    raise KeyError(title_keyword)

part1 = find_block("PART I")
part2 = find_block("PART II")
part3 = find_block("PART III")
part4 = find_block("PART IV")

s_model_overview = find_block("Model Overview")
s_cournot        = find_block("Static Cournot")
s_market_power   = find_block("Market Power")
s_repeated       = find_block("Repeated Game Dynamics")
s_folk           = find_block("Folk Theorem")
s_shapley        = find_block("Coalition Formation")
s_stochastic     = find_block("Stochastic Demand")
s_single_rl      = find_block("Single-Agent Reinforcement Learning")
s_evolutionary   = find_block("Population Dynamics: Evolutionary")
s_bertrand       = find_block("Bertrand Model")
s_capacity       = find_block("Capacity Constraints")
s_welfare        = find_block("Welfare \\& Deadweight Loss")
s_correlated     = find_block("Correlated Equilibrium")
s_marl           = find_block("Multi-Agent Reinforcement Learning")
s_synthesis      = find_block("Cross-Model Synthesis")
s_policy         = find_block("Policy Implications")

# ---------------------------------------------------------------------------
# Step 3. New Part headers & intros
# ---------------------------------------------------------------------------
PART1_NEW = (
    "\\part*{PART I --- Foundations: The Static Market Environment}",
    """\\addcontentsline{toc}{part}{PART I --- Foundations: The Static Market Environment}
% ============================================================================
\\emph{Part I builds the market environment used throughout the thesis: who the}
\\emph{players are, the demand and cost calibration, and the static equilibria}
\\emph{(Cournot, Stackelberg, capacity-constrained, welfare-decomposed, and the}
\\emph{Bertrand robustness check). These five sections deliver the }\\textbf{game-theoretic}
\\emph{environment in which model-free AI agents will be turned loose in Part~III.}


"""
)

PART2_NEW = (
    "\\part*{PART II --- Strategic Dynamics: The Theoretical Anchor}",
    """\\addcontentsline{toc}{part}{PART II --- Strategic Dynamics: The Theoretical Anchor}
% ============================================================================
\\emph{Part II adds time, uncertainty and strategic refinements on top of the}
\\emph{static environment of Part~I. The output of this part is a }\\textbf{rich theoretical}
\\emph{benchmark}\\emph{ --- which equilibria are sustainable, under what discount factor,}
\\emph{with what monitoring structure --- against which we will later evaluate whether}
\\emph{model-free AI agents converge. Section~12 (Evolutionary Dynamics) is the bridge:}
\\emph{it reframes equilibrium as the outcome of an adaptive process at the population}
\\emph{level, paving the way for the individual-learner perspective of Part~III.}


"""
)

PART3_NEW = (
    "\\part*{PART III --- Model-Free AI in the Game-Theoretic Environment}",
    """\\addcontentsline{toc}{part}{PART III --- Model-Free AI in the Game-Theoretic Environment}
% ============================================================================
\\emph{This is the core of the thesis. We turn off the analytical toolkit and turn}
\\emph{on Q-learning agents that observe only the market price (Green-Porter}
\\emph{information structure, \\S{}11). We progressively remove rationality:}
\\emph{first one learner among two rational best-responders (\\S{}13), then a fully}
\\emph{model-free system in which all three producers learn simultaneously (\\S{}14).}
\\emph{The central question of the thesis is whether the equilibria carefully}
\\emph{characterised in Parts~I--II emerge endogenously from learning, without}
\\emph{any prior knowledge of the model.}


"""
)

PART4_NEW = (
    "\\part*{PART IV --- Synthesis}",
    """\\addcontentsline{toc}{part}{PART IV --- Synthesis}
% ============================================================================
\\emph{We confront Parts~I--II (what the theory predicts) with Part~III (what}
\\emph{model-free learners discover) and conclude on the central question:}
\\emph{does AI converge to the game-theoretic equilibria?}


"""
)

# ---------------------------------------------------------------------------
# Step 4. New §1 reading-guide table (replaces the existing one)
# ---------------------------------------------------------------------------
NEW_GUIDE_TABLE = """\\begin{longtable}[]{@{}
  >{\\raggedright\\arraybackslash}p{0.07\\linewidth}
  >{\\raggedright\\arraybackslash}p{0.27\\linewidth}
  >{\\raggedright\\arraybackslash}p{0.10\\linewidth}
  >{\\raggedright\\arraybackslash}p{0.51\\linewidth}@{}}
\\toprule\\noalign{}
\\begin{minipage}[b]{\\linewidth}\\raggedright \\S{} \\end{minipage}
& \\begin{minipage}[b]{\\linewidth}\\raggedright Framework \\end{minipage}
& \\begin{minipage}[b]{\\linewidth}\\raggedright Part \\end{minipage}
& \\begin{minipage}[b]{\\linewidth}\\raggedright Question addressed \\end{minipage} \\\\
\\midrule\\noalign{}
\\endhead
\\bottomrule\\noalign{}
\\endlastfoot
2  & Static Cournot                    & I   & One-shot Nash equilibrium of the triopoly. \\\\
3  & Stackelberg \\& Market Power        & I   & First-mover advantage; Lerner, HHI. \\\\
4  & Capacity Constraints              & I   & Do physical caps reshape the equilibrium? \\\\
5  & Welfare \\& DWL                     & I   & Social cost of oligopoly; carbon-tax interaction. \\\\
6  & Bertrand (robustness)             & I   & Does price competition change the picture? \\\\
7  & Repeated Game Dynamics            & II  & Convergence to Nash under inertia. \\\\
8  & Cartel + Folk Theorem             & II  & Can cooperation be sustained? Critical $\\delta^{*}$. \\\\
9  & Correlated Equilibrium            & II  & Can a mediator Pareto-improve on Nash? \\\\
10 & Coalitions + Shapley              & II  & Fair sharing of cooperative gains. \\\\
11 & Stochastic Demand + Green-Porter  & II  & Cooperation under imperfect monitoring. \\\\
12 & Evolutionary Dynamics             & II  & Which strategies survive imitation? (bridge to RL) \\\\
13 & Single-Agent RL                   & III & Can one learner discover Nash without a model? \\\\
14 & Multi-Agent RL                    & III & Do simultaneous learners reach the cartel? \\\\
15--16 & Synthesis \\& Policy             & IV  & Does AI converge to theory? Implications. \\\\
\\end{longtable}
"""

OLD_GUIDE_TABLE_RE = re.compile(
    r"\\begin\{longtable\}\[\]\{@\{\}\n(?:\s+>\{\\raggedright[^}]*\}p\{0\.2000\\linewidth\}\n){5}@\{\}\}.*?\\end\{longtable\}",
    re.DOTALL,
)
new_overview_content = OLD_GUIDE_TABLE_RE.sub(lambda _m: NEW_GUIDE_TABLE, s_model_overview[1], count=1)
s_model_overview = (s_model_overview[0], new_overview_content)

# ---------------------------------------------------------------------------
# Step 5. Patch transition paragraphs (\textit{\textbf{Transition $\to$} ...})
#        so each section flows into the NEXT one in the new order.
# ---------------------------------------------------------------------------
TRANSITION_RE = re.compile(
    r"\\noindent\\textit\{\\textbf\{Transition \$\\to\$\} [^}]*\}",
    re.DOTALL,
)

def replace_transition(content: str, new_text: str) -> str:
    replacement = "\\noindent\\textit{\\textbf{Transition $\\to$} " + new_text + "}"
    return TRANSITION_RE.sub(lambda _m: replacement, content, count=1)

# §3 Stackelberg → §4 Capacity
s_market_power = (s_market_power[0], replace_transition(
    s_market_power[1],
    "Sections 2--3 assumed unlimited production capacity. Real producers face physical ceilings; "
    "Section 4 turns them on and re-examines every static result.",
))

# §4 (new) Capacity → §5 Welfare (already correct)
# (existing transition already points to Welfare, keep as is)

# §5 (new) Welfare → §6 Bertrand (was: transitions to fragmentation / N-player)
s_welfare = (s_welfare[0], replace_transition(
    s_welfare[1],
    "The static analysis assumed Cournot quantity competition. Section~6 asks whether the "
    "qualitative picture (OPEC dominance, cooperation premium, welfare ranking) survives if "
    "producers compete on price instead of quantity --- the Bertrand robustness check.",
))

# §6 (new) Bertrand → Part II
s_bertrand = (s_bertrand[0], replace_transition(
    s_bertrand[1],
    "OPEC dominance is robust to the competition mode. Part~II now adds time, uncertainty and "
    "strategic refinements --- the dynamic theoretical benchmark against which Part~III's "
    "model-free learners will be evaluated.",
))

# §7 (new) Repeated → §8 Folk Theorem (already correct)

# §8 (new) Folk Theorem → §9 Correlated
s_folk = (s_folk[0], replace_transition(
    s_folk[1],
    "Cooperation is enforceable when players are patient ($\\delta^{*}\\approx 0.53 < 0.95$). "
    "But cooperation through grim-trigger punishment is one mechanism among several. "
    "Section~9 asks: can a mediator achieve a Pareto-improvement over Nash without requiring "
    "binding agreements? This is the correlated-equilibrium refinement.",
))

# §9 (new) Correlated → §10 Shapley
s_correlated = (s_correlated[0], replace_transition(
    s_correlated[1],
    "The correlated equilibrium provides an alternative coordination device. "
    "Whether through patience (Folk Theorem) or mediation (CE), cooperation generates a "
    "surplus that must be shared. Section~10 turns to the cooperative-game question of "
    "what each player's \\emph{fair share} of that surplus is.",
))

# §10 (new) Shapley → §11 Stochastic (already pointing to stochastic ~ keep)

# §11 (new) Stochastic → §12 Evolutionary
s_stochastic = (s_stochastic[0], replace_transition(
    s_stochastic[1],
    "Stochastic monitoring (Green-Porter) shows that cooperation can survive uncertainty, "
    "with regime switching as the equilibrium device. Section~12 changes the lens: instead "
    "of asking what \\emph{rational} players choose, we ask which strategies survive when "
    "producers \\emph{imitate} more successful peers. Evolutionary dynamics is the natural "
    "bridge from rational analysis to model-free learning (Part~III).",
))

# §12 (new) Evolutionary → Part III (the punchline transition)
s_evolutionary = (s_evolutionary[0], replace_transition(
    s_evolutionary[1],
    "At the population level, conditional punishers (Punish) are the only ESS that sustains "
    "cooperation. But what about \\emph{individual} learning? Part~III turns to Q-learning agents "
    "that observe only the market price --- progressively removing rationality from one, two, "
    "and finally all three producers --- and asks whether the equilibria of Parts~I--II "
    "emerge endogenously.",
))

# §13 (new) Single-Agent RL → §14 MARL
s_single_rl = (s_single_rl[0], replace_transition(
    s_single_rl[1],
    "A single learner discovers partial collusion ($P\\approx 62.5$, between Nash 60 and cartel 80). "
    "But it learns against \\emph{rational} myopic best-responders --- the rivals already know the "
    "model. Section~14 progressively removes that rationality: first two learners, then all three, "
    "under the Green-Porter information structure (\\S{}11).",
))

# MARL transition to Synthesis
s_marl = (s_marl[0], replace_transition(
    s_marl[1],
    "We now have a complete picture: theory predicts cooperation is enforceable; "
    "model-free learners with three simultaneous Q-agents reach it spontaneously. "
    "Part~IV synthesises the convergence verdict.",
))

# ---------------------------------------------------------------------------
# Step 6. Renumber section cross-references in body content.
#         Old numbering -> new numbering map.
# ---------------------------------------------------------------------------
OLD_TO_NEW = {
    2: 2,
    3: 3,
    4: 7,
    5: 8,
    6: 10,
    7: 11,
    8: 13,
    9: 12,
    10: 6,
    11: 4,
    12: 5,
    # 13 removed (N-Player)
    14: 9,
    # 15 removed (Empirical)
    16: 14,
    17: 15,
    18: 16,
}

def renumber(content: str) -> str:
    # Use a placeholder phase so we don't double-substitute.
    out = content
    # \S{}N forms
    def s_sub(m):
        n = int(m.group(1))
        if n in OLD_TO_NEW:
            return f"\\S{{}}__NEW{OLD_TO_NEW[n]}__"
        return m.group(0)
    out = re.sub(r"\\S\{\}(\d+)", s_sub, out)
    # "Section N" forms
    def sec_sub(m):
        n = int(m.group(1))
        if n in OLD_TO_NEW:
            return f"Section __NEW{OLD_TO_NEW[n]}__"
        return m.group(0)
    out = re.sub(r"Section (\d+)", sec_sub, out)
    # Resolve placeholders
    out = re.sub(r"__NEW(\d+)__", r"\1", out)
    return out

# Apply renumbering to every section block we'll keep.
section_blocks = [
    s_model_overview, s_cournot, s_market_power,
    s_capacity, s_welfare, s_bertrand,
    s_repeated, s_folk, s_correlated,
    s_shapley, s_stochastic, s_evolutionary,
    s_single_rl, s_marl,
    s_synthesis, s_policy,
]
section_blocks = [(h, renumber(c)) for h, c in section_blocks]

(   s_model_overview, s_cournot, s_market_power,
    s_capacity, s_welfare, s_bertrand,
    s_repeated, s_folk, s_correlated,
    s_shapley, s_stochastic, s_evolutionary,
    s_single_rl, s_marl,
    s_synthesis, s_policy,
) = section_blocks

# ---------------------------------------------------------------------------
# Step 7. Rewrite Bertrand opening paragraph so it's framed as robustness
# ---------------------------------------------------------------------------
BERTRAND_NEW_OPENING = (
    "This section is the \\textbf{robustness check} of Part~I: we have characterised the "
    "static environment in Sections 2--5 under \\textbf{quantity competition} (Cournot). "
    "Real crude is also traded on \\textbf{price} --- OPEC announces target prices, refiners bid, "
    "and grade differentials (WTI vs Brent vs Urals) sustain some market power even at marginal-"
    "cost pricing. The question of this section: do the qualitative conclusions of Sections 2--5 "
    "(OPEC dominance, cost-based ranking, cooperation premium) survive when firms compete on "
    "price instead of quantity?\n\n"
    "Pure homogeneous Bertrand collapses to marginal-cost pricing (the \\emph{Bertrand paradox})."
    " We therefore use a \\textbf{differentiated-products} specification with substitutability "
    "parameter $\\sigma\\in[0,1]$:"
)
s_bertrand = (
    s_bertrand[0],
    re.sub(
        r"Sections 2--9 modelled quantity competition.*?substitutability parameter \$\\sigma\$ \$\\in\$ \{\[\}0, 1\{\]\}:",
        lambda _m: BERTRAND_NEW_OPENING,
        s_bertrand[1],
        count=1,
        flags=re.DOTALL,
    ),
)

# Update Bertrand title
s_bertrand = ("\\section{Bertrand Price Competition (Robustness Check)}", s_bertrand[1])

# ---------------------------------------------------------------------------
# Step 8. Rewrite Synthesis (now §15) and Policy (now §16)
# ---------------------------------------------------------------------------
NEW_SYNTHESIS_TITLE = "\\section{Cross-Model Synthesis: Does Model-Free AI Converge to Theory?}"
NEW_SYNTHESIS = """The central question of this thesis is now testable. Parts~I--II built a
\\textbf{game-theoretic environment} (static equilibria, repeated-game refinements,
correlated equilibrium, cooperative-game allocations, stochastic monitoring, evolutionary
dynamics) and Part~III let \\textbf{model-free Q-learning agents} loose in that environment
with progressively less rationality.

\\subsection{The convergence ladder}
\\begin{longtable}[]{@{}
  >{\\raggedright\\arraybackslash}p{0.22\\linewidth}
  >{\\raggedright\\arraybackslash}p{0.10\\linewidth}
  >{\\raggedright\\arraybackslash}p{0.12\\linewidth}
  >{\\raggedright\\arraybackslash}p{0.10\\linewidth}
  >{\\raggedright\\arraybackslash}p{0.40\\linewidth}@{}}
\\toprule\\noalign{}
\\begin{minipage}[b]{\\linewidth}\\raggedright Configuration \\end{minipage}
& \\begin{minipage}[b]{\\linewidth}\\raggedright $P$ (\\$/bbl) \\end{minipage}
& \\begin{minipage}[b]{\\linewidth}\\raggedright Coll. index \\end{minipage}
& \\begin{minipage}[b]{\\linewidth}\\raggedright Source \\end{minipage}
& \\begin{minipage}[b]{\\linewidth}\\raggedright Interpretation \\end{minipage} \\\\
\\midrule\\noalign{}
\\endhead
\\bottomrule\\noalign{}
\\endlastfoot
Nash benchmark            & 60.00 & 0.00 & \\S{}2  & Theoretical floor: rational, one-shot. \\\\
Single-agent RL           & 62.46 & 0.12 & \\S{}13 & One learner against 2 rational best-responders --- partial collusion. \\\\
2-agent RL                & 65.10 & 0.25 & \\S{}14 & Two learners + 1 myopic rival --- collusion grows. \\\\
3-agent RL (headline)     & 77.87 & 0.89 & \\S{}14 & All three learn, no rational anchor --- near-cartel. \\\\
Cartel benchmark          & 80.00 & 1.00 & \\S{}8  & Theoretical ceiling: full cooperation. \\\\
\\end{longtable}

\\subsection{The answer to the central question}
\\textbf{Yes --- but only when rationality is completely removed.}
\\begin{itemize}
\\item Single-agent RL barely improves on Nash (collusion index 0.12). With rational rivals
acting as competitive anchors, learning cannot escape the static equilibrium.
\\item Removing one rational anchor (2-agent RL) doubles the collusion index to 0.25
but the system still drifts close to Nash.
\\item When \\emph{all} three players learn (3-agent RL), the collusion index jumps to
0.89 --- the system spontaneously discovers near-cartel pricing, purely from the price signal.
\\end{itemize}

\\subsection{Why does AI converge to the cartel?}
The Q-learners do not know the demand function, the rivals' costs, or the discount factor.
Yet they replicate three theoretical results in succession:
\\begin{enumerate}
\\def\\labelenumi{\\arabic{enumi}.}
\\item \\textbf{Folk Theorem (\\S{}8) --- empirically.} With $\\delta_{\\text{RL}}=\\gamma=0.95\\gg\\delta^{*}\\approx 0.53$,
the theory predicts cooperation is sustainable. The 3-agent RL system delivers exactly that.
\\item \\textbf{Green-Porter monitoring (\\S{}11) --- without being told.} Agents observe only the
price (not rivals' quantities), exactly the Green-Porter information structure. Punishment cycles
emerge spontaneously in the learned dynamics.
\\item \\textbf{Evolutionary Punish (\\S{}12) --- as an attractor.} The ESS analysis showed that
conditional punishment is the only stable strategy in the C/D/P game. The MARL agents implicitly
implement that conditional punishment through their Q-tables.
\\end{enumerate}

\\subsection{Why does single-agent RL \\emph{not} converge to the cartel?}
The single-agent setup of \\S{}13 is structurally biased toward Nash: the two rational
best-responders \\emph{cannot} cooperate (they have no internal state, no memory of past play).
This rules out reciprocal cooperation by construction. The result is a useful sanity check
--- learning works --- but it is not the right test of Folk-Theorem convergence.

\\subsection{Robustness}
The 3-agent collusion index of $0.89$ is averaged across 10 random seeds (\\S{}14.4),
with $\\pm 0.23$ standard deviation. Even at the lowest seed the system stays well above Nash.
Reducing the number of training episodes, varying the learning rate, or perturbing the action
grid changes the magnitude but not the direction of the result.

\\subsection{Robustness extensions of Part~I}
Part~I's robustness checks (Bertrand, capacity, welfare) do not break the convergence story.
The Bertrand specification (\\S{}6) yields a different absolute price level but preserves the
cooperation premium that the RL agents discover; capacity constraints (\\S{}4) only \\emph{lower}
the critical discount factor (cooperation \\emph{easier} to sustain); welfare analysis (\\S{}5)
quantifies the consumer cost of the collusive outcome that the learners produce.

\\subsection{Limitations \\& open questions}
\\begin{itemize}
\\item State space is coarse (price-bin $\\times$ own-quantity-bin); a finer discretisation might
shift the speed but not the limit.
\\item No exogenous demand shocks during training; introducing them is the natural follow-up
(does the punishment cycle of \\S{}11 emerge in the learned regime?).
\\item Communication, side-payments and asymmetric discount factors are unmodelled.
\\end{itemize}
"""
s_synthesis = (NEW_SYNTHESIS_TITLE, NEW_SYNTHESIS)

# Pruned Policy section: drop N-Player and Empirical bullets
NEW_POLICY_TITLE = "\\section{Policy Implications}"
NEW_POLICY = """\\subsection{For OPEC member states}
\\begin{itemize}
\\item Maintaining \\textbf{quantity leadership} (Stackelberg, \\S{}3) is strategically optimal.
\\item The Folk-Theorem result ($\\delta^{*}\\approx 0.53$, \\S{}8) implies cartel discipline is enforceable
whenever members plan more than $\\sim$2 quarters ahead.
\\item The correlated-equilibrium framework (\\S{}9) \\textbf{formalises the Secretariat's coordination
role}: OPEC's announced quotas can be reinterpreted as a max-welfare CE distribution, a theoretical
justification that does not require binding contracts.
\\item \\textbf{Capacity is strategically valuable even if unused} (\\S{}4) --- higher capacity lowers
$\\delta^{*}$ by making the punishment threat more severe.
\\end{itemize}

\\subsection{For Russia (OPEC+)}
\\begin{itemize}
\\item Russia's Shapley value (\\S{}10) justifies its inclusion in OPEC+: it brings measurable marginal
value to the coalition.
\\item As Stackelberg follower, Russia's optimal play is to best-respond to OPEC's quota.
\\item \\textbf{The 3-agent RL result (\\S{}14) shows Russia's adaptive behaviour is a \\emph{prerequisite}
for OPEC+ to function as a tacit cartel.} When Russia is a passive myopic player (single- and
2-agent baselines), collusion is partial; when Russia also learns, the system reaches near-cartel
pricing.
\\end{itemize}

\\subsection{For competing producers (US shale)}
\\begin{itemize}
\\item US shale ($c=45$ \\$/bbl) is the \\textbf{most exposed} to price wars and demand collapses
(\\S{}11).
\\item Both Cournot (\\S{}2) and Bertrand (\\S{}6) confirm US is the squeezed player.
\\end{itemize}

\\subsection{For regulators}
\\begin{itemize}
\\item HHI $>$ 2{,}500 in all structures (\\S{}3) confirms \\textbf{structural high concentration}.
\\item The welfare gap competition$\\to$cartel of $\\sim$2{,}363 units (\\S{}5) quantifies the social
value of antitrust enforcement.
\\item \\textbf{Algorithmic collusion via RL} (\\S{}14) is a credible threat: 3 independent Q-learners
observing only the market price reach a collusion index of $\\sim$0.89 --- without communication.
This generalises beyond oil to any oligopoly with adaptive pricing algorithms (airlines, e-commerce,
financial markets). Regulators should focus on the \\emph{information available} to the algorithms
(price signals, posted offers) rather than on the algorithms themselves.
\\end{itemize}

\\subsection{For climate policy}
\\begin{itemize}
\\item The \\textbf{``green-cartel'' effect} (\\S{}5) --- collusion restricts output, which coincidentally
reduces emissions --- is a co-benefit rarely highlighted, but the welfare cost is borne by
\\emph{consumers}.
\\item A \\textbf{carbon tax is more efficient}: it shrinks the collusion premium \\emph{and} funds
redistribution toward the bearers of the welfare loss.
\\end{itemize}
"""
s_policy = (NEW_POLICY_TITLE, NEW_POLICY)

# ---------------------------------------------------------------------------
# Step 9. Assemble new body
# ---------------------------------------------------------------------------
def render(header: str, content: str) -> str:
    return f"{header}\n{content}"

new_body = (
    "\\clearpage\n% ============================================================================\n"
    + f"{PART1_NEW[0]}\n{PART1_NEW[1]}"
    + render(*s_model_overview)
    + render(*s_cournot)
    + render(*s_market_power)
    + render(*s_capacity)
    + render(*s_welfare)
    + render(*s_bertrand)
    + "\\clearpage\n% ============================================================================\n"
    + f"{PART2_NEW[0]}\n{PART2_NEW[1]}"
    + render(*s_repeated)
    + render(*s_folk)
    + render(*s_correlated)
    + render(*s_shapley)
    + render(*s_stochastic)
    + render(*s_evolutionary)
    + "\\clearpage\n% ============================================================================\n"
    + f"{PART3_NEW[0]}\n{PART3_NEW[1]}"
    + render(*s_single_rl)
    + render(*s_marl)
    + "\\clearpage\n% ============================================================================\n"
    + f"{PART4_NEW[0]}\n{PART4_NEW[1]}"
    + render(*s_synthesis)
    + render(*s_policy)
)

# ---------------------------------------------------------------------------
# Step 10. Prune Figures section (remove n_player_* and empirical_* figures)
# ---------------------------------------------------------------------------
FIG_RE = re.compile(
    r"\\begin\{figure\}\[H\][\s\S]*?\\end\{figure\}\n?",
)
def keep_fig(m: re.Match) -> str:
    block = m.group(0)
    drop_keywords = (
        "n_player_price_quantity", "n_player_cooperation", "n_player_opec_power",
        "empirical_price_wars", "empirical_mechanism_match", "empirical_model_vs_history",
    )
    for kw in drop_keywords:
        if kw in block:
            return ""
    return block
new_figures_appendix = FIG_RE.sub(keep_fig, figures_appendix)

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
final = preamble + new_body + new_figures_appendix
REPORT_PATH.write_text(final, encoding="utf-8")

# Sanity-check: list resulting sections
print("---- New section order ----")
for line in final.splitlines():
    if line.startswith("\\part*") or line.startswith("\\section{"):
        print(line)
print()
print(f"Total lines: {len(final.splitlines())}")
