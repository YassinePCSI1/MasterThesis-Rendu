#!/usr/bin/env python3
"""Convert thesis_part_I_and_II.txt to LaTeX (Sections 2 and 3)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "thesis_part_I_and_II.txt"
OUT_FOUNDATIONS = ROOT / "thesis" / "part_I_foundations.tex"
OUT_DYNAMICS = ROOT / "thesis" / "part_II_dynamics.tex"
OUT_BACKMATTER = ROOT / "thesis" / "backmatter_tables.tex"

# Map old section numbers to new LaTeX labels
SECTION_MAP = {
    "3.1": ("2", "Model overview and calibration", "subsec:model"),
    "3.2": ("2", "Static Cournot equilibrium", "subsec:cournot"),
    "3.3": ("2", "Market power and first-mover advantage", "subsec:marketpower"),
    "3.4": ("2", "Capacity constraints", "subsec:capacity"),
    "3.5": ("2", "Welfare and deadweight loss", "subsec:welfare"),
    "3.6": ("2", "Bertrand price competition (robustness check)", "subsec:bertrand"),
    "3.7": ("3", "From static to dynamic: repeated game dynamics", "subsec:repeated"),
    "3.8": ("3", "Cartel, punishment, and the Folk Theorem", "subsec:folk"),
    "3.9": ("3", "Correlated equilibrium", "subsec:ce"),
    "3.10": ("3", "Coalition formation and Shapley values", "subsec:shapley"),
    "3.11": ("3", "Stochastic demand and Green and Porter monitoring", "subsec:greenporter"),
    "3.12": ("3", "Evolutionary game theory", "subsec:evo"),
}

SKIP_PREFIXES = (
    "APPLICATION OF GAME THEORY",
    "PART I AND PART II",
    "Authors:",
    "Institution:",
    "Supervisor section:",
    "Date: May",
    "Formatting target",
    "Citations follow the Harvard",
)


def math_segments(text: str) -> list[tuple[str, bool]]:
    """Split *text* into (segment, is_outside_math) pairs."""
    segments: list[tuple[str, bool]] = []
    buf: list[str] = []
    outside = True
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "$" and (i == 0 or text[i - 1] != "\\"):
            segments.append(("".join(buf), outside))
            buf = []
            outside = not outside
            i += 1
            continue
        buf.append(ch)
        i += 1
    segments.append(("".join(buf), outside))
    return segments


def join_math_segments(segments: list[tuple[str, bool]]) -> str:
    out: list[str] = []
    for chunk, outside in segments:
        if not outside:
            out.append(f"${chunk}$")
        else:
            out.append(chunk)
    return "".join(out)


def sub_outside_math(text: str, pattern: str, repl) -> str:
    """Apply regex substitution only outside $...$ math regions."""
    if not callable(repl):
        repl_str = repl
        repl = lambda _m, r=repl_str: r
    segments = math_segments(text)
    updated: list[tuple[str, bool]] = []
    for chunk, outside in segments:
        if outside:
            chunk = re.sub(pattern, repl, chunk)
        updated.append((chunk, outside))
    return join_math_segments(updated)


def preprocess_phrases(text: str) -> str:
    """Replace full expressions before token-level math wrapping."""
    phrases = [
        ("HHI = sum_i s_i^2 x 10,000", r"$\mathrm{HHI}=\sum_i s_i^2 \times 10{,}000$"),
        ("L_i = (P - c_i)/P", r"$L_i=(P-c_i)/P$"),
        ("(1/2) b (Q_comp - Q)^2", r"$(1/2)b(Q_{\mathrm{comp}}-Q)^2$"),
        ("(a + c_min)/2", r"$(a+c_{\min})/2$"),
        ("P(Q) = A Q^{-1/eta}", r"$P(Q)=A Q^{-1/\eta}$"),
        ("ln P = a - b Q", r"$\ln P = a - bQ$"),
        ("$/bbl", r"\$/bbl"),
    ]
    for old, new in phrases:
        text = text.replace(old, new)
    text = re.sub(r"(?<!\\)\$(?=\d)", r"\\$", text)
    return text


def fix_subscripts(text: str) -> str:
    """Wrap common economic notation in math mode (outside existing math)."""
    reps = [
        (r"\bQ_comp\b", r"$Q_{\mathrm{comp}}$"),
        (r"\b10\^3\b", r"$10^3$"),
        (r"\b10\^2\b", r"$10^2$"),
        (r"\bv\(N\)\b", r"$v(N)$"),
        (r"\bphi_US\b", r"$\phi_{\text{US}}$"),
        (r"\bphi_OPEC\b", r"$\phi_{\text{OPEC}}$"),
        (r"\bphi_RUS\b", r"$\phi_{\text{RUS}}$"),
        (r"\bdelta_a\b", r"$\Delta a$"),
        (r"\bp_bar\b", r"$\bar{p}$"),
        (r"\bpi_sucker\b", r"$\pi^{\text{sucker}}$"),
        (r"\bpi_i\b", r"$\pi_i$"),
        (r"\bq_i\b", r"$q_i$"),
        (r"\bc_i\b", r"$c_i$"),
        (r"\bL_i\b", r"$L_i$"),
        (r"\bs_i\b", r"$s_i$"),
        (r"\bphi_i\b", r"$\phi_i$"),
        (r"\bv\(S\)", r"$v(S)$"),
        (r"\bsum_i\b", r"$\sum_i$"),
        (r"\bc_min\b", r"$c_{\min}$"),
        (r"\bc_OPEC\b", r"$c_{\text{OPEC}}$"),
        (r"\bc_US\b", r"$c_{\text{US}}$"),
        (r"\bc_RUS\b", r"$c_{\text{RUS}}$"),
        (r"\bdelta_OPEC\*", r"$\delta_{\text{OPEC}}^*$"),
        (r"\bdelta_RUS\*", r"$\delta_{\text{RUS}}^*$"),
        (r"\bdelta_US\*", r"$\delta_{\text{US}}^*$"),
        (r"\bdelta_i\*", r"$\delta_i^*$"),
        (r"\bdelta\*", r"$\delta^*$"),
        (r"\bcap_OPEC\b", r"$\mathrm{cap}_{\text{OPEC}}$"),
        (r"\bcap_US\b", r"$\mathrm{cap}_{\text{US}}$"),
        (r"\bcap_RUS\b", r"$\mathrm{cap}_{\text{RUS}}$"),
        (r"\bT_p\b", r"$T_p$"),
        (r"\bmu\b", r"$\mu$"),
        (r"\blambda\*", r"$\lambda^*$"),
        (r"\blambda\b", r"$\lambda$"),
        (r"\bsigma\b", r"$\sigma$"),
        (r"\brho\b", r"$\rho$"),
        (r"\btau\b", r"$\tau$"),
        (r"\bpi_dev\b", r"$\pi^{\text{dev}}$"),
        (r"\bpi_coop\b", r"$\pi^{\text{coop}}$"),
        (r"\bpi_nash\b", r"$\pi^{\text{nash}}$"),
        (r"\bdelta\b", r"$\delta$"),
    ]
    for pat, rep in reps:
        text = sub_outside_math(text, pat, rep)
    return text


SUBSEC_RE = re.compile(r"^(\d+\.\d+)\s+(.+)$")
EQ_LINE_RE = re.compile(
    r"^\s*(.+?)\s*,?\s*\((\d+)\)\s*$"
)
PART_RE = re.compile(r"^PART (I|II)\.")
SEP_RE = re.compile(r"^=+$")


def esc(text: str) -> str:
    """Escape LaTeX special chars outside math delimiters."""
    updated: list[tuple[str, bool]] = []
    for chunk, outside in math_segments(text):
        if outside:
            for ch, rep in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
                chunk = chunk.replace(ch, rep)
        updated.append((chunk, outside))
    return join_math_segments(updated)


def convert_equation_body(body: str) -> str:
    """Convert plain-text equation to LaTeX."""
    body = body.replace(">=", r"\ge")
    body = body.replace("pi_i^coop", r"\pi_i^{\text{coop}}")
    body = body.replace("pi_i^dev", r"\pi_i^{\text{dev}}")
    body = body.replace("pi_i^nash", r"\pi_i^{\text{nash}}")
    body = re.sub(r"delta_i\*", r"\\delta_i^*", body)
    body = re.sub(r"delta\*", r"\\delta^*", body)
    body = body.replace("delta", r"\delta")
    body = body.replace("lambda", r"\lambda")
    body = body.replace("sigma", r"\sigma")
    body = body.replace("pi_i", r"\pi_i")
    body = body.replace("q_{-i}", r"q_{-i}")
    body = body.replace("BR_i", r"\mathrm{BR}_i")
    return body


def is_equation_block(lines: list[str], idx: int) -> tuple[str, int] | None:
    """Detect indented equation lines ending with (n)."""
    first = lines[idx].rstrip()
    if not first.startswith("    "):
        return None

    chunk: list[str] = []
    j = idx
    while j < len(lines):
        line = lines[j].rstrip()
        if not line.strip():
            break
        if not line.startswith("    "):
            break
        chunk.append(line.strip())
        j += 1
        m = EQ_LINE_RE.match(line.strip())
        if m:
            body, num = m.group(1).strip(), m.group(2)
            if "=" not in body:
                return None
            body = convert_equation_body(body)
            latex = (
                f"\\begin{{equation}}\n"
                f"{body}\n"
                f"\\label{{eq:{num}}}\n"
                f"\\end{{equation}}\n"
            )
            return latex, j
    return None


def flush_paragraph(buf: list[str], out: list[str]) -> None:
    if not buf:
        return
    text = " ".join(s.strip() for s in buf if s.strip())
    text = preprocess_phrases(text)
    text = fix_subscripts(text)
    text = esc(text)
    out.append(text + "\n\n")


def should_skip_line(stripped: str) -> bool:
    if any(stripped.startswith(p) for p in SKIP_PREFIXES):
        return True
    if stripped.startswith("PART I. FOUNDATIONS"):
        return True
    if stripped.startswith("PART II. STRATEGIC"):
        return True
    return False


def convert_body(lines: list[str], start: int, end: int) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    i = start
    current_part = None

    while i < end:
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped or SEP_RE.match(stripped):
            flush_paragraph(buf, out)
            buf = []
            i += 1
            continue

        if PART_RE.match(stripped):
            flush_paragraph(buf, out)
            buf = []
            current_part = stripped
            i += 1
            continue

        if should_skip_line(stripped):
            flush_paragraph(buf, out)
            buf = []
            i += 1
            continue

        if stripped.startswith("3. Model,"):
            i += 1
            continue

        m = SUBSEC_RE.match(stripped)
        if m and m.group(1) in SECTION_MAP:
            flush_paragraph(buf, out)
            buf = []
            old_num, title, label = SECTION_MAP[m.group(1)]
            new_num = old_num + "." + m.group(1).split(".")[1]
            out.append(f"\\subsection{{{title}}}\n\\label{{{label}}}\n\n")
            i += 1
            continue

        eq = is_equation_block(lines, i)
        if eq:
            flush_paragraph(buf, out)
            buf = []
            latex, i = eq
            out.append(latex)
            continue

        if stripped.startswith("NOTE ON FIGURES"):
            break

        buf.append(line)
        i += 1

    flush_paragraph(buf, out)
    return out


def convert_tables(lines: list[str]) -> list[str]:
    out = ["\\appendix\n", "\\section{Tables and figure register}\n\\label{sec:backmatter}\n\n"]
    in_table = False
    buf: list[str] = []

    def flush_table():
        nonlocal buf, in_table
        if not buf:
            return
        # Simple verbatim-style table block
        out.append("\\begin{quote}\\small\\ttfamily\n")
        out.extend(line + "\n" for line in buf)
        out.append("\\end{quote}\n\n")
        buf = []
        in_table = False

    for line in lines:
        s = line.rstrip()
        if s.strip().startswith("Table ") or s.strip().startswith("Figure "):
            flush_table()
            out.append(f"\\paragraph{{{esc(s.strip())}}}\n")
            in_table = True
            continue
        if in_table and s.strip():
            buf.append(esc(s))
        elif in_table and not s.strip():
            flush_table()
    flush_table()
    return out


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Part I: from first 3.1 to before PART II
    first_subsec = next(
        i for i, l in enumerate(lines) if l.strip().startswith("3.1 ")
    )
    p2_start = next(i for i, l in enumerate(lines) if l.strip().startswith("3.7 "))
    p1_body = convert_body(lines, first_subsec, p2_start)

    # Part II: from PART II to NOTE ON FIGURES
    note_start = next(i for i, l in enumerate(lines) if l.strip().startswith("NOTE ON FIGURES"))
    p2_body = convert_body(lines, p2_start, note_start)

    OUT_FOUNDATIONS.parent.mkdir(parents=True, exist_ok=True)

    foundations = [
        "\\section{Foundations: The Static Market Environment}\n",
        "\\label{sec:foundations}\n\n",
        "This section introduces the analytical environment used throughout "
        "Parts~I and~II of the thesis. We model the global crude oil market as a "
        "strategic oligopoly among three production blocs and escalate through "
        "successive frameworks, each relaxing one assumption of its predecessor.\n\n",
    ] + p1_body

    dynamics = [
        "\\section{Strategic Dynamics: The Theoretical Anchor}\n",
        "\\label{sec:dynamics}\n\n",
        "The static analysis treats each quarter as a one-shot game. Part~II "
        "embeds the Cournot stage game in time, under imperfect monitoring, "
        "coalitional bargaining, and evolutionary selection.\n\n",
    ] + p2_body

    OUT_FOUNDATIONS.write_text("".join(foundations), encoding="utf-8")
    OUT_DYNAMICS.write_text("".join(dynamics), encoding="utf-8")
    OUT_BACKMATTER.write_text("".join(convert_tables(lines[note_start:])), encoding="utf-8")
    print(f"Wrote {OUT_FOUNDATIONS}")
    print(f"Wrote {OUT_DYNAMICS}")
    print(f"Wrote {OUT_BACKMATTER}")


if __name__ == "__main__":
    main()
