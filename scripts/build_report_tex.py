"""Build outputs/report.tex from outputs/report.md via pandoc + post-processing.

Strategy:
  1. Strip the markdown title + Table-of-Contents block (we have our own
     LaTeX titlepage and \\tableofcontents).
  2. Run pandoc with --top-level-division=chapter so:
       #  PART X        -> \\chapter{}      (we'll rewrite -> \\part*{})
       ## N. Heading    -> \\section{N. ...}
       ### N.M Heading  -> \\subsection{...}
  3. Post-process pandoc output:
       a) drop the auto-generated \\label{...} after every heading
       b) \\chapter{PART X --- ...}  -> \\part*{...} + \\addcontentsline{toc}{part}{...}
       c) \\section{N. Title}        -> \\section{Title}            (drop the leading number)
       d) \\subsection{N.M Title}    -> \\subsection{Title}         (drop the leading number)
       e) shrink longtable widths into something more book-like
  4. Wrap with the existing preamble (lines 1-60 of original report.tex)
     followed by an explicit \\clearpage between parts, and \\end{document}.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "outputs" / "report.md"
OUT = ROOT / "outputs" / "report.tex"

# ── Self-contained preamble (so re-running the script never depends on the
# previous output's preamble being intact).
PREAMBLE = r"""\documentclass[12pt,a4paper]{article}

% ── Packages ────────────────────────────────────────────────────────────────
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{geometry}
\geometry{a4paper, margin=2.5cm}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    urlcolor=blue!60!black,
    citecolor=blue!60!black,
    pdftitle={Game Theory in Oil Producing Countries},
    pdfauthor={Simulation Pipeline}
}
\usepackage{xcolor}
\usepackage{parskip}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{float}
\usepackage{microtype}

% ── Section formatting ───────────────────────────────────────────────────────
\titleformat{\section}{\large\bfseries}{}{0em}{\thesection\quad}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{\thesubsection\quad}

% ── Custom column type ───────────────────────────────────────────────────────
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

% ────────────────────────────────────────────────────────────────────────────
\begin{document}

% ── Title page ───────────────────────────────────────────────────────────────
\begin{titlepage}
    \centering
    \vspace*{3cm}
    {\LARGE\bfseries Game Theory in Oil Producing Countries\par}
    \vspace{0.5cm}
    {\large Quantitative Model Report\par}
    \vspace{1cm}
    \rule{\textwidth}{0.4pt}
    \vspace{0.5cm}

    \vfill
\end{titlepage}

% ── Table of contents ────────────────────────────────────────────────────────
\tableofcontents
\newpage
"""

# ── 1. Slice the body of report.md (from "# PART I" onwards) ───────────────
md = MD.read_text(encoding="utf-8")
m = re.search(r"^# PART I", md, flags=re.MULTILINE)
if not m:
    sys.exit("could not find '# PART I' in report.md")
body_md = md[m.start():]

# Drop the trailing appendix line (we render our own appendix below).
# Actually keep it; it's small and useful.

tmp_md = Path("/tmp/_report_body.md")
tmp_md.write_text(body_md, encoding="utf-8")

# ── 2. Convert with pandoc ────────────────────────────────────────────────
tmp_tex = Path("/tmp/_report_body.tex")
subprocess.run(
    [
        "pandoc",
        str(tmp_md),
        "-t", "latex",
        "-o", str(tmp_tex),
        "--top-level-division=chapter",
        "--wrap=preserve",
    ],
    check=True,
)
body = tmp_tex.read_text(encoding="utf-8")

# ── 3. Post-process pandoc output ─────────────────────────────────────────

# 3a. Drop auto-generated \label{...} that pandoc inserts on every heading.
body = re.sub(r"\\label\{[^}]*\}\n?", "", body)

# 3b. Convert \chapter{PART X --- ...} -> \part*{...} + \addcontentsline.
def chapter_to_part(m: re.Match) -> str:
    title = m.group(1).strip()
    # collapse internal whitespace (pandoc may break long titles across lines)
    title = re.sub(r"\s+", " ", title)
    return (
        "\\clearpage\n"
        "% ════════════════════════════════════════════════════════════════════════════\n"
        f"\\part*{{{title}}}\n"
        f"\\addcontentsline{{toc}}{{part}}{{{title}}}\n"
        "% ════════════════════════════════════════════════════════════════════════════"
    )

body = re.sub(
    r"\\chapter\{([^}]+(?:\n[^}]+)*)\}",
    chapter_to_part,
    body,
)

# 3c. Strip the leading "N." or "N.M." numbering from \section{} and \subsection{}.
def strip_leading_num(m: re.Match) -> str:
    cmd = m.group(1)
    title = re.sub(r"\s+", " ", m.group(2).strip())
    title_clean = re.sub(r"^\d+(?:\.\d+)?\.?\s+", "", title)
    return f"\\{cmd}{{{title_clean}}}"

body = re.sub(
    r"\\(section|subsection|subsubsection)\{([^}]+(?:\n[^}]+)*)\}",
    strip_leading_num,
    body,
)

# 3d. Pandoc's longtables look ugly (giant minipages, no \caption).  Tighten:
#     remove the "(\linewidth - N\tabcolsep) * \real{...}" widths and let
#     LaTeX size columns automatically.  Convert \begin{longtable}[]{@{} ... @{}}
#     to use simple p{} columns with a fixed total width.
def tighten_longtable_widths(text: str) -> str:
    # Replace the verbose column spec with auto-sized centered columns.
    pattern = re.compile(
        r">\{\\raggedright\\arraybackslash\}p\{\(\\linewidth - \d+\\tabcolsep\)\s*\*\s*\\real\{([0-9.]+)\}\}",
        re.MULTILINE,
    )
    return pattern.sub(r">{\\raggedright\\arraybackslash}p{\1\\linewidth}", text)

body = tighten_longtable_widths(body)

# 3e. The first \begin{longtable} after a section comes from "{\def\LTcaptype{none}"
#     — keep it but ensure tables fit on the page width by using parboxes lower.

# 3f. Convert blockquotes "> Transition →" produced by pandoc.  Pandoc converts
#     blockquotes into \begin{quote}...\end{quote}.  We want them as
#     \medskip + \noindent\textit{...}.  Detect single-paragraph quotes and
#     convert; multi-paragraph stays.
def convert_quote_block(m: re.Match) -> str:
    inner = m.group(1).strip()
    # If the inner already has a paragraph break, leave as quote.
    if "\n\n" in inner:
        return m.group(0)
    inner_one_line = re.sub(r"\s+", " ", inner)
    return (
        "\n\\medskip\n"
        f"\\noindent\\textit{{{inner_one_line}}}\n"
    )

body = re.sub(
    r"\\begin\{quote\}\n([\s\S]*?)\n\\end\{quote\}",
    convert_quote_block,
    body,
)

# 3g. Pandoc emits \pandocbounded{...} / \begin{Shaded}... that we want simpler.
#     Strip Shaded blocks since we have no syntax-highlighted code in the report.
body = re.sub(r"\\begin\{Shaded\}[\s\S]*?\\end\{Shaded\}", "", body)

# 3h. Drop the horizontal rules pandoc produces from "---" (we use \clearpage / \medskip).
body = re.sub(
    r"\n\\begin\{center\}\\rule\{0\.5\\linewidth\}\{0\.5pt\}\\end\{center\}\n",
    "\n",
    body,
)

# 3i. Drop trailing "\#" escapes from things like "δ\\* (data: `csv.csv`)"
#     (pandoc preserves backslash-asterisk).
body = body.replace("\\\\*", "*")

# 3j. Pandoc wraps longtables with "{\\def\\LTcaptype{none} ... }" to suppress
#     a phantom caption counter.  This relies on the "none" counter existing,
#     which it does NOT in our preamble.  Strip the wrapper.
body = body.replace("{\\def\\LTcaptype{none} % do not increment counter\n", "")
body = re.sub(r"\\end\{longtable\}\n\}", r"\\end{longtable}", body)

# 3k. Replace problematic Unicode characters with LaTeX equivalents.  Our
#     preamble only enables [utf8]{inputenc}; many maths symbols still trip
#     pdflatex.  This is a comprehensive scrub.
unicode_map = {
    # ── Math operators ────────────────────────────────────────────────
    "−": "$-$",   # U+2212 minus sign
    "×": "$\\times$",
    "÷": "$\\div$",
    "·": "$\\cdot$",
    "⋅": "$\\cdot$",
    "≈": "$\\approx$",
    "≠": "$\\neq$",
    "≤": "$\\le$",
    "≥": "$\\ge$",
    "±": "$\\pm$",
    "≡": "$\\equiv$",
    "∈": "$\\in$",
    "∉": "$\\notin$",
    "∑": "$\\sum$",
    "∏": "$\\prod$",
    "∂": "$\\partial$",
    "∇": "$\\nabla$",
    "∞": "$\\infty$",
    "√": "$\\sqrt{\\,}$",
    "∫": "$\\int$",
    # ── Arrows ─────────────────────────────────────────────────────────
    "→": "$\\to$",
    "←": "$\\leftarrow$",
    "↔": "$\\leftrightarrow$",
    "⇒": "$\\Rightarrow$",
    "⇐": "$\\Leftarrow$",
    "⇔": "$\\Leftrightarrow$",
    # ── Greek letters (lowercase) ──────────────────────────────────────
    "α": "$\\alpha$",
    "β": "$\\beta$",
    "γ": "$\\gamma$",
    "δ": "$\\delta$",
    "ε": "$\\varepsilon$",
    "ζ": "$\\zeta$",
    "η": "$\\eta$",
    "θ": "$\\theta$",
    "κ": "$\\kappa$",
    "λ": "$\\lambda$",
    "μ": "$\\mu$",
    "ν": "$\\nu$",
    "π": "$\\pi$",
    "ρ": "$\\rho$",
    "σ": "$\\sigma$",
    "τ": "$\\tau$",
    "φ": "$\\varphi$",
    "ω": "$\\omega$",
    "χ": "$\\chi$",
    # ── Greek letters (uppercase) ──────────────────────────────────────
    "Δ": "$\\Delta$",
    "Σ": "$\\Sigma$",
    "Π": "$\\Pi$",
    "Ω": "$\\Omega$",
    "Λ": "$\\Lambda$",
    "Φ": "$\\Phi$",
    "Θ": "$\\Theta$",
    # ── Punctuation / quotes ───────────────────────────────────────────
    "…": "\\ldots ",
    "–": "--",      # en-dash
    "—": "---",     # em-dash (already used)
    "‐": "-",       # hyphen (U+2010)
    "‑": "-",       # non-breaking hyphen
    "‘": "`",
    "’": "'",
    "“": "``",
    "”": "''",
    "«": "``",
    "»": "''",
    "•": "$\\bullet$",
    "ø": "\\o{}",
    "Ø": "\\O{}",
    # ── Superscripts / subscripts (best-effort) ───────────────────────
    "²": "$^{2}$",
    "³": "$^{3}$",
    "⁰": "$^{0}$",
    "¹": "$^{1}$",
    "⁴": "$^{4}$",
    "⁵": "$^{5}$",
    "⁻": "$^{-}$",
    "₀": "$_{0}$",
    "₁": "$_{1}$",
    "₂": "$_{2}$",
    "₃": "$_{3}$",
    # ── Misc ───────────────────────────────────────────────────────────
    "✓": "$\\checkmark$",
    "✗": "$\\times$",
    "§": "\\S{}",
    "∗": "$*$",
    "❌": "$\\times$",
    "❌": "$\\times$",
    " ": " ",   # NBSP -> space
    "═": "=",   # box-drawing
    " ": " ",   # narrow no-break space
    "‒": "--",
}
for src, dst in unicode_map.items():
    body = body.replace(src, dst)

# Combining diacritics: e.g. "p\u0304" (p with combining macron) → "$\\bar{p}$".
body = re.sub(r"([A-Za-z])\u0304", r"$\\bar{\1}$", body)
body = re.sub(r"([A-Za-z])\u0301", r"$\\acute{\1}$", body)
body = re.sub(r"([A-Za-z])\u0300", r"$\\grave{\1}$", body)
body = re.sub(r"([A-Za-z])\u0302", r"$\\hat{\1}$", body)
body = re.sub(r"([A-Za-z])\u0303", r"$\\tilde{\1}$", body)
body = re.sub(r"([A-Za-z])\u0307", r"$\\dot{\1}$", body)

# 3l. Remove pandoc-only macros that aren't in our preamble.
body = body.replace("\\tightlist\n", "")

# 3l'.  Pandoc emits images in two ways:
#         (A) image alone in its markdown paragraph -> a proper LaTeX figure
#             block with "\pandocbounded{\includegraphics[...,alt={X}]{Y}}"
#             *inside* the figure;
#         (B) one of several images in the same markdown paragraph (no blank
#             line between) -> bare inline "\pandocbounded{\includegraphics{...}}".
#       To send every image to the Figures appendix we normalise both forms
#       to a single canonical figure block carrying its caption.

# Step A: normalise the existing figure blocks to remove the \pandocbounded
# wrapper while preserving filename + caption + width.
def _sanitise_pattern_a(m: re.Match) -> str:
    inner = m.group(1)
    file_match = re.search(r"\\includegraphics\[[^\]]*\]\{([^{}]+)\}", inner)
    cap_match = re.search(r"\\caption\{([^}]*)\}", inner)
    if not file_match:
        return m.group(0)
    fname = file_match.group(1)
    caption = cap_match.group(1) if cap_match else ""
    return (
        "\\begin{figure}\n"
        "\\centering\n"
        f"\\includegraphics[width=0.85\\linewidth,keepaspectratio]{{{fname}}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}"
    )

body = re.sub(
    r"\\begin\{figure\}\n([\s\S]*?)\n\\end\{figure\}",
    _sanitise_pattern_a,
    body,
)

# Step B: now convert any remaining bare inline pandocbounded-image
# (Pattern B) into a proper figure block too.
def _inline_image_to_figure(m: re.Match) -> str:
    inner = m.group(1)
    alt_match = re.search(r"alt=\{([^}]*)\}", inner)
    file_match = re.search(r"\{([^{}]+\.(?:png|pdf|jpg|jpeg))\}", inner)
    if not file_match:
        return m.group(0)
    caption = alt_match.group(1) if alt_match else ""
    fname = file_match.group(1)
    return (
        "\n\\begin{figure}\n"
        "\\centering\n"
        f"\\includegraphics[width=0.85\\linewidth,keepaspectratio]{{{fname}}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}\n"
    )

body = re.sub(
    r"\\pandocbounded\{(\\includegraphics\[[^\]]*\]\{[^{}]+\})\}",
    _inline_image_to_figure,
    body,
)

# Drop any remaining \pandocbounded wrappers around non-image content.
body = re.sub(r"\\pandocbounded\{([^}]*)\}", r"\1", body)

# pandoc emits "\textbackslash" inside table cells from "\\* in markdown.
# Convert "$\textbackslash$ ..." to a literal asterisk.
body = body.replace("$\\textbackslash$*", "*")

# 3m'. Convert "\section{Appendix: Output Files}" to an unnumbered section
#      so the appendix isn't picked up as section 19.
body = body.replace(
    "\\section{Appendix: Output Files}",
    "\\section*{Appendix: Output Files}\n\\addcontentsline{toc}{section}{Appendix: Output Files}",
)

# 3m. Pandoc 3.9 emits a malformed \\includegraphics[keepaspectratio,alt={X]{Y}}
#     -- strip the alt= option entirely and use a sane width.
body = re.sub(
    r"\\includegraphics\[keepaspectratio,alt=\{[^]]*\]\{([^}]+)\}\}",
    r"\\includegraphics[width=0.85\\linewidth,keepaspectratio]{\1}",
    body,
)
# Same for any other malformed/correct alt= pattern.
body = re.sub(
    r"\\includegraphics\[([^\]]*),alt=\{([^}]*)\}([^\]]*)\]",
    r"\\includegraphics[\1\3]",
    body,
)

# ── 4. Move every \begin{figure}...\end{figure} block to a Figures appendix.
#     Each in-body figure is replaced by an italic cross-reference; runs of
#     consecutive figures are collapsed into a single "See Figs.~..." line.

FIG_BLOCK = re.compile(
    r"\\begin\{figure\}\n(?P<inner>[\s\S]*?)\n\\end\{figure\}",
    re.MULTILINE,
)

# All collected figures, in document order, as fully-formed LaTeX blocks
# (with a stable \label{fig:<basename>} injected just before \end{figure}).
collected_figures: list[str] = []
seen_labels: set[str] = set()


def _basename_label(inner: str) -> str:
    """Derive a unique fig label from the first \\includegraphics filename."""
    m = re.search(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", inner)
    if not m:
        return f"figanon{len(collected_figures)}"
    base = Path(m.group(1)).stem
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    label = base or f"figanon{len(collected_figures)}"
    # Guarantee uniqueness across the document.
    candidate = label
    i = 2
    while candidate in seen_labels:
        candidate = f"{label}_{i}"
        i += 1
    seen_labels.add(candidate)
    return candidate


def _harvest_figure(match: re.Match) -> tuple[str, str]:
    """Return (label, full_figure_block) for a single \\begin{figure} match."""
    inner = match.group("inner").rstrip()
    label = _basename_label(inner)
    # Inject \label{...} immediately before the closing brace of \caption{...}
    # if a caption exists; otherwise place a bare \label after the inner block.
    if "\\caption{" in inner:
        # Add \label inside the caption's braces (LaTeX scopes labels by counter).
        # Match the full \caption{...} including possibly nested braces.
        def _inject_label(c: re.Match) -> str:
            return c.group(0).rstrip("}") + f"\\label{{fig:{label}}}}}"
        inner_with_label = re.sub(
            r"\\caption\{(?:[^{}]|\{[^}]*\})*\}",
            _inject_label,
            inner,
            count=1,
        )
    else:
        inner_with_label = inner + f"\n  \\label{{fig:{label}}}"
    block = "\\begin{figure}[H]\n" + inner_with_label + "\n\\end{figure}"
    return label, block


# Walk the body and replace each *run* of figure blocks (figures separated
# only by whitespace) with a single italicised cross-reference paragraph.
RUN = re.compile(
    r"(?:\\begin\{figure\}\n[\s\S]*?\n\\end\{figure\}\s*)+",
    re.MULTILINE,
)


def _replace_run(run_match: re.Match) -> str:
    run_text = run_match.group(0)
    labels: list[str] = []
    for fig_match in FIG_BLOCK.finditer(run_text):
        label, block = _harvest_figure(fig_match)
        labels.append(label)
        collected_figures.append(block)
    if len(labels) == 1:
        ref = f"Fig.~\\ref{{fig:{labels[0]}}}"
    elif len(labels) == 2:
        ref = f"Figs.~\\ref{{fig:{labels[0]}}} and~\\ref{{fig:{labels[1]}}}"
    else:
        joined = ", ".join(f"\\ref{{fig:{l}}}" for l in labels[:-1])
        ref = f"Figs.~{joined} and~\\ref{{fig:{labels[-1]}}}"
    return f"\n\\medskip\n\\noindent\\emph{{See {ref}.}}\n\n"


body = RUN.sub(_replace_run, body)

# Build the Figures appendix.  Group by part headers if any structure can be
# inferred from the surrounding section; we keep the simpler flat layout
# because cross-references already disambiguate.
figures_appendix = [
    "\\clearpage",
    "% ════════════════════════════════════════════════════════════════════════════",
    "\\section*{Figures}",
    "\\addcontentsline{toc}{section}{Figures}",
    "% ════════════════════════════════════════════════════════════════════════════",
    "",
    "\\noindent\\textit{All figures referenced from the body are collected here in",
    "the order in which they are first cited.}",
    "",
]
figures_appendix.extend(collected_figures)
figures_block = "\n".join(figures_appendix)

# Splice the figures appendix BEFORE the existing "Output Files" appendix
# (so the textual output-files appendix sits at the very end).
APP_MARK = "\\section*{Appendix: Output Files}"
if APP_MARK in body:
    body = body.replace(APP_MARK, figures_block + "\n\n" + APP_MARK)
else:
    body = body.rstrip() + "\n\n" + figures_block + "\n"

print(f"collected {len(collected_figures)} figures into the Figures appendix")

# ── 5. Compose the final document ─────────────────────────────────────────
final = PREAMBLE + "\n" + body.strip() + "\n\n\\end{document}\n"
OUT.write_text(final, encoding="utf-8")
print(f"wrote {OUT}: {len(final.splitlines())} lines")
