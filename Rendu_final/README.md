# Full thesis (LaTeX)

Compile the complete thesis (Parts I–IV) from the repository root:

```powershell
.\build_thesis.ps1
```

Outputs:

- `thesis/build/main.pdf` — build artefact
- `outputs/thesis_full.pdf` — copy for submission

## Structure

| File | Content |
|------|---------|
| `main.tex` | Root document |
| `preamble.tex` | Packages and macros |
| `frontmatter.tex` | Title page + AI abstract + TOC |
| `introduction.tex` | Section 1 |
| `part_I_foundations.tex` | Section 2 (generated) |
| `part_II_dynamics.tex` | Section 3 (generated) |
| `part_III_IV.tex` | Sections 4–6 (MARL + synthesis + policy) |
| `backmatter_tables.tex` | Figure/table register |
| `references.bib` | Bibliography |

Regenerate Parts I–II from `thesis_part_I_and_II.txt`:

```powershell
python scripts/convert_part_I_II_to_latex.py
```

Figures are loaded from `../outputs/` via `\graphicspath`.
