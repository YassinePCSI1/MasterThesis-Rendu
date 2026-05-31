# Build Rendu_final/main.pdf with correct bibliography resolution.
# Usage (from Rendu_final/):  .\build.ps1
#
# Why this script exists: when pdflatex uses -output-directory=build,
# main.aux lands in build/. If bibtex is then launched *inside* build/,
# it cannot see references.bib (which lives one level up) and emits
# "I didn't find a database entry" for every key, producing (?) marks
# in the PDF. The fix is to run bibtex from this directory (where
# references.bib lives) while pointing it at build/main.aux, and to
# prepend this directory to BIBINPUTS so the correct .bib wins.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Build = Join-Path $Root "build"

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Error "pdflatex not found. Install MiKTeX or TeX Live and add it to PATH."
}

Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $Build | Out-Null

    # Make sure bibtex resolves THIS directory's references.bib first.
    $env:BIBINPUTS = "$Root;$Build;$env:BIBINPUTS"

    Write-Host "==> pdflatex pass 1"
    pdflatex -interaction=nonstopmode -output-directory=build main.tex
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "pdflatex pass 1 returned $LASTEXITCODE (see build/main.log)"
    }

    Write-Host "==> bibtex (run from Rendu_final/, aux in build/)"
    bibtex (Join-Path "build" "main")
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "bibtex returned $LASTEXITCODE (see build/main.blg)"
    }

    Write-Host "==> pdflatex pass 2"
    pdflatex -interaction=nonstopmode -output-directory=build main.tex | Out-Null

    Write-Host "==> pdflatex pass 3"
    pdflatex -interaction=nonstopmode -output-directory=build main.tex | Out-Null

    # A fourth pass stabilises page references after the bibliography
    # (which spans several pages) is inserted on pass 2.
    Write-Host "==> pdflatex pass 4"
    pdflatex -interaction=nonstopmode -output-directory=build main.tex | Out-Null

    Copy-Item -Force (Join-Path $Build "main.pdf") (Join-Path $Root "main.pdf")
    Write-Host ""
    Write-Host "SUCCESS: $(Join-Path $Build 'main.pdf')"
    Write-Host "Copied to: $(Join-Path $Root 'main.pdf')"
}
finally {
    Pop-Location
}
