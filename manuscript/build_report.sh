#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -outdir=manuscript -interaction=nonstopmode -halt-on-error manuscript/roy_evo_spatial_rescue_report.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory manuscript manuscript/roy_evo_spatial_rescue_report.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory manuscript manuscript/roy_evo_spatial_rescue_report.tex
fi
