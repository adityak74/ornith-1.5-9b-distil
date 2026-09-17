#!/bin/zsh
# Build the arXiv submission bundle: sources, generated tables/figures, and the
# .bbl (arXiv's AutoTeX does not run BibTeX). Re-run after any result lands.
set -e
cd "$(dirname "$0")/../paper"
uv run python ../scripts/make_paper_tables.py >/dev/null
uv run --with matplotlib python ../scripts/make_paper_figures.py >/dev/null
tectonic --keep-intermediates main.tex >/dev/null 2>&1
test -f main.bbl || { echo "no main.bbl produced"; exit 1; }
rm -f ../arxiv-submission.zip
zip -q ../arxiv-submission.zip main.tex main.bbl references.bib tables_*.tex fig_*.pdf
rm -f main.aux main.log main.blg main.out main.fls main.fdb_latexmk
echo "arxiv-submission.zip:"; unzip -l ../arxiv-submission.zip | awk 'NR>3 && $4!=""' | awk '{print "  "$4"  ("$1" bytes)"}'
