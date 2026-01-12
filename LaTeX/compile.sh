#!/bin/bash
# LaTeX compilation script for thesis

cd "$(dirname "$0")"

echo "Compiling thesis..."

# First pass
pdflatex -interaction=nonstopmode main.tex

# Run biber for bibliography
biber main

# Second pass (for references)
pdflatex -interaction=nonstopmode main.tex

# Third pass (for cross-references)
pdflatex -interaction=nonstopmode main.tex

echo "Compilation complete! Check main.pdf"
echo ""
echo "Note: You may see warnings about missing citations - this is normal if you haven't filled in all references yet."
