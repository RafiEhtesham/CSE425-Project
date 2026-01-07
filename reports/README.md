# NeurIPS-like Paper Build

## Files
- `reports/neurips_paper.tex`: Manuscript referencing figures and tables from `easy`, `medium`, `hard`.

## Figures Referenced (conditionally)
- Easy: `easy/results/vae_tsne_true.png` (included if present)
- Medium: `medium/results/hybrid_tsne_true.png`
- Hard: `hard/results/comprehensive_comparison.png`

If an image is missing, the PDF will show a framed placeholder instead.

## Build (Windows)
Assumes `pdflatex` is installed (MiKTeX/TeX Live).

```powershell
# From the repo root
pdflatex -interaction=nonstopmode -output-directory reports reports/neurips_paper.tex
pdflatex -interaction=nonstopmode -output-directory reports reports/neurips_paper.tex
```

The PDF will be at `reports/neurips_paper.pdf`.

## Optional: Regenerate tables from CSVs
You can generate LaTeX table snippets from the CSVs using `reports/make_tables.py` and manually paste them into the `Results` section if desired.
