# Audio Analysis Outputs

This folder contains presentation-ready figures, machine-readable result tables, and reports generated from the audio dataset.

## Contents

- `figures/` — PNG plots for exploratory analysis, acoustic comparisons, lexical control, and model evaluation.
- `tables/` — validated metadata, acoustic features, descriptive summaries, model metrics, confusion matrix, classification report, and feature importance.
- `reports/` — a text summary and a browsable HTML report generated from the corrected output tables.
- `manifest.csv` — index of generated deliverables.

Regenerate the package from the project root:

```powershell
.\.venv\Scripts\python export_audio_outputs.py
```

The results are exploratory because the dataset contains only 40 first-year morning observations and has no independent external test set.
