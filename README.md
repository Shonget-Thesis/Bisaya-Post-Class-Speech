# Bisaya Post-Class Speech and Affect

This repository contains a 40-observation first-year group dataset and a reproducible exploratory speech-affect analysis for the CS412 research mini-project.

## Scope

- Inputs: pseudonymized WAV recordings and acoustic/prosodic features.
- Reference outcome: independently self-reported valence.
- Primary modeling target: `Pleasant` for valence ratings 4-5 and `Not Pleasant` for ratings 1-3.
- Context variables such as spoken word, year level, class activity, and session are excluded from model predictors.
- Results are preliminary. This group dataset contains only first-year morning observations and cannot estimate year-level or session effects.

The binary target is an implementation decision for this small dataset. It does not imply that neutral and unpleasant experiences are the same emotional state.

## Files

- `Instructions.md` — supplied research protocol.
- `bisaya_post_class_speech_metadata.csv` — pseudonymized metadata and self-reports.
- `Audios/` — WAV recordings referenced by the metadata.
- `bisaya_affect_recognition.ipynb` — interactive EDA and acoustic feature-extraction notebook. Its modeling section is retained as exploratory work.
- `bisaya_extracted_features.csv` — generated acoustic-feature artifact.
- `audio_outputs/` — packaged PNG figures, CSV result tables, and an HTML audio-analysis report.
- `export_audio_outputs.py` — authoritative grouped model evaluation; regenerates the complete `audio_outputs/` package.
- `DATA_QUALITY_REPORT.md` — dataset checks and known limitations.
- `PRIVACY_AND_ETHICS.md` — handling requirements for participant recordings.
- `requirements.txt` — Python dependencies.

## Reproduce the analysis

Python 3.12 is recommended. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 bisaya_affect_recognition.ipynb
```

Run the notebook from beginning to end to regenerate the acoustic feature CSV, then generate the submission-ready tables, figures, and corrected grouped evaluation:

```powershell
.\.venv\Scripts\python export_audio_outputs.py
```

Use the model results in `audio_outputs/`, not the notebook's older exploratory modeling cells. Do not report model numbers from an old feature CSV or from an unexecuted analysis.

### Metadata mapping

All 40 `Audio Filename` values match files in `Audios/` exactly. The `Spoken Word` column contains the researcher's updated analysis labels and is intentionally independent of the historical wording embedded in some filenames. The filename is used only to locate audio; lexical analyses use `Spoken Word`.

## Evaluation design

The authoritative evaluation in `export_audio_outputs.py` uses five-fold `StratifiedGroupKFold` validation grouped by participant code. Scaling occurs within each fold through model pipelines. Candidate models are compared against a majority-class dummy baseline using macro F1, balanced accuracy, accuracy, precision, recall, fold variability, and a confusion matrix.

Because model selection and evaluation use the same 40 observations, the reported winner is exploratory rather than an independent confirmation. A larger consolidated dataset or untouched external test set is required for stronger claims.

## Privacy warning

Voice and video recordings can identify participants. Confirm institutional approval and consent before sharing this repository or its history. The supplied protocol describes audio collection, not video collection; `Videos/` is therefore ignored by Git pending explicit approval. See `PRIVACY_AND_ETHICS.md`.
