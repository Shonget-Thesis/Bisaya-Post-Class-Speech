# Bisaya Post-Class Speech and Affect

This repository contains the group dataset, feature-extraction workflow, exploratory analysis, and generated outputs for the CS412 research mini-project on affective information in short Bisaya post-class speech.

The current dataset contains 40 pseudonymized audio observations from first-year Computer Science students after morning classes. Each recording is paired with an independently reported valence and arousal rating. The project investigates whether acoustic and prosodic characteristics of the recordings contain information related to those self-reports.

## Project status

| Modality | Status | Location |
| --- | --- | --- |
| Audio | Initial feature extraction and exploratory evaluation complete | `Audios/`, `bisaya_extracted_features.csv`, `audio_outputs/` |
| Face | Reserved; no face-feature dataset generated yet | `face/` |
| Multimodal | Reserved until participant-aligned face features are available | `multimodal/` |

The audio results are exploratory. They are not evidence of a validated affect-recognition system.

## Research scope

- Input modality: short WAV speech recordings.
- Reference outcomes: independently self-reported valence and arousal.
- Primary modeling target: `Pleasant` for valence ratings 4-5 and `Not Pleasant` for ratings 1-3.
- Acoustic predictors: 94 duration, pitch, energy, zero-crossing, spectral, MFCC, delta, and delta-delta features.
- Context fields such as spoken word, year level, class activity, and session are excluded from the model predictors.
- Evaluation: five-fold participant-grouped cross-validation with fold-local preprocessing.

The binary target is an analysis decision for this small dataset. It does not imply that neutral and unpleasant experiences are the same emotional state.

## Repository structure

```text
Bisaya-Post-Class-Speech/
|-- Audios/                              # WAV recordings referenced by metadata
|-- audio_outputs/
|   |-- figures/                         # Presentation-ready PNG figures
|   |-- tables/                          # Generated CSV datasets and result tables
|   |-- reports/                         # Text summary and browsable HTML report
|   `-- manifest.csv                     # Index of generated output files
|-- face/                                # Placeholder for approved face features
|-- models/                              # Placeholder for documented external model assets
|-- multimodal/                          # Placeholder for aligned audio-face features
|-- bisaya_affect_recognition.ipynb      # Interactive EDA and audio feature extraction
|-- bisaya_extracted_features.csv        # Current 94-feature acoustic dataset
|-- bisaya_post_class_speech_metadata.csv# Pseudonymized metadata and self-reports
|-- export_audio_outputs.py              # Authoritative grouped evaluation and export script
|-- DATA_QUALITY_REPORT.md               # Validation results and known limitations
|-- PRIVACY_AND_ETHICS.md                # Recording and participant-data safeguards
|-- Instructions.md                      # Supplied project protocol
`-- requirements.txt                     # Python dependencies
```

Raw videos and local Python environments are intentionally excluded from Git.

## Dataset conventions

The metadata table uses `Participant Code` as the observation identifier. All 40 values in `Audio Filename` match files in `Audios/` exactly.

`Spoken Word` contains the researcher's updated analysis label. It is intentionally independent of historical wording that may still appear inside an audio filename:

- Use `Audio Filename` only to locate a recording.
- Use `Spoken Word` for lexical grouping or spoken-word analysis.
- Join modality tables using `Participant Code`, not row order.

The current feature artifact contains 40 rows, 15 metadata or derived-target columns, and 94 acoustic feature columns.

## Installation

Python 3.12 is recommended. Open PowerShell in the repository root and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The local `.venv/` folder is ignored by Git.

## Run the Jupyter notebook

Start JupyterLab from the repository root:

```powershell
.\.venv\Scripts\python -m jupyter lab
```

Open `bisaya_affect_recognition.ipynb`, select the project environment as the kernel, and use **Run > Run All Cells**. The notebook supports interactive exploration and regeneration of the acoustic feature CSV.

For a non-interactive execution:

```powershell
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 bisaya_affect_recognition.ipynb
```

The notebook now evaluates valence and arousal with the same model settings and fold-local participant-grouped evaluation as `export_audio_outputs.py`. It loads the existing feature CSV by default; set `REEXTRACT_FEATURES = True` to regenerate it. Its final cell checks current metrics against saved exports and reports discrepancies without overwriting them. Regenerate the output package when adopting updated results, and keep the runtime versions with the report.

## Generate the audio output package

After the feature CSV is available, run:

```powershell
.\.venv\Scripts\python export_audio_outputs.py
```

The script validates required metadata, checks every referenced WAV file, merges the acoustic features by participant code, regenerates all tables and figures, evaluates the candidate models, and rebuilds `audio_outputs/manifest.csv`.

Generated deliverables include:

- six PNG figures covering affect distributions, spoken-word frequency, waveform and spectrogram examples, acoustic comparisons, the exact-`Kapoy` subset, and model evaluation;
- 15 CSV tables containing validated data, summaries, model comparisons, a confusion matrix, a classification report, and held-out permutation importance;
- an HTML analysis report and a plain-text result summary.

## Evaluation design

The authoritative evaluation compares a majority-class dummy classifier with Random Forest, RBF SVM, Gradient Boosting, 3-nearest neighbors, and Logistic Regression.

It uses five-fold `StratifiedGroupKFold` validation grouped by participant code. Standardization is performed inside the relevant model pipelines so information from a held-out fold is not used during scaling. Models are compared using accuracy, balanced accuracy, macro precision, macro recall, macro F1, fold variability, and out-of-fold predictions.

The current output identifies the RBF SVM as the strongest exploratory non-dummy candidate by macro F1, but its mean accuracy of 0.550 is below the 0.575 majority-class baseline. This is a negative but useful result: the present dataset is not sufficient to support a strong classification claim.

## Known limitations

- Only 40 observations are available.
- All observations represent first-year students and morning sessions.
- There is no untouched external test set.
- Model selection and evaluation use the same small dataset.
- Recording devices, microphone distance, and ambient-noise conditions are not fully documented.
- Neutral and unpleasant ratings are combined only for the current binary experiment.
- Face and multimodal features have not yet been generated.

See `DATA_QUALITY_REPORT.md` for the complete validation summary.

## Next steps

1. Confirm that video processing and sharing are covered by participant consent and institutional approval.
2. Define and document a reproducible face-feature extraction method.
3. Produce one quality-checked face-feature row per participant observation.
4. Join audio and face features on `Participant Code`, with `audio_` and `face_` prefixes.
5. Keep each participant entirely within one validation fold.
6. Consolidate additional groups or collect an external test set before making performance claims.

## Privacy and ethics

Voice and video recordings can identify participants even when filenames use pseudonymous codes. Before publishing, transferring, or merging the dataset, confirm that consent and institutional approval cover the intended use and destination.

Do not commit raw videos unless sharing has been explicitly authorized. Follow the safeguards in `PRIVACY_AND_ETHICS.md`, and do not attempt to infer identity, diagnosis, or sensitive personal traits from the recordings.
