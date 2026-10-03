"""Generate a self-contained output package for the audio analysis."""

from pathlib import Path
import shutil
import warnings

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "audio_outputs"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"
for directory in (FIGURES, TABLES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue`.*")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


def save_figure(fig, filename):
    path = FIGURES / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


metadata = pd.read_csv(ROOT / "bisaya_post_class_speech_metadata.csv")
source_features = pd.read_csv(ROOT / "bisaya_extracted_features.csv")

required = [
    "Participant Code", "Year Level", "Class Activity", "Session", "Spoken Word",
    "Self-Reported Feeling", "Valence (1-5)", "Arousal (1-5)", "Audio Filename",
]
missing = sorted(set(required) - set(metadata.columns))
if missing:
    raise ValueError(f"Missing metadata columns: {missing}")
if metadata[required].isna().any().any():
    raise ValueError("Required metadata fields contain missing values.")
if metadata["Participant Code"].duplicated().any():
    raise ValueError("Participant codes must be unique for this dataset.")

missing_audio = [
    name for name in metadata["Audio Filename"]
    if not (ROOT / "Audios" / name).is_file()
]
if missing_audio:
    raise FileNotFoundError(f"Missing audio files: {missing_audio}")

metadata["Valence_Score"] = metadata["Valence (1-5)"].str.extract(r"^(\d+)").astype(int)
metadata["Arousal_Score"] = metadata["Arousal (1-5)"].str.extract(r"^(\d+)").astype(int)
metadata["Valence_Label"] = metadata["Valence_Score"].map(
    lambda score: "Pleasant" if score >= 4 else ("Neutral" if score == 3 else "Unpleasant")
)
metadata["Valence_Binary"] = metadata["Valence_Score"].map(
    lambda score: "Pleasant" if score >= 4 else "Not Pleasant"
)
metadata["Arousal_Binary"] = metadata["Arousal_Score"].map(
    lambda score: "High Arousal" if score >= 4 else "Low/Mod Arousal"
)

if "duration" not in source_features.columns:
    raise ValueError("The source feature table has no acoustic feature block.")
acoustic_start = source_features.columns.get_loc("duration")
acoustic_columns = source_features.columns[acoustic_start:].tolist()
if len(acoustic_columns) != 94:
    raise ValueError(f"Expected 94 acoustic columns, found {len(acoustic_columns)}.")

feature_block = source_features[["Participant Code", *acoustic_columns]].copy()
full = metadata.merge(feature_block, on="Participant Code", how="left", validate="one_to_one")
if full[acoustic_columns].isna().any().any():
    raise ValueError("The merged acoustic feature block contains missing values.")

# Core data tables.
metadata.to_csv(TABLES / "01_validated_metadata.csv", index=False)
full.to_csv(TABLES / "02_acoustic_feature_dataset.csv", index=False)
shutil.copy2(ROOT / "bisaya_extracted_features.csv", TABLES / "02_source_acoustic_features.csv")


def distribution_table(column, score_column=None):
    group_columns = [column] if score_column is None else [score_column, column]
    result = metadata.groupby(group_columns, dropna=False).size().reset_index(name="Count")
    result["Percent"] = result["Count"] / len(metadata) * 100
    return result


valence_distribution = distribution_table("Valence (1-5)", "Valence_Score")
arousal_distribution = distribution_table("Arousal (1-5)", "Arousal_Score")
activity_by_valence = pd.crosstab(metadata["Class Activity"], metadata["Valence_Label"])
word_frequency = metadata["Spoken Word"].value_counts().rename_axis("Spoken Word").reset_index(name="Count")
word_by_valence = pd.crosstab(metadata["Spoken Word"], metadata["Valence_Label"])

valence_distribution.to_csv(TABLES / "03_valence_distribution.csv", index=False)
arousal_distribution.to_csv(TABLES / "04_arousal_distribution.csv", index=False)
activity_by_valence.to_csv(TABLES / "05_activity_by_valence.csv")
word_frequency.to_csv(TABLES / "06_spoken_word_frequency.csv", index=False)
word_by_valence.to_csv(TABLES / "07_spoken_word_by_valence.csv")

core_acoustic = ["duration", "f0_mean", "f0_std", "rms_mean", "rms_max", "zcr_mean", "sc_mean"]
acoustic_summary = full.groupby("Valence_Label")[core_acoustic].agg(["count", "mean", "std", "median"])
acoustic_summary.columns = [f"{feature}_{stat}" for feature, stat in acoustic_summary.columns]
acoustic_summary.reset_index().to_csv(TABLES / "08_acoustic_summary_by_valence.csv", index=False)

kapoy = full[full["Spoken Word"].astype(str).str.strip().str.casefold().eq("kapoy")].copy()
kapoy_summary = kapoy.groupby("Valence_Binary")[core_acoustic].agg(["count", "mean", "std", "median"])
kapoy_summary.columns = [f"{feature}_{stat}" for feature, stat in kapoy_summary.columns]
kapoy_summary.reset_index().to_csv(TABLES / "09_exact_kapoy_acoustic_summary.csv", index=False)
kapoy[[*required, "Valence_Score", "Arousal_Score", "Valence_Label", "Valence_Binary"]].to_csv(
    TABLES / "10_exact_kapoy_observations.csv", index=False
)

# Figure 1: affect and context overview.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(data=metadata, x="Valence_Score", color="#4C78A8", ax=axes[0, 0])
axes[0, 0].set_title("Self-Reported Valence")
sns.countplot(data=metadata, x="Arousal_Score", color="#F58518", ax=axes[0, 1])
axes[0, 1].set_title("Self-Reported Arousal")
sns.countplot(data=metadata, x="Class Activity", hue="Valence_Label", ax=axes[1, 0])
axes[1, 0].set_title("Valence by Class Activity")
sns.scatterplot(
    data=metadata, x="Valence_Score", y="Arousal_Score", hue="Valence_Label",
    style="Class Activity", s=100, ax=axes[1, 1],
)
axes[1, 1].set_title("Valence-Arousal Space")
axes[1, 1].set_xticks(range(1, 6))
axes[1, 1].set_yticks(range(1, 6))
fig.tight_layout()
save_figure(fig, "01_affect_and_context_overview.png")

# Figure 2: spoken-word frequency.
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=word_frequency, x="Spoken Word", y="Count", color="#4C78A8", ax=ax)
ax.set_title("Frequency of Updated Spoken-Word Labels")
ax.tick_params(axis="x", rotation=45)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")
fig.tight_layout()
save_figure(fig, "02_spoken_word_frequency.png")

# Figure 3: representative signals.
pleasant = full[full["Valence_Label"] == "Pleasant"].iloc[0]
unpleasant = full[full["Valence_Label"] == "Unpleasant"].iloc[0]
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
for row_index, (row, color, label) in enumerate(
    [(pleasant, "#2A9D8F", "Pleasant"), (unpleasant, "#E76F51", "Unpleasant")]
):
    signal, sample_rate = librosa.load(ROOT / "Audios" / row["Audio Filename"], sr=22050, mono=True)
    signal, _ = librosa.effects.trim(signal, top_db=25)
    librosa.display.waveshow(signal, sr=sample_rate, ax=axes[row_index, 0], color=color)
    axes[row_index, 0].set_title(f"{label} waveform: {row['Spoken Word']}")
    spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    image = librosa.display.specshow(spectrum, sr=sample_rate, x_axis="time", y_axis="log", ax=axes[row_index, 1])
    axes[row_index, 1].set_title(f"{label} log-power spectrogram")
    fig.colorbar(image, ax=axes[row_index, 1], format="%+2.0f dB")
fig.tight_layout()
save_figure(fig, "03_representative_waveforms_spectrograms.png")

# Figure 4: acoustic comparisons.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.boxplot(data=full, x="Valence_Label", y="f0_mean", color="#72B7B2", ax=axes[0, 0])
axes[0, 0].set_title("Mean F0 by Valence")
sns.boxplot(data=full, x="Valence_Label", y="rms_mean", color="#54A24B", ax=axes[0, 1])
axes[0, 1].set_title("Mean RMS Energy by Valence")
sns.boxplot(data=full, x="Arousal_Binary", y="sc_mean", color="#EECA3B", ax=axes[1, 0])
axes[1, 0].set_title("Spectral Centroid by Arousal")
sns.scatterplot(data=full, x="f0_std", y="rms_max", hue="Valence_Label", style="Arousal_Binary", s=90, ax=axes[1, 1])
axes[1, 1].set_title("Pitch Variation and Peak Energy")
fig.tight_layout()
save_figure(fig, "04_acoustic_comparisons.png")

# Figure 5: exact Kapoy subset.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, feature, title in zip(
    axes,
    ["f0_mean", "rms_mean", "sc_mean"],
    ["Mean F0", "Mean RMS Energy", "Spectral Centroid"],
):
    sns.boxplot(data=kapoy, x="Valence_Binary", y=feature, color="#B279A2", ax=ax)
    ax.set_title(f"Exact Kapoy: {title}")
fig.tight_layout()
save_figure(fig, "05_exact_kapoy_comparison.png")

# Corrected participant-grouped model comparison.
X = full[acoustic_columns]
y = full["Valence_Binary"].to_numpy()
groups = full["Participant Code"].to_numpy()
models = {
    "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "SVM (RBF)": Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
    ]),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "k-NN (k=3)": Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=3)),
    ]),
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]),
}
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
}
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(splitter.split(X, y, groups=groups))
model_rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=splits, scoring=scoring, n_jobs=1, error_score="raise")
    fold_f1 = scores["test_f1_macro"]
    model_rows.append({
        "Model": name,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Accuracy SD": scores["test_accuracy"].std(ddof=1),
        "Balanced Accuracy Mean": scores["test_balanced_accuracy"].mean(),
        "Macro F1 Mean": fold_f1.mean(),
        "Macro F1 SD": fold_f1.std(ddof=1),
        "Macro F1 95% CI Half-Width": 1.96 * fold_f1.std(ddof=1) / np.sqrt(len(fold_f1)),
        "Macro Precision Mean": scores["test_precision_macro"].mean(),
        "Macro Recall Mean": scores["test_recall_macro"].mean(),
    })
model_results = pd.DataFrame(model_rows).sort_values("Macro F1 Mean", ascending=False).reset_index(drop=True)
model_results.to_csv(TABLES / "11_model_comparison.csv", index=False)

candidate_results = model_results[model_results["Model"] != "Dummy (Most Frequent)"]
best_name = candidate_results.iloc[0]["Model"]
best_model = models[best_name]
predictions = cross_val_predict(best_model, X, y, cv=splits, n_jobs=1)
class_order = ["Pleasant", "Not Pleasant"]
matrix = confusion_matrix(y, predictions, labels=class_order)
pd.DataFrame(matrix, index=class_order, columns=class_order).rename_axis("True").to_csv(
    TABLES / "12_confusion_matrix.csv"
)
report = pd.DataFrame(classification_report(
    y, predictions, labels=class_order, output_dict=True, zero_division=0
)).T
report.to_csv(TABLES / "13_classification_report.csv")

fold_importances = []
for fold_number, (train_index, test_index) in enumerate(splits):
    fold_model = clone(best_model)
    fold_model.fit(X.iloc[train_index], y[train_index])
    importance = permutation_importance(
        fold_model, X.iloc[test_index], y[test_index], scoring="f1_macro",
        n_repeats=20, random_state=42 + fold_number, n_jobs=1,
    )
    fold_importances.append(importance.importances_mean)
mean_importance = np.mean(fold_importances, axis=0)
top_indices = np.argsort(mean_importance)[::-1][:15]
top_features = pd.DataFrame({
    "Feature": [acoustic_columns[index] for index in top_indices],
    "Held-out Permutation Importance": mean_importance[top_indices],
})
top_features.to_csv(TABLES / "14_top_permutation_features.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_order, yticklabels=class_order, ax=axes[0])
axes[0].set_title(f"Out-of-Fold Confusion Matrix: {best_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
sns.barplot(data=top_features.head(10), x="Held-out Permutation Importance", y="Feature", color="#4C78A8", ax=axes[1])
axes[1].set_title("Top Held-out Permutation Features")
fig.tight_layout()
save_figure(fig, "06_model_evaluation.png")

# --- AROUSAL EVALUATION ---
y_arousal = full["Arousal_Binary"].to_numpy()
splits_arousal = list(splitter.split(X, y_arousal, groups=groups))
arousal_model_rows = []
for name, model in models.items():
    scores_a = cross_validate(model, X, y_arousal, cv=splits_arousal, scoring=scoring, n_jobs=1, error_score="raise")
    fold_f1_a = scores_a["test_f1_macro"]
    arousal_model_rows.append({
        "Model": name,
        "Accuracy Mean": scores_a["test_accuracy"].mean(),
        "Accuracy SD": scores_a["test_accuracy"].std(ddof=1),
        "Balanced Accuracy Mean": scores_a["test_balanced_accuracy"].mean(),
        "Macro F1 Mean": fold_f1_a.mean(),
        "Macro F1 SD": fold_f1_a.std(ddof=1),
        "Macro Precision Mean": scores_a["test_precision_macro"].mean(),
        "Macro Recall Mean": scores_a["test_recall_macro"].mean(),
    })
arousal_model_results = pd.DataFrame(arousal_model_rows).sort_values("Macro F1 Mean", ascending=False).reset_index(drop=True)
arousal_model_results.to_csv(TABLES / "15_arousal_model_comparison.csv", index=False)

best_arousal_row = arousal_model_results[arousal_model_results["Model"] != "Dummy (Most Frequent)"].iloc[0]
best_arousal_name = best_arousal_row["Model"]
best_arousal_model = models[best_arousal_name]
arousal_predictions = cross_val_predict(best_arousal_model, X, y_arousal, cv=splits_arousal, n_jobs=1)
arousal_class_order = ["High Arousal", "Low/Mod Arousal"]
arousal_matrix = confusion_matrix(y_arousal, arousal_predictions, labels=arousal_class_order)
pd.DataFrame(arousal_matrix, index=arousal_class_order, columns=arousal_class_order).rename_axis("True").to_csv(
    TABLES / "16_arousal_confusion_matrix.csv"
)
arousal_report = pd.DataFrame(classification_report(
    y_arousal, arousal_predictions, labels=arousal_class_order, output_dict=True, zero_division=0
)).T
arousal_report.to_csv(TABLES / "17_arousal_classification_report.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(arousal_matrix, annot=True, fmt="d", cmap="Purples", xticklabels=arousal_class_order, yticklabels=arousal_class_order, ax=axes[0])
axes[0].set_title(f"Out-of-Fold Confusion Matrix (Arousal): {best_arousal_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
sns.barplot(data=arousal_model_results[arousal_model_results["Model"] != "Dummy (Most Frequent)"], x="Macro F1 Mean", y="Model", color="#9467BD", ax=axes[1])
axes[1].set_title("Arousal Model Comparison (Macro F1)")
fig.tight_layout()
save_figure(fig, "07_arousal_model_evaluation.png")

# --- 2D AFFECT QUADRANT MAPPING ---
def get_quadrant(val, ar):
    is_p = (val == "Pleasant")
    is_h = ("High" in ar)
    if is_p and is_h:
        return "Pleasant-Activated (Q1)"
    elif is_p and not is_h:
        return "Pleasant-Deactivated (Q4)"
    elif not is_p and is_h:
        return "Unpleasant-Activated (Q2)"
    else:
        return "Unpleasant-Deactivated (Q3)"

true_quadrants = [get_quadrant(v, a) for v, a in zip(y, y_arousal)]
pred_quadrants = [get_quadrant(v, a) for v, a in zip(predictions, arousal_predictions)]

quadrant_df = pd.DataFrame({
    "Participant Code": metadata["Participant Code"],
    "Spoken Word": metadata["Spoken Word"],
    "True Valence": y,
    "True Arousal": y_arousal,
    "True Quadrant": true_quadrants,
    "Predicted Quadrant": pred_quadrants,
})
quadrant_counts = quadrant_df["True Quadrant"].value_counts().rename_axis("Affect Quadrant").reset_index(name="Count")
quadrant_counts.to_csv(TABLES / "18_2d_affect_quadrant_distribution.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=quadrant_df, y="True Quadrant", palette="Set2", ax=ax)
ax.set_title("Distribution of Post-Class 2D Affect Quadrants (N=40)")
ax.set_xlabel("Student Count")
fig.tight_layout()
save_figure(fig, "08_2d_affect_quadrants.png")

majority_baseline = metadata["Valence_Binary"].value_counts(normalize=True).max()
best_row = candidate_results.iloc[0]
summary_lines = [
    "Bisaya Post-Class Speech: Audio Analysis Output Summary",
    "",
    f"Observations: {len(metadata)}",
    f"Acoustic features per observation: {len(acoustic_columns)}",
    f"Exact Kapoy observations: {len(kapoy)}",
    f"Best Valence candidate: {best_name}",
    f"  - Mean accuracy: {best_row['Accuracy Mean']:.3f}",
    f"  - Majority-class accuracy baseline: {majority_baseline:.3f}",
    f"  - Mean Macro F1: {best_row['Macro F1 Mean']:.3f} (SD: {best_row['Macro F1 SD']:.3f})",
    f"  - Mean balanced accuracy: {best_row['Balanced Accuracy Mean']:.3f}",
    f"Best Arousal candidate: {best_arousal_name}",
    f"  - Mean accuracy: {best_arousal_row['Accuracy Mean']:.3f}",
    f"  - Mean Macro F1: {best_arousal_row['Macro F1 Mean']:.3f}",
    "",
    "Interpretation: exploratory only. The dataset is small, contains one year level and one session,",
    "and has no independent external test set. Incorporates 2D Affect Quadrant mapping (Valence x Arousal).",
]
(REPORTS / "audio_analysis_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

for superseded_report in (
    REPORTS / "audio_analysis_notebook.html",
    REPORTS / "bisaya_affect_recognition.ipynb",
):
    superseded_report.unlink(missing_ok=True)

html_report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bisaya Post-Class Speech: Audio Analysis & 2D Affect</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1180px; margin: 40px auto; padding: 0 24px; color: #202124; }}
h1, h2 {{ color: #244b74; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
.metric {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; background: #f7fafc; }}
.metric strong {{ display: block; font-size: 1.5rem; margin-top: 5px; }}
.warning {{ border-left: 5px solid #d97706; padding: 12px 16px; background: #fff7ed; }}
img {{ width: 100%; height: auto; margin: 12px 0 28px; border: 1px solid #e5e7eb; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eaf0f6; }}
</style>
</head>
<body>
<h1>Bisaya Post-Class Speech: Audio Analysis & 2D Affect Recognition</h1>
<div class="metrics">
  <div class="metric">Observations<strong>{len(metadata)}</strong></div>
  <div class="metric">Acoustic features<strong>{len(acoustic_columns)}</strong></div>
  <div class="metric">Best Valence Model<strong>{best_name}</strong></div>
  <div class="metric">Valence Macro F1<strong>{best_row['Macro F1 Mean']:.3f}</strong></div>
  <div class="metric">Best Arousal Model<strong>{best_arousal_name}</strong></div>
  <div class="metric">Arousal Macro F1<strong>{best_arousal_row['Macro F1 Mean']:.3f}</strong></div>
</div>
<p class="warning">Exploratory results only. The sample contains one year level and one session, and there is no independent external test set.</p>
<h2>1. Valence Model Comparison</h2>
{model_results.round(3).to_html(index=False)}
<h2>2. Valence Classification Report</h2>
{report.round(3).to_html()}
<h2>3. Arousal Model Comparison</h2>
{arousal_model_results.round(3).to_html(index=False)}
<h2>4. 2D Affect Quadrant Distribution</h2>
{quadrant_counts.to_html(index=False)}
<h2>Figures & Visual Evaluations</h2>
<img src="../figures/06_model_evaluation.png" alt="Valence Model Evaluation">
<img src="../figures/07_arousal_model_evaluation.png" alt="Arousal Model Evaluation">
<img src="../figures/08_2d_affect_quadrants.png" alt="2D Affect Quadrant Distribution">
<img src="../figures/05_exact_kapoy_comparison.png" alt="Exact Kapoy acoustic comparison">
<img src="../figures/04_acoustic_comparisons.png" alt="Acoustic feature comparisons">
</body>
</html>
"""
(REPORTS / "audio_analysis_report.html").write_text(html_report, encoding="utf-8")

manifest_rows = []
descriptions = {
    "figures": "PNG figure generated from the audio analysis",
    "tables": "CSV table generated from the audio analysis",
    "reports": "Human-readable analysis report",
}
for category in ("figures", "tables", "reports"):
    for path in sorted((OUTPUT / category).glob("*")):
        manifest_rows.append({
            "Category": category,
            "File": path.relative_to(OUTPUT).as_posix(),
            "Description": descriptions[category],
        })
pd.DataFrame(manifest_rows).to_csv(OUTPUT / "manifest.csv", index=False)

print(f"Created audio output package at: {OUTPUT}")
print(f"Figures: {len(list(FIGURES.glob('*.png')))}")
print(f"Tables: {len(list(TABLES.glob('*.csv')))}")
print(f"Best Valence model: {best_name}")
print(f"Best Arousal model: {best_arousal_name}")
