"""Train and save Speech Emotion Recognition models (Valence and Arousal) for 2D Affect mapping."""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = ROOT_DIR / "bisaya_post_class_speech_metadata.csv"
FEATURES_PATH = ROOT_DIR / "bisaya_extracted_features.csv"
DUAL_MODEL_OUTPUT_PATH = MODELS_DIR / "bisaya_ser_models.joblib"
VALENCE_MODEL_OUTPUT_PATH = MODELS_DIR / "bisaya_ser_svm.joblib"
MODEL_INFO_PATH = MODELS_DIR / "bisaya_ser_model_info.json"


def train_and_save():
    print(f"Loading data from {METADATA_PATH.name} and {FEATURES_PATH.name}...")
    df_meta = pd.read_csv(METADATA_PATH)
    df_features = pd.read_csv(FEATURES_PATH)

    # 1. Prepare Valence labels: 4-5 = Pleasant, 1-3 = Not Pleasant
    df_meta["Valence_Score"] = df_meta["Valence (1-5)"].str.extract(r"^(\d+)").astype(int)
    df_meta["Valence_Binary"] = df_meta["Valence_Score"].map(
        lambda score: "Pleasant" if score >= 4 else "Not Pleasant"
    )

    # 2. Prepare Arousal labels: 4-5 = High Arousal, 1-3 = Low Arousal
    df_meta["Arousal_Score"] = df_meta["Arousal (1-5)"].str.extract(r"^(\d+)").astype(int)
    df_meta["Arousal_Binary"] = df_meta["Arousal_Score"].map(
        lambda score: "High Arousal" if score >= 4 else "Low Arousal"
    )

    # Locate acoustic columns
    acoustic_start = df_features.columns.get_loc("duration")
    acoustic_columns = df_features.columns[acoustic_start:].tolist()

    df_full = df_meta.merge(
        df_features[["Participant Code", *acoustic_columns]],
        on="Participant Code",
        how="left"
    )

    X = df_full[acoustic_columns]
    y_valence = df_full["Valence_Binary"].to_numpy()
    y_arousal = df_full["Arousal_Binary"].to_numpy()

    print(f"Dataset shape: {X.shape[0]} observations, {X.shape[1]} acoustic features")
    print(f"Valence distribution: {pd.Series(y_valence).value_counts().to_dict()}")
    print(f"Arousal distribution: {pd.Series(y_arousal).value_counts().to_dict()}")

    # Build Valence Pipeline
    valence_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Build Arousal Pipeline
    arousal_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Fitting Valence Model (Pleasant vs Not Pleasant)...")
    valence_pipeline.fit(X, y_valence)

    print("Fitting Arousal Model (High vs Low Arousal)...")
    arousal_pipeline.fit(X, y_arousal)

    # Save Dual Model bundle
    model_bundle = {
        "valence_model": valence_pipeline,
        "arousal_model": arousal_pipeline,
        "feature_names": acoustic_columns,
    }
    joblib.dump(model_bundle, DUAL_MODEL_OUTPUT_PATH)
    print(f"Saved dual model bundle to: {DUAL_MODEL_OUTPUT_PATH}")

    # Also save individual valence model for backwards compatibility
    joblib.dump(valence_pipeline, VALENCE_MODEL_OUTPUT_PATH)

    # Save metadata info
    model_info = {
        "model_architecture": "SVM (RBF Kernel) with StandardScaler",
        "targets": {
            "valence": {
                "classes": list(valence_pipeline.classes_),
                "distribution": {k: int(v) for k, v in pd.Series(y_valence).value_counts().items()}
            },
            "arousal": {
                "classes": list(arousal_pipeline.classes_),
                "distribution": {k: int(v) for k, v in pd.Series(y_arousal).value_counts().items()}
            }
        },
        "affect_quadrants": {
            "Pleasant-Activated": "Pleasant + High Arousal (Energetic, excited)",
            "Pleasant-Deactivated": "Pleasant + Low Arousal (Calm, relaxed, relieved)",
            "Unpleasant-Activated": "Not Pleasant + High Arousal (Tense, frustrated, overwhelmed)",
            "Unpleasant-Deactivated": "Not Pleasant + Low Arousal (Tired, drained, fatigued)"
        },
        "num_features": len(acoustic_columns),
        "n_samples_trained": int(len(X)),
    }
    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved model metadata info to: {MODEL_INFO_PATH}")


if __name__ == "__main__":
    train_and_save()
