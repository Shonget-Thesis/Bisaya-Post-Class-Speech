"""Inference script for Bisaya Speech Emotion Recognition (2D Valence-Arousal Model).

Usage:
    python inference/predict.py Audios/Y1-204_Y1_AM_Kapoy.wav
    python inference/predict.py path/to/your_audio.wav
"""

import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.feature_extractor import extract_audio_features, get_feature_names

DEFAULT_BUNDLE_PATH = ROOT_DIR / "models" / "bisaya_ser_models.joblib"
LEGACY_VALENCE_PATH = ROOT_DIR / "models" / "bisaya_ser_svm.joblib"


def get_affect_region_info(valence: str, arousal: str) -> dict:
    """Map binary Valence and Arousal into Russell's 4-Quadrant Affect Region."""
    is_pleasant = (valence.lower() == "pleasant")
    is_high_arousal = ("high" in arousal.lower())

    if is_pleasant and is_high_arousal:
        return {
            "region": "Pleasant-Activated",
            "quadrant": "Quadrant I (High Valence, High Arousal)",
            "interpretation": "Energetic, excited, or enthusiastic (e.g. Lingaw, Nalipay)",
        }
    elif is_pleasant and not is_high_arousal:
        return {
            "region": "Pleasant-Deactivated",
            "quadrant": "Quadrant IV (High Valence, Low Arousal)",
            "interpretation": "Calm, relaxed, or relieved (e.g. Chill lang, Salamat, Bugnaw)",
        }
    elif not is_pleasant and is_high_arousal:
        return {
            "region": "Unpleasant-Activated",
            "quadrant": "Quadrant II (Low Valence, High Arousal)",
            "interpretation": "Tense, overwhelmed, or frustrated (e.g. Kulbaan, Overwhelmed, Grabe)",
        }
    else:
        return {
            "region": "Unpleasant-Deactivated",
            "quadrant": "Quadrant III (Low Valence, Low Arousal)",
            "interpretation": "Tired, sleepy, or exhausted (e.g. Kapoy, Katulgon, Hago)",
        }


def predict_audio(audio_path: str, model_bundle_path: Path = DEFAULT_BUNDLE_PATH) -> dict:
    """Predict 2D affective state (Valence, Arousal, Affect Region) from an audio file."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Feature Extraction
    features_dict = extract_audio_features(audio_path)
    feature_names = get_feature_names()
    features_vector = [features_dict[name] for name in feature_names]
    df_features = pd.DataFrame([features_vector], columns=feature_names)

    # 2. Load Models
    if model_bundle_path.is_file():
        bundle = joblib.load(model_bundle_path)
        valence_model = bundle.get("valence_model")
        arousal_model = bundle.get("arousal_model")
    elif LEGACY_VALENCE_PATH.is_file():
        valence_model = joblib.load(LEGACY_VALENCE_PATH)
        arousal_model = None
    else:
        raise FileNotFoundError(
            f"Model not found at {model_bundle_path}. "
            "Please run 'python inference/train_and_save_model.py' first."
        )

    # 3. Valence Prediction
    val_pred = valence_model.predict(df_features)[0]
    val_probs = {}
    if hasattr(valence_model, "predict_proba"):
        probs = valence_model.predict_proba(df_features)[0]
        for cls_name, prob in zip(valence_model.classes_, probs):
            val_probs[cls_name] = float(prob)

    # 4. Arousal Prediction (if dual model loaded)
    arousal_pred = None
    arousal_probs = {}
    affect_info = None

    if arousal_model is not None:
        arousal_pred = arousal_model.predict(df_features)[0]
        if hasattr(arousal_model, "predict_proba"):
            probs_a = arousal_model.predict_proba(df_features)[0]
            for cls_name, prob in zip(arousal_model.classes_, probs_a):
                arousal_probs[cls_name] = float(prob)

        affect_info = get_affect_region_info(val_pred, arousal_pred)

    result = {
        "audio_file": audio_path.name,
        "valence": {
            "prediction": val_pred,
            "probabilities": val_probs,
        },
        "arousal": {
            "prediction": arousal_pred,
            "probabilities": arousal_probs,
        } if arousal_pred else None,
        "affect_profile": affect_info,
        "acoustic_summary": {
            "duration_sec": round(features_dict["duration"], 2),
            "pitch_f0_mean_hz": round(features_dict["f0_mean"], 1),
            "pitch_f0_std_hz": round(features_dict["f0_std"], 1),
            "rms_energy_mean": round(features_dict["rms_mean"], 4),
            "zero_crossing_rate_mean": round(features_dict["zcr_mean"], 4),
            "spectral_centroid_mean_hz": round(features_dict["sc_mean"], 1),
        },
        "raw_features": features_dict,
    }
    return result


def format_cli_output(result: dict):
    """Print clean ASCII 2D prediction report to console."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  BISAYA SPEECH AFFECT RECOGNITION (VALENCE + AROUSAL)")
    print(f"{sep}")
    print(f" Audio File       : {result['audio_file']}")
    print(f" Duration         : {result['acoustic_summary']['duration_sec']} s")
    print(f" Pitch Mean (F0)  : {result['acoustic_summary']['pitch_f0_mean_hz']} Hz (std: {result['acoustic_summary']['pitch_f0_std_hz']})")
    print(f" RMS Energy       : {result['acoustic_summary']['rms_energy_mean']}")
    print(f" Zero-Crossing    : {result['acoustic_summary']['zero_crossing_rate_mean']}")
    print("-" * 65)

    # Valence Output
    val_pred = result["valence"]["prediction"]
    print(f" [1] VALENCE PREDICTION : [{val_pred.upper()}]")
    if result["valence"]["probabilities"]:
        for cls_name, prob in result["valence"]["probabilities"].items():
            bar_len = int(prob * 25)
            bar = "#" * bar_len + "-" * (25 - bar_len)
            print(f"     * {cls_name:13s}: {prob*100:5.1f}% [{bar}]")

    # Arousal Output
    if result.get("arousal"):
        ar_pred = result["arousal"]["prediction"]
        print(f"\n [2] AROUSAL PREDICTION : [{ar_pred.upper()}]")
        if result["arousal"]["probabilities"]:
            for cls_name, prob in result["arousal"]["probabilities"].items():
                bar_len = int(prob * 25)
                bar = "#" * bar_len + "-" * (25 - bar_len)
                print(f"     * {cls_name:13s}: {prob*100:5.1f}% [{bar}]")

    # 2D Affect Profile Output
    if result.get("affect_profile"):
        aff = result["affect_profile"]
        print("-" * 65)
        print(f" [3] 2D AFFECT REGION   : [{aff['region'].upper()}]")
        print(f"     Quadrant           : {aff['quadrant']}")
        print(f"     Interpretation     : {aff['interpretation']}")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict 2D Affect (Valence + Arousal -> Affect Region) from Bisaya speech recording."
    )
    parser.add_argument("audio_path", type=str, help="Path to the .wav audio file")
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_BUNDLE_PATH), help="Path to trained model bundle"
    )
    args = parser.parse_args()

    try:
        res = predict_audio(args.audio_path, Path(args.model))
        format_cli_output(res)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
