"""Acoustic feature extraction module for Bisaya Speech Emotion Recognition.

Extracts the exact 94 acoustic and prosodic features matching the project dataset:
- Duration
- Pitch / Fundamental Frequency (F0: mean, std, max, min) via YIN algorithm
- RMS Energy (mean, std, max)
- Zero-Crossing Rate (ZCR: mean, std)
- Spectral Centroid, Bandwidth, and Rolloff (mean, std)
- 13 MFCCs (mean, std)
- 13 Delta MFCCs (mean, std)
- 13 Delta-Delta MFCCs (mean, std)
"""

from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import librosa


def extract_audio_features(
    audio_path: Union[str, Path],
    sr_target: int = 22050,
    top_db_silence: int = 25,
) -> Dict[str, float]:
    """Extract 94 acoustic features from a given WAV audio file.

    Args:
        audio_path: Path to the audio file.
        sr_target: Target sample rate for loading. Default is 22050 Hz.
        top_db_silence: Decibel threshold below peak to consider as silence.

    Returns:
        Dict mapping feature names to extracted numeric values.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(str(audio_path), sr=sr_target)

    # Silence trimming
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db_silence)
    if len(y_trimmed) == 0:
        y_trimmed = y

    duration = float(librosa.get_duration(y=y_trimmed, sr=sr))

    # 1. Pitch / Fundamental Frequency (F0) using YIN algorithm
    fmin = float(librosa.note_to_hz("C2"))
    fmax = float(librosa.note_to_hz("C7"))
    f0 = librosa.yin(y_trimmed, fmin=fmin, fmax=fmax, sr=sr)
    f0_valid = f0[(f0 >= 65) & (f0 <= 500)]

    if len(f0_valid) > 0:
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))
        f0_max = float(np.max(f0_valid))
        f0_min = float(np.min(f0_valid))
    else:
        f0_mean, f0_std, f0_max, f0_min = 0.0, 0.0, 0.0, 0.0

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y_trimmed)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms))

    # 3. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y_trimmed)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 4. Spectral Descriptors
    sc = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
    sc_mean = float(np.mean(sc))
    sc_std = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)[0]
    sb_mean = float(np.mean(sb))
    sb_std = float(np.std(sb))

    sro = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
    sro_mean = float(np.mean(sro))
    sro_std = float(np.std(sro))

    # 5. MFCCs (13), Deltas, Delta-Deltas
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)

    features: Dict[str, float] = {
        "duration": duration,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_max": f0_max,
        "f0_min": f0_min,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_max": rms_max,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "sc_mean": sc_mean,
        "sc_std": sc_std,
        "sb_mean": sb_mean,
        "sb_std": sb_std,
        "sro_mean": sro_mean,
        "sro_std": sro_std,
    }

    for i in range(13):
        idx = i + 1
        features[f"mfcc_{idx}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{idx}_std"] = float(np.std(mfcc[i]))
        features[f"mfcc_delta_{idx}_mean"] = float(np.mean(mfcc_d[i]))
        features[f"mfcc_delta_{idx}_std"] = float(np.std(mfcc_d[i]))
        features[f"mfcc_delta2_{idx}_mean"] = float(np.mean(mfcc_dd[i]))
        features[f"mfcc_delta2_{idx}_std"] = float(np.std(mfcc_dd[i]))

    return features


def get_feature_names() -> List[str]:
    """Return ordered list of 94 feature names expected by the model."""
    names = [
        "duration",
        "f0_mean", "f0_std", "f0_max", "f0_min",
        "rms_mean", "rms_std", "rms_max",
        "zcr_mean", "zcr_std",
        "sc_mean", "sc_std",
        "sb_mean", "sb_std",
        "sro_mean", "sro_std",
    ]
    for i in range(1, 14):
        names.extend([
            f"mfcc_{i}_mean",
            f"mfcc_{i}_std",
            f"mfcc_delta_{i}_mean",
            f"mfcc_delta_{i}_std",
            f"mfcc_delta2_{i}_mean",
            f"mfcc_delta2_{i}_std",
        ])
    return names
