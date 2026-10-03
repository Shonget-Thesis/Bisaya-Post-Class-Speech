# Bisaya Speech Emotion Recognition — 2D Affect Inference Pipeline

This folder contains the self-contained inference pipeline and acoustic feature extractor for the CS412 Bisaya Post-Class Speech Emotion Recognition project.

It implements **Russell's 2D Circumplex Model of Affect**, estimating both **Valence** (Pleasantness) and **Arousal** (Energy Activation) to map voice recordings into 4 emotional quadrants.

---

## 📁 Directory Structure

```text
inference/
├── feature_extractor.py       # 94-feature extraction using librosa (YIN F0, RMS, ZCR, MFCCs)
├── train_and_save_model.py    # Trains dual Valence & Arousal SVM models to `models/`
├── predict.py                 # Live inference script mapping audio to 2D Affect Regions
└── README.md                  # Documentation and usage guide
```

---

## 🗺️ The 4-Quadrant Affect Mapping

```text
                     HIGH AROUSAL (Active / High Energy)
                                     ▲
                                     │
           QUADRANT II               │               QUADRANT I
     [ Unpleasant-Activated ]        │         [ Pleasant-Activated ]
     • Tense, Frustrated, Anxious    │         • Energetic, Excited
     • e.g. "Grabe!", "Kulbaan"      │         • e.g. "Lingaw!", "Nadasig"
                                     │
 UNPLEASANT ─────────────────────────┼─────────────────────────► PLEASANT
 (Low Valence)                       │                           (High Valence)
                                     │
           QUADRANT III              │               QUADRANT IV
    [ Unpleasant-Deactivated ]       │        [ Pleasant-Deactivated ]
     • Tired, Drained, Exhausted     │         • Calm, Relaxed, Relieved
     • e.g. "Kapoy", "Katulgon"      │         • e.g. "Chill lang", "Salamat"
                                     │
                                     ▼
                     LOW AROUSAL (Passive / Low Energy)
```

---

## 🚀 How to Use

### 1. Train and Save the Models
```bash
python inference/train_and_save_model.py
```

### 2. Run Predictions on Any Audio File
```bash
python inference/predict.py Audios/Y1-208_Y1_AM_Kapoya.wav
```

### 3. Example Output
```text
=================================================================
  BISAYA SPEECH AFFECT RECOGNITION (VALENCE + AROUSAL)
=================================================================
 Audio File       : Y1-208_Y1_AM_Kapoya.wav
 Duration         : 2.02 s
 Pitch Mean (F0)  : 208.5 Hz (std: 91.8)
 RMS Energy       : 0.05
 Zero-Crossing    : 0.1335
-----------------------------------------------------------------
 [1] VALENCE PREDICTION : [NOT PLEASANT]
     * Not Pleasant :  54.3% [#############------------]
     * Pleasant     :  45.7% [###########--------------]

 [2] AROUSAL PREDICTION : [HIGH AROUSAL]
     * High Arousal :  82.8% [####################-----]
     * Low Arousal  :  17.2% [####---------------------]
-----------------------------------------------------------------
 [3] 2D AFFECT REGION   : [UNPLEASANT-ACTIVATED]
     Quadrant           : Quadrant II (Low Valence, High Arousal)
     Interpretation     : Tense, overwhelmed, or frustrated (e.g. Kulbaan, Overwhelmed, Grabe)
=================================================================
```

---

## 🔬 Python API Usage in Code

```python
from inference.predict import predict_audio

result = predict_audio("path/to/recording.wav")

print("Valence:", result["valence"]["prediction"])         # 'Pleasant' or 'Not Pleasant'
print("Arousal:", result["arousal"]["prediction"])         # 'High Arousal' or 'Low Arousal'
print("Affect Region:", result["affect_profile"]["region"])# 'Unpleasant-Activated'
print("Interpretation:", result["affect_profile"]["interpretation"])
```
