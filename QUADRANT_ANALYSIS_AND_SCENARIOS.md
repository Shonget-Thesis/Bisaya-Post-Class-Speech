# 2D Affect Quadrant Analysis & Real-World Student Scenarios
## Bisaya Speech Emotion Recognition in Post-Class University Environments

This guide explains how **Russell's 2D Circumplex Model of Affect** is analyzed in this research, how audio features map into the 4 emotional quadrants, and how different classroom scenarios translate into concrete predictions.

---

## 🗺️ 1. The 2D Affect Coordinate System

In Russell's Circumplex Model (1980), human affective experience is mapped as a continuous 2D plane formed by two orthogonal axes:

1. **Horizontal Axis ($\text{X-Axis}$) = Valence (Pleasantness)**
   * Measures cognitive and hedonic quality: *Unpleasant (Negative)* $\longleftrightarrow$ *Pleasant (Positive)*.
2. **Vertical Axis ($\text{Y-Axis}$) = Arousal (Activation / Physiological Energy)**
   * Measures autonomic nervous system mobilization: *Deactivated (Low Energy)* $\longleftrightarrow$ *Activated (High Energy)*.

```text
                           HIGH AROUSAL (+Y)
                 [Activated / High Physiological Energy]
                                    ▲
                                    │
          QUADRANT II               │               QUADRANT I
    [ UNPLEASANT-ACTIVATED ]        │         [ PLEASANT-ACTIVATED ]
    • Tense, Frustrated, Anxious    │         • Energetic, Excited, Inspired
    • High Energy, Low Valence      │         • High Energy, High Valence
    • e.g. "Kulbaan", "Grabe"       │         • e.g. "Lingaw", "Nadasig"
                                    │
UNPLEASANT (-X) ────────────────────┼────────────────────► PLEASANT (+X)
[Negative Hedonic Tone]             │              [Positive Hedonic Tone]
                                    │
          QUADRANT III              │               QUADRANT IV
   [ UNPLEASANT-DEACTIVATED ]       │        [ PLEASANT-DEACTIVATED ]
    • Tired, Exhausted, Depleted    │         • Calm, Relieved, Relaxed
    • Low Energy, Low Valence       │         • Low Energy, High Valence
    • e.g. "Kapoy", "Katulgon"      │         • e.g. "Chill lang", "Salamat"
                                    │
                                    ▼
                            LOW AROUSAL (-Y)
                [Deactivated / Low Physiological Energy]
```

---

## ⚙️ 2. The 4-Step Analytical Pipeline

When a student speaks into the microphone, the system analyzes the recording through four systematic steps:

```text
Raw Audio (.wav)
      │
      ▼
[Step 1: Feature Extraction] ──► 94 Acoustic Descriptors (Pitch F0, RMS Energy, ZCR, MFCCs)
      │
      ▼
[Step 2: Dual Model Prediction]
      ├─► Valence Classifier (SVM RBF)           ──► P(Pleasant) vs P(Not Pleasant)
      └─► Arousal Classifier (Logistic Reg.)     ──► P(High Arousal) vs P(Low Arousal)
      │
      ▼
[Step 3: Coordinate Assignment] ─────────────────► (Valence Label, Arousal Label)
      │
      ▼
[Step 4: Russell Quadrant Mapping] ──────────────► Quadrant I, II, III, or IV + Interpretation
```

### Acoustic Drivers Behind the Axes:
* **What drives Arousal?** Autonomic arousal tightens vocal folds and boosts respiratory push. The classifier detects **higher mean pitch ($F_0$)**, **widened pitch variance**, and **higher Root Mean Square (RMS) loudness**.
* **What drives Valence?** Pleasantness subtly alters the shape of the vocal tract and harmonic decay. The classifier detects **high-order MFCC timbre coefficients (MFCC 8, 10)** and **spectral centroid balance**.

---

## 🎭 3. Real-World Classroom Scenarios

Here is how real Computer Science academic situations map into the 4 quadrants, complete with verified recordings from the dataset:

---

### Scenario 1: Quadrant I — *Pleasant-Activated* (Excited & Engaged)
* **Academic Context**: A student just successfully resolved a major bug in their hands-on Data Structures programming lab, or finished an engaging game development lecture.
* **Student's Feeling**: Motivated, energized, proud, and enthusiastic.
* **Bisaya Vocabulary Spoken**: *"Lingaw!"* (Fun), *"Nadasig"* (Inspired), *"Nalipay"* (Happy), *"Yes sa wakas!"*.
* **Acoustic Signature**:
  * **Pitch ($F_0$)**: Elevated fundamental frequency ($> 180\text{ Hz}$).
  * **RMS Energy**: High speech volume ($> 0.08$).
  * **Spectral Quality**: Bright, energetic vocal resonance.
* **Classifier Prediction**:
  * Valence: `Pleasant` ($> 50\%$)
  * Arousal: `High Arousal` ($> 50\%$)
* **Real Dataset Sample**:
  * Audio: `Audios/Y1-202_Y1_AM_Nadasig.wav`
  * Meaning: The student felt enthusiastic and inspired after morning class.

---

### Scenario 2: Quadrant II — *Unpleasant-Activated* (Stressed & Frustrated)
* **Academic Context**: A student just ran out of time during a live timed coding exam, encountered unexpected syntax errors right before submission, or felt overwhelmed by a difficult algorithm discussion.
* **Student's Feeling**: Anxious, tense, panicky, flustered, or frustrated.
* **Bisaya Vocabulary Spoken**: *"Kulbaan"* (Nervous/Anxious), *"Grabe"* (Intense/Overwhelming), *"Gubot"* (Chaotic/Messy), *"Impyerno bay"* (It was hell, bro).
* **Acoustic Signature**:
  * **Pitch ($F_0$)**: Abnormally high or unstable pitch caused by muscle tension ($200\text{--}240\text{ Hz}$).
  * **RMS Energy**: High or strained volume burst.
  * **Zero-Crossing Rate (ZCR)**: Elevated, reflecting tense, unvoiced breath turbulence.
* **Classifier Prediction**:
  * Valence: `Not Pleasant` ($> 50\%$)
  * Arousal: `High Arousal` ($> 50\%$)
* **Real Dataset Sample**:
  * Audio: `Audios/Y1-208_Y1_AM_Kapoya.wav`
  * Output: **Valence: Not Pleasant (54.3%) | Arousal: High Arousal (96.1%)**
  * Interpretation: High physiological stress and distress.

---

### Scenario 3: Quadrant III — *Unpleasant-Deactivated* (Exhausted & Drained)
* **Academic Context**: A student just finished a 4-hour afternoon theoretical lecture (e.g., Automata & Complexity Theory) or spent the whole night pulling an all-nighter before class.
* **Student's Feeling**: Mental burnout, sleep deprivation, physical and cognitive depletion.
* **Bisaya Vocabulary Spoken**: *"Kapoy"* (Tired), *"Katulgon"* (Sleepy), *"Hago"* (Fatigued/Weary), *"Alimoot"* (Suffocating heat).
* **Acoustic Signature**:
  * **Pitch ($F_0$)**: Low, monotone, flat pitch contour with minimal variation ($110\text{--}130\text{ Hz}$).
  * **RMS Energy**: Soft, quiet, or muffled loudness ($< 0.06$).
  * **Duration**: Drawn-out, lethargic vowel prolongation.
* **Classifier Prediction**:
  * Valence: `Not Pleasant` ($> 50\%$)
  * Arousal: `Low/Mod Arousal` ($> 50\%$)
* **Real Dataset Sample**:
  * Audio: `Audios/Y1-204_Y1_AM_Kapoy.wav`
  * Output: **Valence: Not Pleasant (52.4%) | Arousal: Low/Mod Arousal (94.1%)**
  * Interpretation: Severe post-class cognitive depletion and fatigue.

---

### Scenario 4: Quadrant IV — *Pleasant-Deactivated* (Relieved & Calm)
* **Academic Context**: A student just submitted their final project on time, completed their last class on a Friday, or enjoyed an air-conditioned lecture with no immediate deadlines.
* **Student's Feeling**: Relaxed, peaceful, relieved, calm satisfaction.
* **Bisaya Vocabulary Spoken**: *"Chill lang"* (Just chilling), *"Salamat"* (Thank goodness / grateful), *"Bugnaw"* (Comfortably cool), *"Sayon"* (Easy/Manageable).
* **Acoustic Signature**:
  * **Pitch ($F_0$)**: Moderate, steady, warm pitch contour ($130\text{--}150\text{ Hz}$).
  * **RMS Energy**: Low-to-moderate, gentle vocal projection without stress peaks.
  * **Spectral Flatness**: Smooth, harmonic acoustic decay.
* **Classifier Prediction**:
  * Valence: `Pleasant` ($> 50\%$)
  * Arousal: `Low/Mod Arousal` ($> 50\%$)
* **Real Dataset Sample**:
  * Audio: `Audios/Y1-203_Y1_AM_Kapoy_nga_dili.wav`
  * Output: **Valence: Pleasant (52.3%) | Arousal: Low/Mod Arousal (88.0%)**
  * Interpretation: Peaceful relief ("Tired, but not really tired / relieved").

---

## 🔍 4. The Deep-Dive Case Study: The "Kapoy" Dichotomy

The most powerful demonstration of our 2D quadrant analysis is resolving the word **"Kapoy"**:

| Scenario | Utterance Spoken | Pitch ($F_0$) | Energy (RMS) | Predicted Quadrant | True Student State |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Student A** | *"Kapoy..."* | $119.7\text{ Hz}$ (Low/Flat) | Low, slow | **Quadrant III** (Unpleasant-Deactivated) | Burnout / Exhausted |
| **Student B** | *"Kapoya oy!"* | $208.5\text{ Hz}$ (High/Spike) | High burst | **Quadrant II** (Unpleasant-Activated) | Panic / Overwhelmed |
| **Student C** | *"Kapoy nga dili"* | $142.8\text{ Hz}$ (Warm/Steady) | Moderate, gentle | **Quadrant IV** (Pleasant-Deactivated) | Relieved / Mission Accomplished |

### Why This Matters for Your Defense:
If an emotion recognition system only looked at **text words**, it would treat Student A, Student B, and Student C as having the exact same emotion. 
By using **2D Acoustic Quadrant Mapping**, the system correctly identifies that Student A needs rest, Student B is suffering high-pressure panic, and Student C is actually feeling positive relief!

---

## ⚖️ 5. Borderline / Ambiguous Scenarios & Probability Confidence

In real-world deployment, students do not always fall neatly into the center of a quadrant. The system outputs continuous probability bars to express uncertainty:

### Example: Borderline Valence (52.3% Pleasant vs. 47.7% Not Pleasant)
* **What it means**: The student experienced mixed feelings (e.g., class was mentally hard, but satisfying).
* **How to interpret it**: The primary quadrant is Quadrant IV, but the close probability distribution indicates mild ambivalence or transition between quadrants.
* **Panel Defense Takeaway**: Continuous probabilities reflect realistic human psychology—emotions are not rigid discrete boxes; they are fluid regions in a continuous circumplex space.

---

## 🎓 6. Practical Educational Impact

If implemented in a University Learning Management System (LMS) or intelligent tutoring system:
* **Quadrant I Alerts**: High engagement $\rightarrow$ Indicates optimal pedagogy and teaching methods.
* **Quadrant II Alerts**: High stress / frustration $\rightarrow$ Signals that an exam or lab assignment was disproportionately difficult or paced too fast.
* **Quadrant III Alerts**: Burnout / depletion $\rightarrow$ Advises instructors to integrate breaks or reschedule intensive coding labs.
* **Quadrant IV Alerts**: Calm satisfaction $\rightarrow$ Indicates a successful conclusion of a milestone.
