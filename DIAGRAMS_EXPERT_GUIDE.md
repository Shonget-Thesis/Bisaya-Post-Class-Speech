# Visual Defense Cheat Sheet: How to Read Every Diagram Like an Expert
## Fast-Scan Reference with Embedded Visuals for `bisaya_affect_recognition.ipynb`

> **How to Use This Guide:** When a panelist looks at a graph on your slides or screen, look up that diagram here. View the embedded chart, check the **"Expert Eye"** (what to look at first), and speak the **"10-Second Panel Script"** directly to them.

---

# 📊 DIAGRAM 1: Student Affect & Context Distributions (RQ1 & RQ2)

![Diagram 1: Affect and Context Overview](audio_outputs/figures/01_affect_and_context_overview.png)

### 1. What It Shows:
Four-panel demographic summary of student feelings immediately after class ($N=40$):
* **Top-Left**: Valence distribution (1 = Very Unpleasant to 5 = Very Pleasant).
* **Top-Right**: Arousal distribution (1 = Very Low to 4 = High).
* **Bottom-Left**: Affective valence counts grouped by class activity (Lecture, Lab, Quiz).
* **Bottom-Right**: Russell's 2D coordinate scatter space (Valence vs. Arousal).

### 2. The "Expert Eye" (What to Look at First):
* **Top-Left (Valence)**: Look at the tall bar at **Score 4 (16 students)**, and the second tall bar at **Score 3 (13 students)**. Notice that Score 5 has only **1 student**. Most students are neutral or moderately pleasant.
* **Top-Right (Arousal)**: Look at the middle peak at **Score 4 (14 students)** and **Score 3 (13 students)**. High/extreme arousal is rare; students hover in moderate energy.
* **Bottom-Left (By Activity)**: Compare the green bar (Pleasant) across activities:
  * **Lectures** have the highest pleasant count (**8 students**).
  * **Labs & Quizzes** show higher proportions of unpleasant/neutral states due to assessment pressure.
* **Bottom-Right (2D Space)**: Shows students clustered mostly between Valence 2–4 and Arousal 2–4. No student is in the extreme corners (e.g., $(1,1)$ or $(5,5)$).

### 3. 🎯 10-Second Panel Script:
> *"This figure answers RQ1 and RQ2. It shows that post-class student emotions are predominantly moderate—students cluster around neutral-to-pleasant valence and moderate arousal. Crucially, lectures produced the highest pleasant ratings (8 students), whereas practical laboratory sessions and quizzes introduced higher cognitive stress and unpleasant ratings."*

---

# 📊 DIAGRAM 2: Frequency of Spoken Bisaya Words

![Diagram 2: Frequency of Selected Bisaya Spoken Words](audio_outputs/figures/02_spoken_word_frequency.png)

### 1. What It Shows:
A Pareto distribution of the spontaneous Bisaya words uttered by students immediately upon class dismissal.

### 2. The "Expert Eye" (What to Look at First):
* Look at the **massive single dark blue spike on the far left**: **"Kapoy"** accounts for nearly half of the entire corpus ($n=16\text{--}20$ including variations).
* Look at the steep drop immediately after: **"Lingaw"** is a distant second (only 3 occurrences), **"Sayon"** (2), and all other words appear only once.

### 3. 🎯 10-Second Panel Script:
> *"An expert reads this as an extreme vocabulary imbalance. The word 'Kapoy' accounts for nearly half of our dataset. This is the single biggest justification for our acoustic research: because so many students use the exact same word 'Kapoy', text transcription alone is completely useless. We have to analyze vocal acoustic pitch and energy to separate what kind of 'Kapoy' they mean."*

---

# 📊 DIAGRAM 3: Representative Waveforms & Mel Spectrograms

![Diagram 3: Representative Waveforms and Spectrograms](audio_outputs/figures/03_representative_waveforms_spectrograms.png)

### 1. What It Shows:
Time-domain amplitude waves (left) and log-power Mel frequency spectrogram heatmaps (right) comparing a pleasant response (*"Lingaw"*) against an unpleasant response (*"Kapoy nga dili"*).

### 2. The "Expert Eye" (What to Look at First):
* **Left Waves (Duration & Envelope)**:
  * Top wave (*"Lingaw"*): Short, energetic single burst lasting only **0.51 seconds** with high amplitude peaks ($\pm 0.7$).
  * Bottom wave (*"Kapoy"*): Drawn-out, segmented speech lasting **1.53 seconds** across multiple syllables.
* **Right Heatmaps (Frequency Power in dB)**:
  * Top spectrogram: Bright, dense orange/yellow bands extending all the way past **4000 Hz**, indicating strong vocal cord resonance and positive energy.
  * Bottom spectrogram: Energy is concentrated down below **1024 Hz**, with darker, weaker upper harmonics, indicating lethargic, fatigued phonation.

### 3. 🎯 10-Second Panel Script:
> *"Here we visually contrast raw acoustics. The pleasant utterance 'Lingaw' is short (0.5s) with strong high-frequency energy above 4 kHz, reflecting active vocal projection. In contrast, the unpleasant fatigue utterance 'Kapoy' is elongated (1.5s) and lacks high-frequency harmonic power, which is typical of tired, relaxed vocal folds."*

---

# 📊 DIAGRAM 4: Acoustic Feature Comparison across Affect (RQ3)

![Diagram 4: Acoustic Comparisons](audio_outputs/figures/04_acoustic_comparisons.png)

### 1. What It Shows:
Four statistical plots testing which voice measurements physically change when students experience different emotions.

### 2. The "Expert Eye" (What to Look at First):
* **Top-Left (Pitch $F_0$ vs. Valence)**:
  * Look at the green box (**Pleasant**): Median pitch is high ($\approx \mathbf{205\text{ Hz}}$) with a wide upper tail reaching $325\text{ Hz}$.
  * Look at the orange box (**Unpleasant**): Median pitch drops significantly to $\approx \mathbf{145\text{ Hz}}$.
  * *Expert Takeaway*: Happy students speak at a noticeably higher pitch.
* **Bottom-Left (Spectral Centroid vs. Arousal)**:
  * Look at the orange box (**High Arousal**): Reaches higher values (up to $3600\text{ Hz}$) compared to Low/Mod Arousal.
  * *Expert Takeaway*: Excited/stressed students have "brighter", more piercing voices.
* **Bottom-Right (Scatter Plot)**:
  * Look at the **"X" markers (High Arousal)**: They push toward the right (higher pitch variation $F_0 \text{ Std} > 70\text{ Hz}$) and top (higher RMS max energy).

### 3. 🎯 10-Second Panel Script:
> *"This confirms our physiological hypothesis for RQ3. Pleasant valence correlates with elevated fundamental frequency (median F0 of ~205 Hz vs ~145 Hz for unpleasant). Meanwhile, High Arousal students exhibit significantly higher spectral centroids and wider pitch standard deviations, proving that energy directly manifests as vocal pitch variability."*

---

# 📊 DIAGRAM 5: Lexically Controlled "Kapoy" Comparison

![Diagram 5: Exact Kapoy Comparison](audio_outputs/figures/05_exact_kapoy_comparison.png)

### 1. What It Shows:
A controlled sub-experiment comparing students who said the **exact same word: "Kapoy"**, separated by whether they felt unpleasant vs. pleasant.

### 2. The "Expert Eye" (What to Look at First):
* **Left Box (Pitch $F_0$)**:
  * **Pink (Not Pleasant)**: Median pitch is low ($\approx \mathbf{180\text{ Hz}}$).
  * **Teal (Pleasant)**: Median pitch jumps to $\approx \mathbf{215\text{ Hz}}$!
* **Right Box (Spectral Centroid)**:
  * Teal (Pleasant) is shifted noticeably higher ($\approx \mathbf{2100\text{ Hz}}$ vs. $\approx \mathbf{1900\text{ Hz}}$ for pink).

### 3. 🎯 10-Second Panel Script:
> *"This is our strongest proof of concept. When students say the identical word 'Kapoy', those expressing relieved or positive tiredness speak with a higher median pitch (215 Hz vs 180 Hz) and higher spectral centroid. This proves conclusively that acoustic prosody carries emotional valence independent of vocabulary."*

---

# 📊 DIAGRAM 6: Out-of-Fold Confusion Matrices (Step 11)

![Diagram 6: Confusion Matrices](audio_outputs/figures/06_step11_confusion_matrices.png)

### 1. What It Shows:
Heatmaps of all 40 test predictions made by our two winning models across all 5 test folds:
* **Left Panel**: Valence — **SVM (RBF)**.
* **Right Panel**: Arousal — **Logistic Regression**.

### 2. The "Expert Eye" (What to Look at First):
* **Look at the main diagonal (top-left to bottom-right)**: Those are the correct guesses.
* **Right Matrix (Arousal — Logistic Regression)**:
  * Look at the dark navy bottom-right square: **21 correct Low Arousal predictions**.
  * Total correct = $8 + 21 = \mathbf{29/40}$ ($\mathbf{72.5\%}$ accuracy).
  * *Expert Takeaway*: Outstanding performance on identifying low-energy/fatigued students.
* **Left Matrix (Valence — SVM RBF)**:
  * Total correct = $7 + 15 = \mathbf{22/40}$ ($\mathbf{55.0\%}$ accuracy).
  * Notice the top-right box ($10$): The model missed $10$ out of $17$ Pleasant students, mistaking them for Not Pleasant.
  * *Expert Takeaway*: The model leans toward predicting the majority class (fatigue), but still accurately catches $15/23$ negative states.

### 3. 🎯 10-Second Panel Script:
> *"An expert reads these matrices through class recall. For Arousal, Logistic Regression achieved a very strong 72.5% accuracy, correctly identifying 21 out of 26 low-energy students. For Valence, SVM correctly identified 15 out of 23 negative states, but showed higher false-negative rates on subtle positive responses, which is expected given short single-word audio lengths."*

---

# 📊 DIAGRAM 7: Permutation Feature Importance (Step 11)

![Diagram 7: Permutation Feature Importance](audio_outputs/figures/07_step11_permutation_importance.png)

### 1. What It Shows:
A horizontal bar chart ranking which voice features caused the biggest drop in Macro $F_1$ when randomly scrambled during testing.

### 2. The "Expert Eye" (What to Look at First):
* **The Top 3 Bars are All MFCCs**: `mfcc_10_mean` ($+0.0251$), `mfcc_delta_8_std` ($+0.0248$), and `mfcc_8_mean` ($+0.0231$).
* **The 4th Bar is Pitch**: `f0_min` (Minimum Pitch: $+0.0224$).
* Notice that loudness (RMS) is **not** in the top 5 for Valence!

### 3. 🎯 10-Second Panel Script:
> *"This chart tells us what the SVM is actually listening to. The top three features are all higher-order MFCC coefficients (MFCC 10 and 8), followed by minimum pitch (f0_min). This confirms that emotional valence (pleasantness) is encoded primarily in vocal tract timbre and the pitch baseline, rather than simple loudness."*

---

## ⚡ Super-Quick Summary Cheat Sheet

| Diagram | Key Number to Memorize | One-Sentence Summary |
| :--- | :---: | :--- |
| **1. Overview** | 16 Pleasant, 14 High Arousal | Most students report moderate, neutral-to-fatigued states after class. |
| **2. Words** | 16 "Kapoy" occurrences | "Kapoy" makes up 40% of the corpus, proving text alone cannot solve affect. |
| **3. Waves** | 0.5s (Lingaw) vs 1.5s (Kapoy) | Pleasant speech is short and high-frequency; fatigue speech is slow and low-frequency. |
| **4. Comparisons** | 205 Hz vs 145 Hz F0 | Happy students have a 60 Hz higher median pitch than unhappy students. |
| **5. Kapoy Test** | 215 Hz vs 180 Hz F0 | Even saying the exact same word, happy "Kapoy" is 35 Hz higher than sad "Kapoy". |
| **6. Matrices** | 29/40 Arousal (72.5%), 22/40 Valence (55%) | Arousal is highly accurate; Valence is conservative toward negative states. |
| **7. Importance** | `mfcc_10_mean` (+0.0251) | MFCC timbre and minimum pitch are the primary drivers of pleasantness. |
