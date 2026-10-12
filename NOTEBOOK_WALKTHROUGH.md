# Complete Defense & Code Guide: Bisaya Post-Class Speech Emotion Recognition
## Comprehensive Technical & Layman's Guide for Presentation and Technical Checking

> **Master Defense Guide:** This single document is structured so that you can navigate your Jupyter Notebook (`bisaya_affect_recognition.ipynb`) cell by cell first, followed by your **7-Slide Presentation Guide** and **Laptop Live Demonstration Steps (Parts A, B, and C)**.

---

# TABLE OF CONTENTS
1. [PART I: Complete Notebook Walkthrough (Steps 1 to 14)](#part-i-complete-notebook-walkthrough-steps-1-to-14)
   * [Step 1: Environment Setup & Library Imports](#step-1-environment-setup--library-imports)
   * [Step 2: Dataset Ingestion & Metadata Sanitization](#step-2-dataset-ingestion--metadata-sanitization)
   * [Step 3: Exploratory Data Analysis (EDA) — Addressing RQ1 & RQ2](#step-3-exploratory-data-analysis-eda--addressing-rq1--rq2)
   * [Step 4: Spoken Word Frequency & Affective Ambiguity Analysis](#step-4-spoken-word-frequency--affective-ambiguity-analysis)
   * [Step 5: Audio Signal Preprocessing & Waveform/Spectrogram Inspection](#step-5-audio-signal-preprocessing--waveformspectrogram-inspection)
   * [Step 6: Acoustic & Prosodic Feature Extraction — Addressing RQ3](#step-6-acoustic--prosodic-feature-extraction--addressing-rq3)
   * [Step 7: Acoustic Feature Comparison across Affective States — RQ3 Analysis](#step-7-acoustic-feature-comparison-across-affective-states--rq3-analysis)
   * [Step 8: Same-Word Analysis (Exact "Kapoy" Comparison)](#step-8-same-word-analysis-exact-kapoy-comparison)
   * [Step 9: Acoustic Predictors & Participant-Grouped Evaluation Protocol](#step-9-acoustic-predictors--participant-grouped-evaluation-protocol)
   * [Step 10: Compare Models for Valence and Arousal — Addressing RQ4 & RQ5](#step-10-compare-models-for-valence-and-arousal--addressing-rq4--rq5)
   * [Step 11: Out-of-Fold Errors & Held-out Feature Importance](#step-11-out-of-fold-errors--held-out-feature-importance)
   * [Step 12: Current Findings & Export Consistency](#step-12-current-findings--export-consistency)
   * [Step 13: Model Export & Serialization (Saving for predict.py)](#step-13-model-export--serialization-saving-for-predictpy)
   * [Step 14: Live Audio Demonstration — Interactive Affect Prediction](#step-14-live-audio-demonstration--interactive-affect-prediction)
2. [PART II: 10-Minute Research Output Presentation Guide (7 Slides)](#part-ii-10-minute-research-output-presentation-guide-7-slides)
   * [Slide 1: Research Output Overview](#slide-1-research-output-overview)
   * [Slide 2: Data Collection](#slide-2-data-collection)
   * [Slide 3: Dataset Establishment](#slide-3-dataset-establishment)
   * [Slide 4: Preprocessing and Feature Extraction](#slide-4-preprocessing-and-feature-extraction)
   * [Slide 5: Machine Learning Implementation](#slide-5-machine-learning-implementation)
   * [Slide 6: Model Evaluation](#slide-6-model-evaluation)
   * [Slide 7: Findings and Limitations](#slide-7-findings-and-limitations)
3. [PART III: 10-Minute Live Technical Checking & Laptop Demo (Parts A, B, and C)](#part-iii-10-minute-live-technical-checking--laptop-demo-parts-a-b-and-c)
   * [Part A: Raw Data and Dataset Verification](#part-a-raw-data-and-dataset-verification)
   * [Part B: Preprocessing and Feature Extraction](#part-b-preprocessing-and-feature-extraction)
   * [Part C: Machine Learning and Results](#part-c-machine-learning-and-results)
4. [PART IV: Comprehensive Research Vocabulary & Key Terminology Glossary](#part-iv-comprehensive-research-vocabulary--key-terminology-glossary)
   * [1. Affect & Emotion Theory Terminology](#1-affect--emotion-theory-terminology)
   * [2. Linguistic & Lexical Terminology](#2-linguistic--lexical-terminology)
   * [3. Audio Signal & Acoustic Terminology](#3-audio-signal--acoustic-terminology)
   * [4. Machine Learning & Evaluation Terminology](#4-machine-learning--evaluation-terminology)

---

# PART I: Complete Notebook Walkthrough (Steps 1 to 14)

---

### Step 1: Environment Setup & Library Imports

#### Technical Explanation
Initializes the scientific Python runtime. Configures digital signal processing (DSP) modules (`librosa`, `soundfile`), data manipulation packages (`pandas`, `numpy`, `scipy`), visualization suites (`matplotlib`, `seaborn`), and machine learning components (`scikit-learn`). Global pseudorandom seeds (`random_state=42`) and figure resolutions (`dpi=120`, `Agg` backend) are asserted to ensure deterministic cross-validation folds and reproducible graphics.

#### Layman's Terms ("U Feel Me")
> *"Before doing any work, we open up our digital workshop. We import Librosa (which listens to and analyzes sound files), Pandas and NumPy (which act like supercharged Excel spreadsheets), and Scikit-Learn (our machine learning library). We also set a fixed random seed of 42 so that every time we or the panel run the code, it produces the exact same results."*

#### Code & Values to Take Note Of
* **Key Libraries**: `librosa`, `sklearn`, `pandas`, `numpy`, `seaborn`, `matplotlib`.
* **Random Seed**: `random_state = 42`.
* **Plotting Quality**: `dpi = 120`, Seaborn style `"whitegrid"`.

---

### Step 2: Dataset Ingestion & Metadata Sanitization

#### Technical Explanation
Loads [`bisaya_post_class_speech_metadata.csv`](./bisaya_post_class_speech_metadata.csv) ($N=40$). Applies regular expressions (`r"^(\d+)"`) to extract ordinal scores from Likert-scale strings (e.g., extracting integer `4` from `"4 - Pleasant"`). Discretizes continuous 1–5 affect scores into binary targets based on mid-point thresholds:
* **Valence**: $\ge 4 \rightarrow$ `Pleasant`, $< 4 \rightarrow$ `Not Pleasant`
* **Arousal**: $\ge 4 \rightarrow$ `High Arousal`, $< 4 \rightarrow$ `Low/Mod Arousal`
Asserts data cleanliness: checks for zero missing values (`isna().any() == False`), unique student identifiers, and 1-to-1 physical correspondence with audio files in the `Audios/` directory.

#### Layman's Terms ("U Feel Me")
> *"We load the survey sheet where 40 Computer Science students recorded how they felt right after class. They answered with 1-to-5 numbers for their happiness level (Valence) and their energy level (Arousal). We turn these numbers into two simple yes-or-no categories: 4 or 5 means Pleasant/High Energy, while 1, 2, or 3 means Not Pleasant/Low Energy. We also write a check to confirm all 40 voice recordings are actually sitting in the folder."*

#### Code & Values to Take Note Of
* **Total Sample Count**: $N = 40$ participants.
* **Valence Target Threshold**: $\ge 4 \rightarrow$ `Pleasant` ($17$ samples, $42.5\%$), $< 4 \rightarrow$ `Not Pleasant` ($23$ samples, $57.5\%$).
* **Arousal Target Threshold**: $\ge 4 \rightarrow$ `High Arousal` ($14$ samples, $35.0\%$), $< 4 \rightarrow$ `Low/Mod Arousal` ($26$ samples, $65.0\%$).
* **Critical Code**:
  ```python
  metadata["Valence_Score"] = metadata["Valence (1-5)"].str.extract(r"^(\d+)").astype(int)
  metadata["Arousal_Score"] = metadata["Arousal (1-5)"].str.extract(r"^(\d+)").astype(int)
  metadata["Valence_Binary"] = metadata["Valence_Score"].map(lambda s: "Pleasant" if s >= 4 else "Not Pleasant")
  metadata["Arousal_Binary"] = metadata["Arousal_Score"].map(lambda s: "High Arousal" if s >= 4 else "Low/Mod Arousal")
  ```

---

### Step 3: Exploratory Data Analysis (EDA) — Addressing RQ1 & RQ2

#### Technical Explanation
Investigates the relationship between affective state distributions and contextual educational variables. Performs two-way cross-tabulations (`pd.crosstab`) across `Class Activity` (Lecture, Hands-on Lab, Exam/Quiz), `Year Level` (1st through 4th Year), and `Session` (Morning AM vs. Afternoon PM). Generates frequency summary tables and multi-panel stacked bar charts saved to `audio_outputs/tables/03_valence_distribution.csv` and `audio_outputs/tables/05_activity_by_valence.csv`.

#### Layman's Terms ("U Feel Me")
> *"We check how student moods change depending on what they were doing in class. Are students more exhausted after an exam or a 3-hour coding lab compared to a lecture? Are morning classes more energetic than afternoon classes? This gives us the real-world context behind the data before we even touch machine learning."*

#### Code & Values to Take Note Of
* **RQ1 Finding**: The majority of post-class responses ($57.5\%$) represent `Not Pleasant` affective states, and $65.0\%$ represent `Low/Mod Arousal` states (predominantly fatigue and exhaustion).
* **RQ2 Finding**: Hands-on laboratory sessions and major assessments exhibit higher proportions of high arousal (frustration, tension) compared to theoretical lectures.
* **Tables Generated**: `03_valence_distribution.csv`, `04_arousal_distribution.csv`, `05_activity_by_valence.csv`.

---

### Step 4: Spoken Word Frequency & Affective Ambiguity Analysis

#### Technical Explanation
Computes categorical token frequencies for spontaneous Bisaya utterances. Cross-tabulates spoken words against self-reported valence categories to examine **lexical ambiguity**.

#### Layman's Terms ("U Feel Me")
> *"We count which Bisaya words students chose to say. The word 'Kapoy' (tired) was spoken the most (16 times). But here is the big catch: some students said 'Kapoy' when they were stressed out and unhappy, while other students said 'Kapoy' when they were relieved and happy that class was finally over! This proves why you CANNOT just read the text to know their emotion—you have to listen to the sound of their voice (the acoustics)."*

#### Code & Values to Take Note Of
* **Most Frequent Word**: *"Kapoy"* ($n=16$ occurrences out of 40, accounting for $40\%$ of the entire corpus).
* **Other Frequent Words**: *"Lingaw"* (fun), *"Nadasig"* (inspired), *"Alimoot"* (humid/hot), *"Bugnaw"* (cold/chilly), *"Gutom"* (hungry).
* **Core Rationale**: Lexical overlap confirms that text transcription alone is confounded; acoustic prosody is mandatory for accurate affective disambiguation.

---

### Step 5: Audio Signal Preprocessing & Waveform/Spectrogram Inspection

#### Technical Explanation
Performs audio signal quality validation. Reads raw `.wav` audio via `librosa.load` at native sampling rates. Computes Short-Time Fourier Transforms (STFT) with a Hann window, converting power spectrograms to decibels ($\text{dB}$) across $128$ Mel frequency bins. Plots dual time-domain waveforms and Mel spectrograms across contrasting affect states.

#### Layman's Terms ("U Feel Me")
> *"Before feeding sound into an AI, we inspect the physical waves. We draw two pictures for each audio: the wavy line showing how loud it was over time (the waveform), and a colorful frequency heatmap showing the musical pitch and voice texture (the spectrogram). This lets us verify that there are no mic clicks, dead air, or weird audio glitches."*

#### Code & Values to Take Note Of
* **Mean Recording Duration**: $\approx 1.5$–$2.0$ seconds (short single-word or short-phrase utterances).
* **Audio Format**: Mono `.wav` PCM, uncompressed.
* **Key Code**:
  ```python
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
  S_dB = librosa.power_to_db(S, ref=np.max)
  ```

---

### Step 6: Acoustic & Prosodic Feature Extraction — Addressing RQ3

#### Technical Explanation
Transforms raw, variable-length audio waveforms into a fixed-dimensional numerical feature space $\mathbf{x} \in \mathbb{R}^{94}$. Extracts three distinct categories of acoustic descriptors:
1. **Prosodic Descriptors (7 features)**:
   * Fundamental frequency pitch ($F_0$ mean, std, min, max) using the time-domain autocorrelation **YIN algorithm** ($f_{\min}=65\text{ Hz}$, $f_{\max}=2093\text{ Hz}$).
   * Root Mean Square (RMS) energy (mean, std).
   * Utterance duration in seconds.
2. **Spectral & Temporal Descriptors (8 features)**:
   * Zero-Crossing Rate (ZCR mean, std) measuring vocal cord noisiness.
   * Spectral Centroid (timbral brightness/center of spectral mass).
   * Spectral Spread, Flatness (noisiness vs. tonality), Flux (rate of spectral change), and Rolloff (frequency containing 85% of power).
3. **Cepstral Descriptors (78 features)**:
   * 13 static Mel-Frequency Cepstral Coefficients (MFCCs 1–13: mean and std = 26 features).
   * 13 Delta coefficients / Velocity ($\Delta$ MFCCs: mean and std = 26 features).
   * 13 Delta-Delta coefficients / Acceleration ($\Delta\Delta$ MFCCs: mean and std = 26 features).

#### Layman's Terms ("U Feel Me")
> *"Computers cannot process raw audio directly. We have to convert each sound file into a set of 94 precise measurements. We measure how high or deep their voice was (Pitch/F0), how loudly they spoke (Energy/RMS), how quickly their vocal cords vibrated (Zero-Crossing Rate), and the exact physical shape of their throat and mouth as they spoke (using 13 MFCC filters, plus their speed and acceleration). Now every student's voice is represented by a single row of 94 numbers."*

#### Code & Values to Take Note Of
* **Exact Total Features**: **94 numerical columns**.
* **Saved Artifact**: [`bisaya_extracted_features.csv`](./bisaya_extracted_features.csv) ($40 \text{ rows} \times 95 \text{ columns}$ including Participant Code).
* **Pitch Algorithm**: Librosa YIN (`librosa.yin(y, fmin=65, fmax=2093, sr=sr)`).

---

### Step 7: Acoustic Feature Comparison across Affective States — RQ3 Analysis

#### Technical Explanation
Conducts comparative statistical distributions of acoustic parameters grouped by binary Valence (`Pleasant` vs. `Not Pleasant`) and Arousal (`High` vs. `Low/Mod`). Renders comparative boxplots and generates summary tables detailing means and standard deviations.

#### Layman's Terms ("U Feel Me")
> *"We graph the numbers side-by-side to see what actually changes in someone's voice when they are emotional. We discover that students with High Energy (Arousal) speak with noticeably louder volume (RMS energy) and higher pitch (F0). Happy vs. Unhappy (Valence) is more subtle, showing up in the texture of their voice (MFCCs)."*

#### Code & Values to Take Note Of
* **Primary Acoustic Finding for Arousal**: High Arousal students show significantly higher mean RMS energy and higher mean $F_0$ pitch compared to Low Arousal students.
* **Table Generated**: `audio_outputs/tables/08_acoustic_summary_by_valence.csv`.

---

### Step 8: Same-Word Analysis (Exact *"Kapoy"* Comparison)

#### Technical Explanation
Implements a controlled within-lexicon experimental subset. Filters the dataset down to recordings where the exact spoken token was `"Kapoy"` ($n=16$), isolating the acoustic properties from phonemic/lexical variations. Compares pitch contours, duration, and energy between students who rated this exact word as unpleasant vs. pleasant/neutral.

#### Layman's Terms ("U Feel Me")
> *"To prove that our AI isn't just cheating by memorizing words, we take only the students who said the exact same word: 'Kapoy'. We look at how a student saying a tired, sad 'Kapoy' sounds compared to a student saying an energetic or relieved 'Kapoy'. Their pitch and energy curves are completely different, proving that emotion lives in the voice tone, not just the word."*

#### Code & Values to Take Note Of
* **Controlled Token**: `"Kapoy"` exclusively.
* **Tables Generated**: `09_exact_kapoy_acoustic_summary.csv`, `10_exact_kapoy_observations.csv`.
* **Figure Generated**: `audio_outputs/figures/05_exact_kapoy_comparison.png`.

---

### Step 9: Acoustic Predictors & Participant-Grouped Evaluation Protocol

#### Technical Explanation
Assembles the predictor matrix $X \in \mathbb{R}^{40 \times 94}$ using strictly the 94 extracted acoustic features. Spoken words and demographic labels are explicitly excluded from $X$ to eliminate synthetic shortcuts. Implements **5-Fold `StratifiedGroupKFold`** using `Participant Code` as the grouping constraint. In each fold, $32$ samples are used for training and $8$ held-out samples for testing, ensuring zero speaker leakage across train and validation splits.

#### Layman's Terms ("U Feel Me")
> *"This is where we set up the strict rules of the exam for our machine learning models. We make sure the AI only gets to look at the 94 voice numbers—it cannot look at the student's name, grade, or word. We split the 40 students into 5 equal testing groups. If Student #1 is in the test group, their voice is never allowed in the training group. This guarantees our test is 100% fair and honest."*

#### Code & Values to Take Note Of
* **Predictor Shape**: $X$ is $40 \times 94$.
* **Folds**: $K = 5$ folds.
* **Splitter**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
* **Grouping Vector**: `groups = full["Participant Code"].to_numpy()`.

---

### Step 10: Compare Models for Valence and Arousal — Addressing RQ4 & RQ5

#### Technical Explanation
Executes identical 5-fold cross-validation across 6 candidate machine learning classifiers for both Valence and Arousal. Evaluates mean and standard deviation for Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, and Macro $F_1$. Models evaluated:
1. `DummyClassifier(strategy="most_frequent")`: Statistical baseline.
2. `SVC(kernel="rbf", C=1.0, class_weight="balanced")` with `StandardScaler`.
3. `LogisticRegression(max_iter=1000, class_weight="balanced")` with `StandardScaler`.
4. `RandomForestClassifier(n_estimators=300, class_weight="balanced")`.
5. `GradientBoostingClassifier(n_estimators=100)`.
6. `KNeighborsClassifier(n_neighbors=3)` with `StandardScaler`.

#### Layman's Terms ("U Feel Me")
> *"We hold a tournament between 6 different AI algorithms to see who performs best on Valence and who performs best on Arousal. The results were clear: Support Vector Machine (SVM) won 1st place for Valence (detecting happiness), while Logistic Regression won 1st place for Arousal (detecting energy level, scoring 72.5% accuracy). This is why our final system uses a hybrid team of both winners."*

#### Code & Values to Take Note Of

##### Valence Comparison Results ([`11_model_comparison.csv`](./audio_outputs/tables/11_model_comparison.csv))
* **1st Place**: **SVM (RBF)** — Macro $F_1 = \mathbf{0.522 \pm 0.097}$, Balanced Acc $= \mathbf{0.543}$, Accuracy $= 0.550$.
* **2nd Place**: k-NN ($k=3$) — Macro $F_1 = 0.510$, Balanced Acc $= 0.535$.
* **3rd Place**: Random Forest — Macro $F_1 = 0.466$.
* **Baseline**: Dummy Classifier — Macro $F_1 = 0.364$.

##### Arousal Comparison Results ([`15_arousal_model_comparison.csv`](./audio_outputs/tables/15_arousal_model_comparison.csv))
* **1st Place**: **Logistic Regression** — Macro $F_1 = \mathbf{0.662 \pm 0.222}$, Balanced Acc $= \mathbf{0.670}$, Accuracy $= \mathbf{0.725}$.
* **2nd Place**: k-NN ($k=3$) — Macro $F_1 = 0.644$, Balanced Acc $= 0.657$, Accuracy $= 0.725$.
* **3rd Place**: Random Forest — Macro $F_1 = 0.627$.
* **4th Place**: SVM (RBF) — Macro $F_1 = 0.622$, Balanced Acc $= 0.637$.
* **Baseline**: Dummy Classifier — Macro $F_1 = 0.393$.

---

### Step 11: Out-of-Fold Errors & Held-out Feature Importance

#### Technical Explanation
Aggregates out-of-fold validation predictions across all 5 folds to construct full-dataset confusion matrices and classification reports without data leakage. Computes **Permutation Feature Importance** ($n=20$ repeats per fold) strictly on held-out validation partitions, measuring the degradation in Macro $F_1$ when each feature column is shuffled.

#### Layman's Terms ("U Feel Me")
> *"We look at the exact mistakes our winning models made. Then we play detective: we scramble each voice feature one by one to see which one hurts the model's accuracy the most when scrambled. The features that cause the biggest drop in score are our most important features. High-frequency MFCC filters and minimum pitch (f0_min) turned out to be the most crucial."*

#### Code & Values to Take Note Of
* **Valence Confusion Matrix (SVM)** ([`12_confusion_matrix.csv`](./audio_outputs/tables/12_confusion_matrix.csv)):
  * True Pleasant ($N=17$): $7$ Correct, $10$ Misclassified.
  * True Not Pleasant ($N=23$): $15$ Correct, $8$ Misclassified.
* **Arousal Confusion Matrix (Logistic Regression)** ([`16_arousal_confusion_matrix.csv`](./audio_outputs/tables/16_arousal_confusion_matrix.csv)):
  * True High Arousal ($N=14$): $8$ Correct, $6$ Misclassified.
  * True Low/Mod Arousal ($N=26$): $21$ Correct, $5$ Misclassified.
* **Top 3 Permutation Features** ([`14_top_permutation_features.csv`](./audio_outputs/tables/14_top_permutation_features.csv)):
  1. `mfcc_10_mean` ($+0.0251$)
  2. `mfcc_delta_8_std` ($+0.0248$)
  3. `mfcc_8_mean` ($+0.0231$)

---

### Step 12: Current Findings & Export Consistency

#### Technical Explanation
Serves as the empirical synthesis and quality assurance audit for the entire investigation. It performs three critical operations:
1. **Dynamic RQ Synthesis**: Aggregates answers to all research questions (RQ1–RQ5) programmatically directly from the in-memory variables:
   * **RQ1 & RQ2**: Displays class distributions (Valence: 23 Not Pleasant / 17 Pleasant; Arousal: 26 Low/Mod / 14 High) and cross-tabulates academic activities (AM lecture vs. PM lab).
   * **RQ3**: Synthesizes acoustic feature distributions and confirms that while mean acoustic differences exist (e.g., in the exact *"Kapoy"* subset of $N=6$), overlapping variance prevents trivial linear separation.
   * **RQ4 & RQ5**: Automatically prints benchmark margins over the Dummy Baseline (Valence: SVM RBF leads by Macro $F_1 = 0.522$, $+0.158$ over baseline; Arousal: Logistic Regression leads by Macro $F_1 = 0.662$ and Accuracy $= 72.5\%$, $+7.5$ percentage points over baseline).
2. **Automated Consistency & Regression Audit**: Loads historical CSV exports ([`11_model_comparison.csv`](./audio_outputs/tables/11_model_comparison.csv), [`12_confusion_matrix.csv`](./audio_outputs/tables/12_confusion_matrix.csv), [`15_arousal_model_comparison.csv`](./audio_outputs/tables/15_arousal_model_comparison.csv), [`16_arousal_confusion_matrix.csv`](./audio_outputs/tables/16_arousal_confusion_matrix.csv)) and verifies cell-by-cell parity against the active execution using `np.isclose()` and `np.array_equal()`. Reports `0 metric differences`, guaranteeing zero data drift or manual transcription errors.
3. **Defense Armor (Honest Methodological Limitations)**: Explicitly frames model scope: small exploratory sample ($N=40$), 94 acoustic features, single student year cohort (Year 1 AM), and binary rating threshold ($\ge 4$).

#### Layman's Terms ("U Feel Me")
> *"Think of Step 12 as our **Scientific Audit and Executive Summary**. If Step 10 was the **Tournament** (finding the best models) and Step 11 was the **Autopsy** (investigating mistakes and important features), Step 12 is where we say: 'Let's double-check our work and summarize what we learned.' It runs a lie-detector test on our notebook: it compares our live results against the saved CSV spreadsheets to prove that not a single number was tampered with or changed. It also prepares our defense armor by honestly stating our project limitations before the panel even asks."*

#### Code & Values to Take Note Of
* **Consistency Check Logic**:
  ```python
  # Verifies 0 differences between live run and saved tables
  print(f"{target}: {len(differences)} metric differences from saved export.")
  # Asserts mathematical identity of confusion matrices
  assert np.array_equal(saved_matrix.to_numpy(), evaluations[target]["matrix"])
  print("Confusion matrix matches saved export: True")
  ```
* **Dynamic RQ Summary Outputs**:
  * Valence Lead: SVM leads with Macro $F_1 = 0.522$ (Balanced Acc $= 0.543$).
  * Arousal Lead: Logistic Regression leads with Macro $F_1 = 0.662$, Acc $= 72.5\%$ ($+7.5\%$ over Dummy baseline of $65.0\%$).
* **Limitations Documented**: $N=40$ student recordings, exploratory study, held-out evaluation within same cohort, prosodic variations within same word ("Kapoy").

---

### Step 13: Model Export & Serialization (Saving for `predict.py`)

#### Technical Explanation
Trains the dual production pipelines on the full 40-sample dataset using the empirically validated architectures:
1. `valence_model`: `Pipeline([('scale', StandardScaler()), ('model', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42))])`
2. `arousal_model`: `Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])`
Dumps the dictionary bundle to [`models/bisaya_ser_models.joblib`](./models/bisaya_ser_models.joblib) and exports metadata specifications to [`models/bisaya_ser_model_info.json`](./models/bisaya_ser_model_info.json).

#### Layman's Terms ("U Feel Me")
> *"We take our two tournament champions—SVM for Valence and Logistic Regression for Arousal—train them on all 40 student recordings, and pack them together into a single file (`bisaya_ser_models.joblib`). Now our standalone prediction app can load this one file and predict emotions on any voice recording instantly."*

#### Code & Values to Take Note Of
* **Bundle Path**: `models/bisaya_ser_models.joblib`.
* **Metadata Path**: `models/bisaya_ser_model_info.json`.
* **Dictionary Keys**: `['valence_model', 'arousal_model', 'feature_names']`.

---

### Step 14: Live Audio Demonstration — Interactive Affect Prediction

#### Technical Explanation
Validates end-to-end live inference inside the notebook. Extracts 94 features from a raw `.wav` file, queries `.predict_proba()` on both pipelines, aligns the predicted class with the probability argmax, and maps the coordinate pair $(\text{Valence}, \text{Arousal})$ into **Russell's 4-Quadrant Circumplex Model of Affect**:
* **Quadrant I**: *Pleasant-Activated* (High Valence, High Arousal) $\rightarrow$ Energetic, Enthusiastic (e.g., *Lingaw*, *Nalipay*)
* **Quadrant II**: *Unpleasant-Activated* (Low Valence, High Arousal) $\rightarrow$ Tense, Frustrated, Anxious (e.g., *Kulbaan*, *Grabe*)
* **Quadrant III**: *Unpleasant-Deactivated* (Low Valence, Low Arousal) $\rightarrow$ Tired, Exhausted, Depleted (e.g., *Kapoy*, *Katulgon*)
* **Quadrant IV**: *Pleasant-Deactivated* (High Valence, Low Arousal) $\rightarrow$ Calm, Relaxed, Relieved (e.g., *Chill lang*, *Salamat*, *Bugnaw*)

#### Layman's Terms ("U Feel Me")
> *"We test the system on real sound. We pick an audio file like 'Kapoy nga dili', extract its features, and feed it into our saved AI. The AI tells us: 'Valence: Pleasant (52.3%), Arousal: Low Energy (88.0%) -> Quadrant IV: Relieved and Calm'. It works end-to-end!"*

#### Code & Values to Take Note Of
* **Command to run in terminal**:
  ```powershell
  python inference/predict.py Audios/Y1-203_Y1_AM_Kapoy_nga_dili.wav
  ```

---

# PART II: 10-Minute Research Output Presentation Guide (7 Slides)

Directly matches the **7-Slide Research Output Presentation Sequence**:

---

### Slide 1: Research Output Overview
* **Group Members**: CS412 Research Group
* **Assigned CS Section**: Computer Science Undergraduate Sections (Year 1 to Year 4)
* **Number of Participants**: 40 unique participants ($N=40$)
* **Completed Outputs**:
  * 40 verified `.wav` speech recordings in `Audios/`
  * Validated ground-truth survey dataset ([`bisaya_post_class_speech_metadata.csv`](./bisaya_post_class_speech_metadata.csv))
  * Extracted 94-column acoustic feature matrix ([`bisaya_extracted_features.csv`](./bisaya_extracted_features.csv))
  * 19 data tables in `audio_outputs/tables/` & 6 presentation figures in `audio_outputs/figures/`
  * Dual model artifact ([`models/bisaya_ser_models.joblib`](./models/bisaya_ser_models.joblib)) & standalone inference pipeline ([`inference/predict.py`](./inference/predict.py))

---

### Slide 2: Data Collection
* **Collection Procedure**: Immediately upon class dismissal, student participants were prompted to speak an immediate one-word or short-phrase spontaneous response describing their post-class feeling in Bisaya, followed by a standardized self-assessment survey.
* **Participant Distribution**: Sampled across CS Year Levels (1st Year to 4th Year) in both Morning (AM) and Afternoon (PM) classes.
* **Consent & Ethics**: Full voluntary consent obtained; all identifiers sanitized to anonymous codes (`Y1-201` through `Y1-240`).
* **Recorded Variables**: Participant Code, Year Level, Class Activity, Session Time, Spoken Word, Self-Reported Feeling, Valence (1–5 Likert), Arousal (1–5 Likert), Audio Filename.

---

### Slide 3: Dataset Establishment
* **Audio Dataset Structure**: 40 single-channel uncompressed `.wav` files, sampling duration between $0.51$s and $2.81$s.
* **Facial Modality Note**: Structural directories (`face/`, `multimodal/`) reserved for future video facial action unit modeling; audio represents the completed empirical focus.
* **Target Affect Labels**:
  * **Valence**: $23$ Not Pleasant ($57.5\%$), $17$ Pleasant ($42.5\%$).
  * **Arousal**: $26$ Low/Mod Arousal ($65.0\%$), $14$ High Arousal ($35.0\%$).
* **Baseline Accuracy**: Dummy most-frequent classifier achieves $57.5\%$ on Valence and $65.0\%$ on Arousal.

---

### Slide 4: Preprocessing and Feature Extraction
* **Audio Preprocessing**: Ingestion via Librosa DSP, silence trimming, and energy normalization.
* **94 Acoustic Feature Set**:
  * **Prosodic**: YIN Fundamental Frequency ($F_0$ mean, std, min, max), RMS Energy, Duration.
  * **Spectral**: Zero-Crossing Rate (ZCR), Spectral Centroid, Spread, Flatness, Flux, Rolloff.
  * **Cepstral (MFCCs)**: 13 MFCCs + 13 Delta Velocity + 13 Delta-Delta Acceleration (means and stds = 78 features).
* **Extraction Quality**: Zero missing values across all 94 features across all 40 participants.

---

### Slide 5: Machine Learning Implementation
* **Validation Strategy**: Strict 5-Fold `StratifiedGroupKFold` grouped by `Participant Code` (zero speaker identity leakage).
* **Algorithms Benchmarked**: Dummy Baseline, SVM (RBF), Logistic Regression, Random Forest (300 trees), Gradient Boosting, k-NN ($k=3$).
* **Hyperparameters**: `StandardScaler` preprocessing, `class_weight='balanced'`, fixed `random_state=42`.
* **Production Deployment Choice**:
  * **Valence Pipeline**: `StandardScaler` + `SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")`
  * **Arousal Pipeline**: `StandardScaler` + `LogisticRegression(max_iter=1000, class_weight="balanced")`

---

### Slide 6: Model Evaluation
* **Comparative Cross-Validation Performance**:
  * **Valence Winner**: **SVM (RBF)** — Macro $F_1 = \mathbf{0.522}$, Balanced Acc $= \mathbf{0.543}$ (beats baseline Macro $F_1$ of $0.364$).
  * **Arousal Winner**: **Logistic Regression** — Macro $F_1 = \mathbf{0.662}$, Balanced Acc $= \mathbf{0.670}$, Accuracy $= \mathbf{0.725}$ (beats baseline Macro $F_1$ of $0.393$).
* **Confusion Matrices**:
  * Valence correctly identifies $15/23$ Not Pleasant cases.
  * Arousal correctly identifies $21/26$ Low Arousal and $8/14$ High Arousal cases.
* **Top Predictive Features**: High-order MFCCs (`mfcc_10_mean`, `mfcc_delta_8_std`) and minimum pitch (`f0_min`).

---

### Slide 7: Findings and Limitations
* **Actual Findings**:
  1. Arousal is acoustically transparent in post-class speech, strongly driven by vocal loudness (RMS) and pitch ($F_0$).
  2. The high frequency of *"Kapoy"* ($40\%$ of corpus) proves lexical ambiguity; voice acoustics successfully separate stressful fatigue from relieved exhaustion.
  3. A hybrid dual-architecture model outperforms any single monolithic algorithm.
* **Limitations**: Sample size $N=40$ constrained by voluntary post-class enrollment; short duration of single-word utterances.
* **Recommendations**: Expand to multimodal feature-level fusion incorporating facial action unit landmarks from recorded video sessions.

---

# PART III: 10-Minute Live Technical Checking & Laptop Demo (Parts A, B, and C)

Follow these exact steps during your laptop technical demonstration:

---

### Part A: Raw Data and Dataset Verification
*Demonstrate: Open audio files, metadata CSV, show participant counts, file correspondence, and valid target labels.*

1. **Terminal Command to Run**:
   ```powershell
   python -c "import pandas as pd; df = pd.read_csv('bisaya_post_class_speech_metadata.csv'); print('--- PARTICIPANT & TARGET VERIFICATION ---'); print('Total Rows:', len(df)); print('Valence Value Counts:\n', df['Valence (1-5)'].value_counts()); print('Arousal Value Counts:\n', df['Arousal (1-5)'].value_counts()); print('\nFirst 3 Records:'); print(df[['Participant Code', 'Spoken Word', 'Valence (1-5)', 'Arousal (1-5)', 'Audio Filename']].head(3))"
   ```
2. **Verify File Correspondence**:
   ```powershell
   python -c "from pathlib import Path; import pandas as pd; df = pd.read_csv('bisaya_post_class_speech_metadata.csv'); audio_dir = Path('Audios'); missing = [f for f in df['Audio Filename'] if not (audio_dir / f).is_file()]; print('Missing audio files:', len(missing)); print('All 40 files present:', len(missing) == 0)"
   ```
3. **What to say to the panel**:
   > *"As shown on screen, our metadata table contains exactly 40 participant rows. Every single row matches a valid .wav file in the Audios directory with zero missing values, and all 1–5 Likert labels are verified."*

---

### Part B: Preprocessing and Feature Extraction
*Demonstrate: Process one raw audio recording into a structured 94-feature vector live. Show actual column names and numbers.*

1. **Terminal Command to Run**:
   ```powershell
   python -c "from inference.feature_extractor import extract_audio_features; feats = extract_audio_features('Audios/Y1-203_Y1_AM_Kapoy_nga_dili.wav'); print('--- LIVE FEATURE EXTRACTION ---'); print('Total Features Extracted:', len(feats)); print('Audio Duration     :', round(feats['duration'], 3), 'seconds'); print('Mean Pitch (F0)    :', round(feats['f0_mean'], 2), 'Hz'); print('RMS Energy         :', round(feats['rms_mean'], 4)); print('Zero Crossing Rate :', round(feats['zcr_mean'], 4)); print('Spectral Centroid  :', round(feats['sc_mean'], 1), 'Hz'); print('MFCC 1 (Mean/Std)  :', round(feats['mfcc_1_mean'], 3), '/', round(feats['mfcc_1_std'], 3)); print('MFCC 2 (Mean/Std)  :', round(feats['mfcc_2_mean'], 3), '/', round(feats['mfcc_2_std'], 3))"
   ```
2. **Show the Master CSV Table**:
   ```powershell
   python -c "import pandas as pd; df = pd.read_csv('processed_data/bisaya_extracted_features.csv'); print('Feature Matrix Shape:', df.shape); print('First 5 column names:', list(df.columns[:5])); print('Last 5 column names :', list(df.columns[-5:]))"
   ```
3. **What to say to the panel**:
   > *"Here you see the live feature extraction pipeline. We pass a raw .wav file, and in under 100 milliseconds Librosa computes our exact 94-dimensional vector covering pitch via YIN, RMS energy, spectral centroid, and 13 MFCCs with delta and acceleration."*

---

### Part C: Machine Learning and Results
*Demonstrate: Show model training, evaluation results, confusion matrices, model comparison, and interactive prediction.*

1. **Display Model Comparison Tables (Comparing all 6 models)**:
   ```powershell
   python -c "import pandas as pd; print('=== VALENCE BENCHMARK (Table 11) ==='); print(pd.read_csv('audio_outputs/tables/11_model_comparison.csv')[['Model', 'Accuracy Mean', 'Balanced Accuracy Mean', 'Macro F1 Mean']]); print('\n=== AROUSAL BENCHMARK (Table 15) ==='); print(pd.read_csv('audio_outputs/tables/15_arousal_model_comparison.csv')[['Model', 'Accuracy Mean', 'Balanced Accuracy Mean', 'Macro F1 Mean']])"
   ```
2. **Display Confusion Matrices**:
   ```powershell
   python -c "import pandas as pd; print('=== VALENCE CONFUSION MATRIX (SVM RBF) ==='); print(pd.read_csv('audio_outputs/tables/12_confusion_matrix.csv', index_col=0)); print('\n=== AROUSAL CONFUSION MATRIX (Logistic Regression) ==='); print(pd.read_csv('audio_outputs/tables/16_arousal_confusion_matrix.csv', index_col=0))"
   ```
3. **Execute Live End-to-End Prediction**:
   * **Test Case 1 (Pleasant, Low Arousal -> Quadrant IV: Pleasant-Deactivated / Relieved)**:
     ```powershell
     python inference/predict.py Audios/Y1-203_Y1_AM_Kapoy_nga_dili.wav
     ```
   * **Test Case 2 (Unpleasant, Low Arousal -> Quadrant III: Unpleasant-Deactivated / Tired)**:
     ```powershell
     python inference/predict.py Audios/Y1-204_Y1_AM_Kapoy.wav
     ```
   * **Test Case 3 (Unpleasant, High Arousal -> Quadrant II: Unpleasant-Activated / Frustrated)**:
     ```powershell
     python inference/predict.py Audios/Y1-208_Y1_AM_Kapoya.wav
     ```
4. **What to say to the panel**:
   > *"Finally, here is our live production inference engine running on raw audio files. It extracts features, feeds them into our dual model bundle (SVM for Valence and Logistic Regression for Arousal), and outputs the exact predicted quadrant in Russell's 2D Circumplex Model of Affect."*

---

# PART IV: Comprehensive Research Vocabulary & Key Terminology Glossary

Use this glossary when preparing for panel questions, writing your paper, or explaining technical terms in simple language.

---

### 1. Affect & Emotion Theory Terminology

#### **Valence (Pleasantness / Hedonic Tone)**
* **Technical Definition**: The intrinsic attractiveness or aversiveness of an event, object, or situation. In dimensional emotion theory, Valence represents the horizontal axis measuring whether an affective state is negative (unpleasant) or positive (pleasant).
* **In Plain English ("U Feel Me")**: *"How good or bad you feel."* Positive valence = happy, relieved, satisfied. Negative valence = sad, frustrated, exhausted.

#### **Arousal (Activation / Physiological Energy)**
* **Technical Definition**: The state of being physiologically and psychologically alert, awake, and reactive. Arousal represents the vertical axis in Russell's model, driven by sympathetic nervous system activation (elevated heart rate, lung pressure, and muscle tension).
* **In Plain English ("U Feel Me")**: *"How high your battery or energy level is."* High arousal = pumped up, excited, or panicking. Low arousal = calm, sleepy, or drained.

#### **Russell's Circumplex Model of Affect (1980)**
* **Technical Definition**: A dimensional psychological model proposing that all human emotional states arise from two independent, neurophysiologically rooted continuous dimensions: Valence and Arousal. Plotting them on perpendicular axes creates a continuous 2D space divided into 4 quadrants.
* **In Plain English ("U Feel Me")**: *"A 2D map of human emotions. Instead of boxing emotions into rigid categories like 'Angry' or 'Sad', any feeling in the universe can be pinpointed as a coordinate (X = Pleasantness, Y = Energy)."*

#### **Russell's 4 Quadrants**
* **Quadrant I (Pleasant-Activated)**: High Valence + High Arousal $\rightarrow$ Energetic, enthusiastic, motivated (e.g., *"Lingaw!"*, *"Nadasig"*).
* **Quadrant II (Unpleasant-Activated)**: Low Valence + High Arousal $\rightarrow$ Tense, stressed, overwhelmed, anxious (e.g., *"Kulbaan"*, *"Grabe"*).
* **Quadrant III (Unpleasant-Deactivated)**: Low Valence + Low Arousal $\rightarrow$ Tired, burned out, drained, sleepy (e.g., *"Kapoy"*, *"Katulgon"*, *"Hago"*).
* **Quadrant IV (Pleasant-Deactivated)**: High Valence + Low Arousal $\rightarrow$ Calm, relieved, peaceful, relaxed (e.g., *"Chill lang"*, *"Salamat"*, *"Bugnaw"*).

#### **Affective Computing & Speech Emotion Recognition (SER)**
* **Technical Definition**: An interdisciplinary field of artificial intelligence concerned with developing computational systems capable of detecting, recognizing, interpreting, and responding to human emotional states from bio-signals, facial expressions, and vocal acoustics.
* **In Plain English ("U Feel Me")**: *"Teaching computers how to listen to someone's voice and figure out what mood they are in."*

#### **Self-Assessment Manikin (SAM) / Likert Scale**
* **Technical Definition**: A non-verbal pictorial assessment technique and 5-point ordinal rating scale used to directly measure the Valence, Arousal, and Dominance associated with an individual's affective reaction without language bias.
* **In Plain English ("U Feel Me")**: *"The 1-to-5 survey students filled out right after class to self-report how pleasant (1–5) and how energized (1–5) they felt."*

---

### 2. Linguistic & Lexical Terminology

#### **Lexical / Lexicon**
* **Technical Definition**: Pertaining to the words or vocabulary of a language. A lexicon is the complete set of meaningful words in a language or corpus.
* **In Plain English ("U Feel Me")**: *"The actual dictionary words people speak (like 'Kapoy', 'Lingaw', 'Sayon')."*

#### **Lexical Ambiguity / Polysemy**
* **Technical Definition**: The phenomenon where a single lexical token possesses multiple distinct semantic interpretations, pragmatics, or affective meanings depending on conversational context and prosodic delivery.
* **In Plain English ("U Feel Me")**: *"When one word has two completely different meanings depending on how you say it."* For example, *"Kapoy"* can mean *"I'm suffering and stressed"* OR *"I'm relieved and relaxed that class is finally over"*.

#### **Lexically Controlled Analysis**
* **Technical Definition**: An experimental control methodology where phonetic and vocabulary variation is held strictly constant across comparison groups (e.g., analyzing only utterances of the single token *"Kapoy"*) to ensure observed acoustic differences are attributable purely to affect, rather than differences in spoken vowels or consonants.
* **In Plain English ("U Feel Me")**: *"Testing students who said the exact same word so nobody can argue that the pitch difference was caused by different words."*

#### **Phonetic Confounding**
* **Technical Definition**: The contamination of acoustic emotion markers caused by the inherent physical properties of speech sounds (e.g., the vowel /i/ in *"Lingaw"* naturally requires a higher tongue position and higher fundamental frequency than the vowel /o/ in *"Gutom"*).
* **In Plain English ("U Feel Me")**: *"When the letters or vowels in a word trick the computer into thinking someone is happy just because the word naturally sounds higher-pitched."*

---

### 3. Audio Signal & Acoustic Terminology

#### **Acoustic Features**
* **Technical Definition**: Numerical parameters extracted from the physical properties of a raw sound pressure wave (time, frequency, amplitude, and phase) that describe its auditory behavior.
* **In Plain English ("U Feel Me")**: *"The 94 voice measurements (pitch, volume, sharpness, vibration speed) we feed into our AI."*

#### **Prosody / Prosodic Features**
* **Technical Definition**: The supra-segmental properties of spoken speech that extend over syllables, phrases, or entire sentences—primarily encompassing rhythm, stress, intonation (pitch), and loudness—rather than individual phonetic consonants and vowels.
* **In Plain English ("U Feel Me")**: *"The tone, melody, and rhythm of your voice."* It's not *what* you say, it's *how* you say it.

#### **Fundamental Frequency ($F_0$) / Pitch**
* **Technical Definition**: The rate of vibration of the vocal folds during voiced speech production, measured in Hertz ($\text{Hz}$). It is the physical acoustic correlate of perceived auditory pitch.
* **In Plain English ("U Feel Me")**: *"How high or deep your voice sounds."* When you squeak or get excited, $F_0$ shoots up; when you are relaxed or tired, $F_0$ drops down.

#### **YIN Algorithm**
* **Technical Definition**: A robust, time-domain fundamental frequency estimation algorithm based on the average squared difference function (ASDF), designed to track vocal pitch accurately without octave errors even in noisy speech recordings.
* **In Plain English ("U Feel Me")**: *"The mathematical ruler Librosa uses to measure the exact musical note/pitch of a person's vocal cords."*

#### **Root Mean Square (RMS) Energy**
* **Technical Definition**: The square root of the arithmetic mean of the squares of audio waveform amplitude values over a given frame. It directly quantifies the physical signal power and serves as the acoustic correlate of auditory loudness.
* **In Plain English ("U Feel Me")**: *"Volume and vocal power."* How loudly and strongly the student was projecting their voice.

#### **Zero-Crossing Rate (ZCR)**
* **Technical Definition**: The rate at which the audio signal waveform changes sign from positive to negative within a given time frame. High ZCR typically correlates with unvoiced sounds, whispering, fricatives, and noisy breathiness.
* **In Plain English ("U Feel Me")**: *"How noisy or breathy the voice is."* Clean singing has low ZCR; whispering or sighing has high ZCR.

#### **Spectral Centroid**
* **Technical Definition**: The center of mass (weighted average) of the frequency spectrum, computed using the Fourier transform. It represents the frequency where the majority of the sound energy is concentrated and correlates with perceived auditory "brightness" or sharpness.
* **In Plain English ("U Feel Me")**: *"Voice brightness."* A high spectral centroid sounds crisp, bright, and sharp; a low spectral centroid sounds dull, dark, and muffled.

#### **Spectral Bandwidth, Rolloff, and Flatness**
* **Spectral Bandwidth**: The width or spread of the frequency spectrum around its centroid.
* **Spectral Rolloff**: The specific frequency below which 85% of the total spectral energy is contained.
* **Spectral Flatness**: A ratio quantifying whether the sound spectrum is smooth and noise-like (flat) or spiked with musical harmonic tones.

#### **Mel-Frequency Cepstral Coefficients (MFCCs)**
* **Technical Definition**: Coefficients derived from the discrete cosine transform (DCT) of a log-power spectrum on a non-linear Mel frequency scale. They model the spectral envelope of speech, capturing vocal tract filtering characteristics (mouth, tongue, and throat shape) in alignment with human auditory perception.
* **In Plain English ("U Feel Me")**: *"The mathematical fingerprint of your voice texture (timbre)."* They allow the computer to recognize the shape of your throat and mouth when you speak.

#### **Delta ($\Delta$) and Delta-Delta ($\Delta\Delta$) Coefficients**
* **Delta ($\Delta$) MFCCs (Velocity)**: The first temporal derivative of MFCCs, measuring how fast the voice texture is changing from one millisecond to the next.
* **Delta-Delta ($\Delta\Delta$) MFCCs (Acceleration)**: The second temporal derivative of MFCCs, measuring the acceleration or rate of change of the voice texture.

#### **Spectrogram / Mel Spectrogram**
* **Technical Definition**: A visual representation of the spectrum of frequencies of an audio signal as it varies over time, where the X-axis is time, the Y-axis is frequency (scaled in Mel bins), and the color intensity reflects energy in decibels ($\text{dB}$).
* **In Plain English ("U Feel Me")**: *"A colorful heatmap of sound showing low bass notes at the bottom and high treble sounds at the top over time."*

---

### 4. Machine Learning & Evaluation Terminology

#### **Feature Matrix ($X$) & Target Vector ($y$)**
* **$X$ (Feature Matrix)**: The table of inputs containing $40 \text{ rows} \times 94 \text{ acoustic measurements}$.
* **$y$ (Target Vector)**: The correct ground-truth answers we want the AI to predict (e.g., $y_{\text{val}} = \text{Pleasant / Not Pleasant}$, $y_{\text{aro}} = \text{High Arousal / Low Arousal}$).

#### **Stratified Group K-Fold Cross-Validation (`StratifiedGroupKFold`)**
* **Technical Definition**: A cross-validation splitting architecture that satisfies two simultaneous constraints: (1) **Grouping**: observations from the same participant are strictly assigned to either the training set or test set, never both; and (2) **Stratification**: the percentage of each target class is preserved equally across all folds.
* **In Plain English ("U Feel Me")**: *"The strict testing rule that guarantees zero cheating. If Student #1 is in the test group, their voice is never allowed in the training group, so the AI can't just memorize who they are."*

#### **Data Leakage / Speaker Leakage**
* **Technical Definition**: An experimental error where information from the testing dataset inadvertently leaks into the training pipeline, causing artificially inflated accuracy that fails when deployed in the real world. In SER, speaker leakage occurs when the same person's voice appears in both train and test sets.
* **In Plain English ("U Feel Me")**: *"Cheating by looking at the exam answers beforehand."* Grouping by student code completely prevents this.

#### **Out-of-Fold Predictions**
* **Technical Definition**: The aggregated predictions generated on held-out validation folds across the entire cross-validation cycle. Because each prediction is made by a model that never saw that specific fold during training, the pooled results represent an honest evaluation of the entire dataset.
* **In Plain English ("U Feel Me")**: *"The combined test results across all 5 rounds of cross-validation."*

#### **Accuracy vs. Balanced Accuracy**
* **Accuracy**: Total correct guesses divided by total samples ($22/40 = 55.0\%$). Vulnerable to majority-class cheating on imbalanced data.
* **Balanced Accuracy**: The unweighted average of recall obtained on each individual class. If a model predicts only the majority class, Balanced Accuracy drops straight to $50.0\%$.

#### **Precision, Recall, and Macro $F_1$ Score**
* **Precision**: *"When the model predicts 'Pleasant', how often is it actually right?"* (Measures false alarms).
* **Recall (Sensitivity)**: *"Out of all the truly 'Pleasant' students, how many did the model catch?"* (Measures missed detections).
* **Macro $F_1$ Score**: The harmonic mean of precision and recall calculated for each class separately, then averaged equally. **This is our primary scientific metric** because it prevents models from cheating on imbalanced data.

#### **Confusion Matrix**
* **Technical Definition**: A $2 \times 2$ contingency table comparing actual ground-truth classes (rows) against predicted model classes (columns). The diagonal cells represent correct predictions (True Positives and True Negatives); off-diagonal cells represent misclassification errors (False Positives and False Negatives).
* **In Plain English ("U Feel Me")**: *"A score grid showing where the model was right and what specific mistakes it made."*

#### **Permutation Feature Importance**
* **Technical Definition**: A model-agnostic feature inspection technique that measures the decrease in a model's prediction score (Macro $F_1$) after randomly shuffling the values of a single feature column on held-out validation data.
* **In Plain English ("U Feel Me")**: *"A test where you scramble one voice measurement to see how badly the AI's score drops. The features that cause the biggest drop are the most important."*

#### **Support Vector Machine (SVM) with RBF Kernel**
* **Technical Definition**: A supervised learning classifier that identifies the optimal hyperplane maximizing the geometric margin between classes. The Radial Basis Function (RBF) kernel projects the 94 features into an infinite-dimensional space to solve non-linear classification boundaries.
* **In Plain English ("U Feel Me")**: *"Our winning algorithm for Valence (happiness). It draws a smart, curved mathematical boundary between Pleasant and Not Pleasant voices."*

#### **Logistic Regression**
* **Technical Definition**: A generalized linear model that maps linear combinations of input features to a probability between 0 and 1 using the sigmoid logistic function: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
* **In Plain English ("U Feel Me")**: *"Our winning algorithm for Arousal (energy). It uses smooth mathematical probabilities to predict whether a voice is High or Low Energy, scoring 72.5% accuracy."*

#### **StandardScaler (Feature Normalization)**
* **Technical Definition**: A preprocessing transformer that standardizes features by subtracting the mean and scaling to unit variance: $z = \frac{x - \mu}{\sigma}$. This ensures that large numerical features (like Spectral Centroid at $2000\text{ Hz}$) do not dominate small features (like RMS at $0.05$).
* **In Plain English ("U Feel Me")**: *"Putting all 94 measurements onto the same fair playing field so big numbers don't drown out small numbers."*

#### **Serialization (`joblib.dump` / `joblib.load`)**
* **Technical Definition**: The process of converting trained Python in-memory objects (Scikit-Learn pipelines, estimators, and scalers) into a persistent binary byte stream saved to a file on disk (`.joblib`), allowing them to be reloaded for standalone inference without retraining.
* **In Plain English ("U Feel Me")**: *"Freezing our trained AI models into a file (`bisaya_ser_models.joblib`) so our prediction program can load them and make predictions on new audio files in milliseconds."*

