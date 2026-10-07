# PowerPoint Outline

**Project:** Bisaya Post-Class Speech and Affective State Recognition Among Computer Science Students  
**Course:** CS412 Research Mini-Project  
**Length:** 10 slides total, including the title and conclusion.  
**Basis:** [Instructions.md](Instructions.md), the current dataset, exported results, and inference implementation.

Use the slide content below on screen. Keep presenter notes and detailed sources in PowerPoint speaker notes, with brief source footers where useful. References and questions fit within these 10 slides. No additional or backup slides are needed.

## Slide 1 — Project Title and Purpose

- **Bisaya Post-Class Speech and Affective State Recognition Among Computer Science Students**
- CS412 Research Mini-Project
- [Group number, members, instructor, and presentation date]
- Purpose: Explore whether short, naturally spoken Bisaya responses contain acoustic information related to students' self-reported affect after class.

**Visual:** A simple title layout.

**Presenter note:** Introduce the work as an exploratory audio study. The selected word does not automatically identify the student's emotion. The study uses independent self-reports as the reference.

**Source:** Instructions.md, sections I–III and XX.

## Slide 2 — Research Questions and Scope

- What affective states do students report after class?
- How does affect vary with year level, class activity, and session?
- What acoustic characteristics appear across affective states?
- How well can acoustic features classify self-reported affect?
- Which candidate model performs best?

**Scope note on slide:** Current sample: 40 first-year students' observations after morning classes. Year-level and session differences cannot be evaluated.

**Visual:** A concise numbered list of the five research questions.

**Presenter note:** The project addresses all five questions in its design, but the current data support only descriptive activity comparisons for the educational-context question.

**Source:** Instructions.md, sections II–III; DATA_QUALITY_REPORT.md.

## Slide 3 — Collection, Ethics, and Dataset Quality

- **Collection:** Approach eligible students immediately after class, obtain consent, and record the educational context.
- **Natural response:** Record a short Bisaya response without emotional coaching, followed by an independent feeling description and valence/arousal ratings from 1–5.
- **Ethics:** Participation is voluntary, with no academic penalty for declining. Use participant codes and restrict access to identifiable recordings.
- **Dataset:** 40 first-year morning observations: 16 Lecture, 15 Laboratory, and 9 Quiz.
- **Quality checks:** 40/40 WAV files matched and opened successfully, with no duplicate participant codes or missing required metadata.
- **Remaining limits:** Mixed recording settings and incomplete collection documentation require cautious interpretation.

**Visual:** A short numbered collection sequence above a compact quality summary showing 40 records, 40 matched/readable WAV files, and zero duplicate codes or missing required fields. Keep the detailed checks below in speaker notes.

### Presenter notes — Collection procedure

1. **Eligibility and timing:** The protocol calls for students from the assigned CS year level who have just completed a scheduled class. Each participant ordinarily contributes one observation unless an approved design permits repeated observations. The current dataset contains 40 distinct participant codes, all first year and morning.
2. **Consent before recording:** Explain the purpose, requested response, data collected, storage arrangements, and intended academic use. Obtain consent before starting any recording. Do not pressure students to participate.
3. **Educational context:** Record year level, the main activity in the class that just ended, and session. Apply a consistent morning/afternoon–evening boundary. That boundary is not documented in the current repository.
4. **Natural speech:** Ask for a short Bisaya reaction in the student's natural voice. Do not ask the student to sound happy, sad, angry, or tired, or to repeat the word until it sounds emotional. The protocol recommends comparable equipment/settings, a reasonably quiet location, suitable microphone distance, and no voice-enhancement effects.
5. **Independent self-report:** Immediately afterward, ask how the student actually feels and obtain separate valence and arousal ratings. The chosen word and the researcher's perception of the voice must not determine the labels.
6. **Record linkage:** Store the participant code, context, spoken word, feeling description, ratings, and corresponding WAV filename. Use the filename to locate audio and the current `Spoken Word` field for lexical analysis. Document any normalization of spoken-word labels.

### Presenter notes — Ethics and privacy

- **Voluntary participation:** Students may refuse or discontinue participation without academic penalty. Explain this before collecting data.
- **Pseudonymization:** Use codes rather than names or student numbers in filenames, feature tables, and reports. Voice recordings can still identify people, so participant codes do not make the recordings fully anonymous.
- **Protected records:** Keep consent forms and any identity-to-code mapping outside the analysis dataset and public repository. The project safeguards call for separate, encrypted, access-controlled storage of the mapping and an institutionally defined retention/deletion schedule.
- **Authorized use:** Consent and institutional approval must cover the intended collection, storage, analysis, presentation, and sharing. Report aggregate findings and avoid exposing individual participants. Audio consent must not be assumed to authorize video or face processing because the supplied protocol specifies audio.
- **Evidence status:** These are protocol requirements and documented safeguards. The repository does not establish that consent, approval, secure storage, or retention arrangements have been completed. Present the group's actual status accurately rather than describing every requirement as a verified action.

### Presenter notes — Dataset quality findings

| Check | Documented result | Interpretation |
| --- | --- | --- |
| Observation count | 40 metadata rows and 40 WAV files | Meets the numerical group minimum; the class-wide target is 280 |
| Participant codes | 40 unique codes, no duplicates | No repeated code in the current table; this does not independently verify participant identity |
| File linkage | 40/40 exact filename matches; no unmatched WAV files | Every metadata record links to an available recording |
| File readability | All 40 WAV files opened successfully | Confirms readable files, not speech clarity or freedom from noise |
| Metadata completeness | No missing required fields; optional `Notes` empty in all 40 rows | Required fields are populated, but completeness alone does not establish accuracy |
| Source audio | Stereo, 16-bit; 28 files at 48 kHz and 12 at 44.1 kHz | Source settings vary and need consistent preprocessing |
| Recording duration | 0.407–2.547 seconds; mean approximately 0.870 seconds | Very short responses provide limited speech material |
| Feature artifact | 40 rows, 94 acoustic features, 109 total columns; finite-value check reported | The documented feature table has the expected structure |
| Sample coverage | First year only, morning only, three class activities | Year-level/session effects cannot be estimated from this sample |
| Collection documentation | Device, microphone distance, ambient noise, and session boundary not documented | Their influence cannot be assessed fully |

**Interpretation to explain aloud:** “The dataset is structurally complete for an exploratory group analysis: each record has its required fields and a readable, matching audio file. Valid research observations also depend on consent, protocol compliance, and a review of speech quality. Resampling standardizes the audio format, but it does not remove differences in microphones, background noise, or collection conditions.”

**Sources:** Instructions.md, sections IV–XII and XVIII; DATA_QUALITY_REPORT.md; PRIVACY_AND_ETHICS.md; audio_outputs/tables/05_activity_by_valence.csv.

## Slide 4 — Audio Processing, Features, and Targets

- Convert recordings to mono at 22.05 kHz and trim silence with `top_db=25`.
- Extract 94 features: duration, pitch, RMS energy, zero-crossing rate, spectral measures, and MFCCs with their derivatives.
- Use one feature row per recording, joined to metadata by participant code.
- Exclude spoken word, year level, activity, and session from model predictors.

| Target | Ratings 4–5 | Ratings 1–3 |
| --- | --- | --- |
| Valence | Pleasant: 17 | Not Pleasant: 23 |
| Arousal | High: 14 | Low/Moderate: 26 |

**Visible qualification:** Binary grouping is an implementation choice. The lower groups include neutral valence and moderate arousal.

**Visual:** A compact processing sequence and the two-row target table.

**Presenter note:** The 94 features consist of 1 duration, 4 pitch, 3 RMS, 2 zero-crossing, 6 spectral, and 78 MFCC/delta/delta-delta summaries. The feature CSV contains 40 rows and 109 total columns, including 15 metadata or derived-target fields. The protocol does not prescribe these exact preprocessing settings or binary thresholds.

**Sources:** bisaya_extracted_features.csv; inference/feature_extractor.py; export_audio_outputs.py; Instructions.md, sections IX and XIV–XV.

## Slide 5 — Reported Affect and Educational Context

- Valence: 10 Unpleasant, 13 Neutral, and 17 Pleasant/Very Pleasant observations.
- Arousal: 3 Very Low, 10 Low, 13 Moderate, and 14 High observations.
- Pleasant ratings by activity: Lecture 50.0%, Laboratory 40.0%, and Quiz 33.3%.
- “Kapoy” appears in 20 of 40 observations, followed by “Lingaw” (3) and “Sayon” (2).

**Visible qualification:** These are descriptive patterns from a small sample, not evidence that class activity causes an affective state.

**Visual:** Use the relevant panels from [01_affect_and_context_overview.png](audio_outputs/figures/01_affect_and_context_overview.png). Keep word frequencies as a short text annotation.

**Presenter note:** The Pleasant category here combines 16 Pleasant and 1 Very Pleasant rating. No participant reported Very Unpleasant valence or Very High arousal. Activity group sizes differ, so compare percentages together with the counts on slide 3. Use the metadata's `Spoken Word` for lexical grouping, not historical words inside filenames. Document the updated labels and observed words beyond those listed in the protocol.

**Sources:** audio_outputs/tables/03_valence_distribution.csv, 04_arousal_distribution.csv, 05_activity_by_valence.csv, and 06_spoken_word_frequency.csv; README.md.

## Slide 6 — Acoustic Findings and Same-Word Analysis

- Across the full dataset, mean estimated pitch is 158.9 Hz for Unpleasant, 190.1 Hz for Neutral, and 200.2 Hz for Pleasant responses.
- The exact “Kapoy” subset contains 20 observations: 8 Pleasant, 9 Neutral, and 3 Unpleasant.
- Within “Kapoy,” mean pitch is 201.9 Hz for Pleasant and 178.6 Hz for Not Pleasant responses.
- The same word occurs with different self-reported states. Acoustic comparisons remain exploratory.

**Visual:** Use the pitch comparison from [05_exact_kapoy_comparison.png](audio_outputs/figures/05_exact_kapoy_comparison.png), with the subset counts visible.

**Presenter note:** This covers the required lexically controlled analysis. The binary subset contains 8 Pleasant and 12 Not Pleasant responses and excludes different exact labels such as “Kapoya.” Group means do not establish statistical significance, a reliable emotion rule, or successful classification within this subset. Speaker and recording differences may contribute to the patterns.

**Sources:** Instructions.md, section XVII; audio_outputs/tables/08_acoustic_summary_by_valence.csv, 09_exact_kapoy_acoustic_summary.csv, and 10_exact_kapoy_observations.csv.

## Slide 7 — Machine Learning and Evaluation

- Compare RBF SVM, k-NN, Random Forest, Gradient Boosting, and Logistic Regression with a majority-class baseline.
- Use five-fold stratified cross-validation grouped by participant code.
- Fit standardization inside the applicable training-fold pipelines.
- Rank candidates by mean macro F1 and also report accuracy, balanced accuracy, and confusion matrices.
- Model selection and evaluation use the same small dataset, without an external test set.

**Visual:** A simple five-fold diagram showing that a participant stays within one held-out fold.

**Presenter note:** Participant grouping prevents a speaker's recordings from appearing in both training and validation. The current dataset has one observation per participant. Macro F1 weights the two classes equally. Choosing the best model using the same folds limits how strongly we can interpret the reported performance.

**Sources:** Instructions.md, sections XV–XVI; export_audio_outputs.py.

## Slide 8 — Model Results and Classification Errors

| Target and model | Mean accuracy | Mean macro F1 |
| --- | ---: | ---: |
| Valence: RBF SVM | 55.0% | 0.522 |
| Valence: majority baseline | 57.5% | 0.364 |
| Arousal: Logistic Regression | 72.5% | 0.662 |
| Arousal: majority baseline | 65.0% | 0.393 |

**Out-of-fold confusion matrices:** Rows are actual classes, columns are predicted classes.

| Valence: SVM | Pleasant | Not Pleasant |
| --- | ---: | ---: |
| Pleasant | 7 | 10 |
| Not Pleasant | 8 | 15 |

| Arousal: Logistic Regression | High | Low/Moderate |
| --- | ---: | ---: |
| High | 8 | 6 |
| Low/Moderate | 5 | 21 |

**Takeaway on slide:** Valence SVM leads by macro F1 but falls below baseline accuracy. Arousal Logistic Regression exceeds baseline accuracy by 7.5 percentage points.

**Visual:** A compact results table above two small confusion matrices. Keep the remaining comparison details in speaker notes.

**Presenter note:** For valence, candidate accuracy/macro F1 results are k-NN 57.5%/0.510, Random Forest 52.5%/0.466, Gradient Boosting 50.0%/0.442, and Logistic Regression 45.0%/0.429. For arousal, they are k-NN 72.5%/0.644, Random Forest 70.0%/0.627, SVM 67.5%/0.622, and Gradient Boosting 60.0%/0.577. The leading valence model has balanced accuracy 0.543 and macro F1 fold SD 0.097. The leading arousal model has balanced accuracy 0.670 and macro F1 fold SD 0.222. Valence Pleasant recall is only 41.2%. These findings remain exploratory. Mean fold macro F1 differs from pooled out-of-fold macro F1, which is 0.531 for valence and 0.693 for arousal.

**Sources:** audio_outputs/tables/11_model_comparison.csv through 13_classification_report.csv, and 15_arousal_model_comparison.csv through 17_arousal_classification_report.csv.

## Slide 9 — Prototype and Project Outputs

- The prototype accepts a WAV file, extracts 94 features, and predicts valence and arousal.
- A combined affect region summarizes the two predicted dimensions.
- Project outputs include metadata, the feature dataset, analysis code, evaluation tables, figures, and reports.
- Audio analysis is implemented. Face and multimodal feature datasets remain future work.

**Visible qualification:** The saved prototype uses SVM for both targets. The 72.5% arousal evaluation result belongs to Logistic Regression.

**Visual:** A concise diagram of audio input, feature extraction, two model predictions, and combined output.

**Presenter note:** The saved models train on all 40 observations. A demonstration on a training recording illustrates operation only. Model probabilities do not verify a student's actual emotion. The observed self-report combinations are Pleasant/High (10), Pleasant/Low–Moderate (7), Not Pleasant/High (4), and Not Pleasant/Low–Moderate (19). These are descriptive counts, not joint prediction accuracy. Use these precise labels because the existing “Unpleasant” and “Deactivated” quadrant names include neutral and moderate ratings. Specific emotion descriptions such as “excited” or “frustrated” are interpretive examples, not validated target classes.

**Sources:** inference/train_and_save_model.py; inference/predict.py; models/bisaya_ser_model_info.json; audio_outputs/tables/18_2d_affect_quadrant_distribution.csv; face/README.md; multimodal/README.md.

## Slide 10 — Conclusions, Limitations, and Next Steps

- The project provides an implemented exploratory workflow for Bisaya post-class affect recognition.
- Valence classification remains weak. Arousal results show preliminary improvement over baseline, with substantial fold variability.
- Main limits: 40 observations, 94 predictors, one year level/session, mixed recording conditions, and no external test set.
- Next steps: expand the sample, standardize recording documentation, and evaluate on independent participants.

**Closing line:** The current evidence supports further research, but does not establish reliable recognition for new students. Questions?

**Visual:** A clean conclusion slide with a brief source footer: “CS412 Instructions.md; project data quality report; exported evaluation tables.”

**Presenter note:** Binary grouping combines distinct original ratings and also limits interpretation. Preserve the original ratings, document lexical normalization, and align the saved arousal model with a justified model choice. Face or multimodal extensions need a separate method and appropriate approval because the supplied protocol specifies audio. Required outputs include separately maintained consent documentation and the group report as well as the computational artifacts. State their actual completion status accurately.

**Sources:** Instructions.md, sections XVIII and XX; DATA_QUALITY_REPORT.md; PRIVACY_AND_ETHICS.md; README.md; export_audio_outputs.py and its result tables.

## Protocol Coverage Map

This is an outline planning aid, not an additional slide.

| Required topic | Slide(s) |
| --- | --- |
| Project overview, objectives, and research questions | 1–2 |
| Participants, collection, consent, and data quality | 3 |
| Independent self-reports and modeling targets | 3–5 |
| Preprocessing and acoustic feature extraction | 4 |
| Educational context and spoken-word exploration | 5 |
| Acoustic characteristics and same-word analysis | 6 |
| Model comparison and participant-grouped evaluation | 7–8 |
| Prototype and project deliverables | 9 |
| Interpretation, limitations, and next steps | 8–10 |
| References | Source footers and speaker notes within slides 1–10 |
