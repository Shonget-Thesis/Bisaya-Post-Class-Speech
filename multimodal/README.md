# Multimodal Modality

This folder is reserved for participant-aligned audio and facial features.

No multimodal dataset has been generated yet because the face-feature dataset is not available. Multimodal processing should begin only after the face modality is approved, extracted, and quality-checked.

The future combined dataset should:

1. Contain one row per observation.
2. Join audio and face features using `Participant Code`, not row position or filenames alone.
3. Preserve the independently reported valence and arousal outcomes.
4. Prefix predictors with `audio_` and `face_` so their modality is clear.
5. Report missing modalities and exclude contextual variables from the initial acoustic/visual prediction experiment.
6. Keep every participant in a single validation fold.

Do not label an audio-only result as multimodal.
