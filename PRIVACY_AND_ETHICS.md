# Privacy and Ethics Handling

The dataset contains human voice recordings and locally available video files. These can identify participants even when filenames use pseudonyms.

## Required checks before sharing

1. Confirm that informed consent was obtained before every recording.
2. Confirm that the approved protocol permits the intended storage location, repository access level, analysis, presentation, and publication.
3. Confirm whether consent and institutional approval explicitly cover video. The supplied research protocol describes audio collection and audio-based modeling only.
4. Confirm that the mapping between participant identity and participant code is stored separately, encrypted, and access-controlled.
5. Confirm the retention and deletion schedule required by the institution.

## Repository rules

- Do not commit consent forms, names, student numbers, identity-code mappings, or other direct identifiers.
- Do not commit `Videos/` without separate documented approval; the folder is ignored by Git.
- Treat WAV files and Git history as sensitive participant data. Verify repository visibility and authorization before pushing.
- Share derived feature tables only under the approved protocol. Acoustic features may still carry identifying information.
- Report only aggregate results; do not describe individual participants.

This repository does not itself prove that consent or institutional approval exists. Those records should be maintained in the institutionally approved secure location, not in a public source repository.
