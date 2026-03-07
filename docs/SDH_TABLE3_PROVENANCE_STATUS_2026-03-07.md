# SDH Table 3 Provenance Status

- Status date: 2026-03-07
- Outcome: `B` — the exact manuscript SDH source artifact has not yet been recovered.

## What Was Recovered

- Submitted manuscript snapshot tag: `submission-snapshot-2026-02-23` at commit `d379ec9a91d9dbce26bedd022190a335544139d5`
- Submitted build outputs under `submission/` contain the `N=100` Table 3 language and an embedded `Table 3b` appendix with per-strategy values and `N=30/30/40` counts.
- Current frozen SDH artifact snapshot: `results_submission_full/cpt_specificity_disamb_only.csv`

## What Was Searched

- Git history for `AoM_JoLLLI/AoM_paper.md`
- Git history for `scripts/make_tables.py`
- Git history for `aom/metrics/disamb.py`, `aom_eval.py`, and `scripts/run_submission_full_strong.sh`
- Current workspace result directories and submission build outputs for SDH/specificity CSVs, tables, and notes

## Current Provenance Conclusion

- The current frozen SDH artifact snapshot does not match the submitted manuscript Table 3 story.
- The current frozen SDH artifact snapshot reports `N=104` directions per model and, within the reported run, uses only the `matched_token` control path.
- This snapshot refers to the local `results_submission_full/` artifact set hash-frozen in `docs/SUBMISSION_SNAPSHOT_2026-03-07.md`; it is not git-tracked at the submission commit.
- The recovered CSV rows currently carry artifact-origin commit `ea24714787ac14d6641f15fc08483a463dfb70b9`; manuscript/docs-only release commits layered on top of this snapshot should therefore use the frozen-artifact packaging path rather than `STRICT_SHA=1`.
- Because no exact older SDH CSV or archived result directory matching the submitted `N=100` / strategy-stratified story has been recovered yet, revision text in this repo is scoped to the current frozen artifact snapshot rather than to the unrecovered submitted SDH analysis.
