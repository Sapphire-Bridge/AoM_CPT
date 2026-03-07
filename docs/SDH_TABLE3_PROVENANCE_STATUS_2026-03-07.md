# SDH Table 3 Provenance Status

- Status date: 2026-03-07
- Outcome: `B` — the exact manuscript SDH source artifact has not yet been recovered.

## What Was Recovered

- Submitted manuscript snapshot tag: `submission-snapshot-2026-02-23` at commit `d379ec9a91d9dbce26bedd022190a335544139d5`
- Submitted build outputs under `submission/` contain the `N=100` Table 3 language and an embedded `Table 3b` appendix with per-strategy values and `N=30/30/40` counts.
- Current checked-in canonical SDH artifact: `results_submission_full/cpt_specificity_disamb_only.csv`

## What Was Searched

- Git history for `AoM_JoLLLI/AoM_paper.md`
- Git history for `scripts/make_tables.py`
- Git history for `aom/metrics/disamb.py`, `aom_eval.py`, and `scripts/run_submission_full_strong.sh`
- Current workspace result directories and submission build outputs for SDH/specificity CSVs, tables, and notes

## Current Provenance Conclusion

- The current checked-in canonical SDH artifact does not match the submitted manuscript Table 3 story.
- The current checked-in canonical SDH artifact reports `N=104` directions per model and, within the reported run, uses only the `matched_token` control path.
- Because no exact older SDH CSV or archived result directory matching the submitted `N=100` / strategy-stratified story has been recovered yet, revision text in this repo is scoped to the checked-in canonical artifact rather than to the unrecovered submitted SDH analysis.
