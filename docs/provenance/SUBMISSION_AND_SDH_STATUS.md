# Submission and SDH Provenance Status

Consolidated: 2026-03-25
Sources consolidated from dated notes created on 2026-03-07.

## Purpose

- Preserve the small amount of provenance information that remains useful for the under-review JoLLI submission.
- Keep submitted-snapshot facts separate from the current workspace state and from post-submission claim-scope repairs.
- Replace several overlapping one-off notes with one maintained record.

## Submitted Snapshot

- Submission tag: `submission-snapshot-2026-02-23`
- Submitted paper commit: `d379ec9a91d9dbce26bedd022190a335544139d5`
- Date submitted: 2026-02-23
- Submitted manuscript snapshot hashes:
  - `AoM_JoLLLI/AoM_paper.md` -> `218b28904aeac69e46fa5d71d541d99f77dcc351c7ab1859227ff09a29a6f7e1`
  - `AoM_JoLLLI/AoM_evidence_contract.md` -> `9d3182ab6e6c9111642b215e69628cdc548c479fc5ab7a5172f4d92528093ff6`

## Under-Review Separation Rule

- The submitted manuscript snapshot, the current repo state, and post-submission analyses are separate records.
- Post-submission robustness checks or claim-scope repairs must not silently replace the submitted artifact lineage.

## SDH Table 3 Provenance Status

- Status date of the original audit: 2026-03-07
- Outcome: the exact submitted SDH source artifact has not been recovered.

### What Was Recovered

- Submitted manuscript snapshot tag: `submission-snapshot-2026-02-23`
- Submitted build outputs under `submission/` contain the `N=100` Table 3 language and an embedded `Table 3b` appendix with per-strategy values and `N=30/30/40` counts.
- Current checked-in canonical SDH artifact: `results_submission_full/cpt_specificity_disamb_only.csv`

### What Was Searched

- Git history for `AoM_JoLLLI/AoM_paper.md`
- Git history for `scripts/make_tables.py`
- Git history for `aom/metrics/disamb.py`, `aom_eval.py`, and `scripts/run_submission_full_strong.sh`
- Workspace result directories and submission build outputs for SDH/specificity CSVs, tables, and notes

### Current Provenance Conclusion

- The current checked-in canonical SDH artifact does not match the submitted manuscript Table 3 story.
- The current checked-in canonical SDH artifact reports `N=104` directions per model and, within the recovered run, uses only the `matched_token` control path.
- Until an exact older SDH CSV or archived result directory matching the submitted `N=100` and strategy-stratified story is recovered, revision text in this repo should be scoped to the checked-in canonical artifact rather than to the unrecovered submitted SDH analysis.

## 2026-03-07 On-Disk Artifact Freeze

Treat this section as an archival record of the `results_submission_full/` filesystem state on 2026-03-07, not as a description of current workspace contents.

- `results_submission_full/aom_eval.csv` -> `55e88c6d46b98300232f0dc0cb989681a8caf501f1f0638374fc5241d2612b3d`
- `results_submission_full/cf_patching.csv` -> `7884865f44608e971556a7ff22146364f905d2e7b268907e2dbe7857314a8f61`
- `results_submission_full/cf_patching_part1.csv` -> `158f464dfaec9d386ab2cfa914e00f9f40d9f69fc6169acbf968bcddd75aeb43`
- `results_submission_full/cf_patching_part2.csv` -> `c04a6d9c0eb90a080bd020e2909a146afe815ba08b5a4fd5149099af01bf6af5`
- `results_submission_full/cf_patching_shift_vs_subinv.csv` -> `e4a743442aa3aa58d08eef0feaca183841d284d11e1308300c6e950c577795f7`
- `results_submission_full/coh_patching.csv` -> `83beb2044ab29d752724a7a162f1cc597c3b530ab66f5629c0dfcad2fb9b658d`
- `results_submission_full/coh_patching_qwen.csv` -> `17eee25897ee08d398550f2f35b360bb1ac9ee95d49b64c6d2623d07fdb72465`
- `results_submission_full/coh_patching_qwen15.csv` -> `70f531770eab4b0b63a2bce8680b4a21d56556d4155ddd725b10580e8b4330ce`
- `results_submission_full/coh_patching_qwen3b.csv` -> `9057352ce2ec51657a3cf56940bcc41df101a65bc617c880278f13dcab6176f1`
- `results_submission_full/cpt_specificity_disamb_only.csv` -> `b5882fc99e0e7ef5f50911bffc93d70fe5f4490e1ee696465ad17241d86869a5`

## Claim-Scope Repair Summary

- A post-submission audit found an SDH/Table 3 manuscript-versus-artifact mismatch.
- The checked-in matched-token SDH artifact supports target-span concentration relative to nearby matched spans.
- It does not support the broader strategy-sensitivity language from the submitted text.
- Layer-sweep summaries such as `mean_max_effect` and `mean_argmax_layer` should be described as descriptive post-selection summaries, not confirmatory estimates.
- Core CF and COH results were not implicated by this specific provenance issue.
