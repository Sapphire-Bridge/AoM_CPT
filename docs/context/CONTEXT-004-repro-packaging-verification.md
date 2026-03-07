# Project Context 004 - Reproduction Pipelines, Tables, Evidence Checks, and Packaging (release path)

## Scope (files covered)

- `scripts/run_submission_full_strong.sh:215` - Canonical submission run sequence (behavioral + specificity + CF/COH patching).
- `scripts/run_paper.py:640` - Preset mode runner (`smoke`, `m1max`, `a100`) and run-manifest writer.
- `scripts/make_tables.py:17` - Strict table generation gate with provenance cross-checks.
- `tables/table_aom_eval.py:45` - AoM table extraction from `aom_eval.csv`.
- `tables/table_cf_patching.py:45` - CF patching table extraction.
- `tables/table_coh_patching.py:45` - COH patching table extraction.
- `scripts/report_results.py:14` - Wrapper over `aom.reporting` results summary generation.
- `scripts/check_evidence_contract.py:36` - Evidence ID presence check (paper vs contract).
- `scripts/check_evidence_contract_fields.py:184` - Artifact/field-level evidence contract check.
- `tests/test_evidence_contract_ids.py:11` - Pytest wrapper for evidence-ID consistency.
- `tests/test_evidence_contract_fields.py:34` - Pytest wrapper for evidence-field resolution.
- `scripts/build_replication_bundle.sh:127` - Bundle builder and required-artifact gate.
- `scripts/final_repro_cleanroom.sh:205` - Clean-room export, run, checks, and archive copyback.
- `Makefile:8` - `make tables` entrypoint.
- `PAPER_VERIFICATION_GUIDE.md:14` - Human runbook for end-to-end verification.

## Architecture Overview

- There are two primary execution surfaces:
- canonical publication runner (`run_submission_full_strong.sh`)
- preset runner (`scripts/run_paper.py`)
- Table generation is treated as a data-integrity gate, not pure formatting.
- Evidence contract checks tie manuscript claim IDs to concrete artifact columns/JSON paths.
- Packaging scripts enforce artifact presence before archive creation.
- Clean-room script validates reproducibility from a pinned git revision, optionally with locked dependencies.

## Key Flows

- Flow A: canonical submission suite
- Input: dataset dir + run parameters.
- Validation: dataset files and manifest must exist before run (`scripts/run_submission_full_strong.sh:178`).
- Side effects: writes canonical CSVs, per-artifact manifests, report, and tables.
- Output: `results_submission_full/*` and `tables_submission_full/*`.

- Flow B: preset runner (`smoke`/`m1max`/`a100`)
- Input: mode + results/data dirs + hardware flags.
- Validation: mode-specific checks (for example, A100 patching + `device_map` incompatibility).
- Side effects: executes subcommands (`aom_eval.py`, patching CLIs, `report_results.py`) and writes `RUN_MANIFEST.json`.
- Output: mode-specific result directory with reproducibility metadata.

- Flow C: strict table regeneration
- Input: CSV artifacts and output table dir.
- Validation: required provenance fields, uniqueness checks, cross-file consistency checks.
- Side effects: invokes table scripts per artifact.
- Output: `.tex` files only when provenance gates pass.
- Entry: `scripts/make_tables.py:117`.

- Flow D: evidence-contract verification
- Input: manuscript markdown + evidence contract + selected results dir.
- Validation: evidence IDs, artifact existence, CSV columns, JSON paths.
- Side effects: CI-friendly nonzero exit on mismatch.
- Output: pass/fail diagnostics.
- Entry: `scripts/check_evidence_contract.py:43`, `scripts/check_evidence_contract_fields.py:198`.

- Flow E: replication bundle and clean-room
- Input: run outputs + selected mode + archive path.
- Validation: required directories/files before tarball creation.
- Side effects: tarball with curated paths and SHA-256 digest.
- Output: bundle archive and `.sha256` file.
- Entries: `scripts/build_replication_bundle.sh:187`, `scripts/final_repro_cleanroom.sh:273`.

## Security / Invariants

- Tables cannot mix provenance across CSV inputs.
- Enforced by cross-check columns in `scripts/make_tables.py:14` and mismatch checks at `scripts/make_tables.py:166` onward.

- Evidence claims must resolve to concrete artifacts and fields.
- Enforced by `scripts/check_evidence_contract_fields.py:239` onward.

- Bundle builder requires canonical artifacts before packaging.
- Enforced by `require_file/require_dir` gates (`scripts/build_replication_bundle.sh:187` onward).

- Clean-room run can forbid dirty source tree by default.
- Enforced at `scripts/final_repro_cleanroom.sh:189`.

- Archive excludes transient cache/compiled artifacts.
- Enforced via tar excludes (`scripts/build_replication_bundle.sh:257`).

- Canonical submission script can relax git requirement for run rows (`--no-require_git`).
- Applied in `scripts/run_submission_full_strong.sh:193`; document this when interpreting provenance.

## Interfaces

### Primary command interfaces

- Canonical run:
- `bash scripts/run_submission_full_strong.sh [options]`
- Outputs listed in script header (`scripts/run_submission_full_strong.sh:33`).

- Preset run:
- `python scripts/run_paper.py {smoke|m1max|a100} [options]`
- Mode parser at `scripts/run_paper.py:1085`.

- Strict tables:
- `MAKE_TABLES_STRICT=1 make tables RESULTS_DIR=<...> TABLES_OUT_DIR=<...>`
- Make target at `Makefile:8`.

- Evidence checks:
- `python scripts/check_evidence_contract.py`
- `python scripts/check_evidence_contract_fields.py --results_dir <dir>`

- Clean-room:
- `bash scripts/final_repro_cleanroom.sh [options]`

### Output interfaces

- Canonical result files include:
- `aom_eval.csv`
- `cpt_specificity_disamb_only.csv`
- `cf_patching.csv`
- `coh_patching.csv`
- `*.manifest.json` companions
- `RUN_MANIFEST.json`
- `results_report.md`

- Required CSV provenance columns for table/evidence checks:
- `dataset_bundle_id`, `git_commit`, `argv_sha256` (`scripts/make_tables.py:12`, `scripts/check_evidence_contract_fields.py:19`).

## Observability

- Submission script logs executed commands to `commands.log` (`scripts/run_submission_full_strong.sh:205`).
- `run_paper.py` prints every subprocess command line before execution (`scripts/run_paper.py:59`).
- `report_results.py` emits markdown summary inventory for entire results tree.
- Evidence check scripts produce explicit failing row/field identifiers to stderr.

## Operational Notes / Gotchas

- `scripts/check_evidence_contract_fields.py` remaps `results*/...` tokens to provided `--results_dir` (`scripts/check_evidence_contract_fields.py:90`).
- `tests/test_evidence_contract_fields.py` skips when no compatible results dir is found (`tests/test_evidence_contract_fields.py:37`).
- `run_paper.py smoke` builds a tiny local model/tokenizer for fully offline tests (`scripts/run_paper.py:655`).
- `run_paper.py a100` forbids patching with `device_map`; split runs if needed (`scripts/run_paper.py:911`).
- There is no checked-in CI workflow file in this scan; local scripts are the primary verification contract.

## Risks / Limitations

- Reproducibility assumes external model availability unless `--local_files_only` with local weights is used.
- Table scripts depend on exact column names; metric rename refactors can silently break paper generation without strict mode.
- Evidence contract rows reference both active and deferred artifacts; citing deferred IDs can create verification drift.
- `TODO(verify):` add explicit release checklist file if this repo will enforce PR-time validation in external CI.
