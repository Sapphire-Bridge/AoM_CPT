# Contributing

## Scope

This repository is a research companion for AoM/CPT experiments and paper artifacts. Contributions should prioritize reproducibility, provenance, and evidence integrity.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

## Development Rules

- Keep research logic in `aom/`.
- Keep orchestration/release automation in `scripts/`.
- Prefer deterministic behavior and explicit CLI flags.
- Preserve provenance fields required by table/evidence checks:
  - `dataset_bundle_id`
  - `git_commit`
  - `argv_sha256`

## Validation Before PR

Run all of:

```bash
pytest -q
python scripts/check_evidence_contract.py
python scripts/check_evidence_contract_fields.py --results_dir results_submission_full
MAKE_TABLES_STRICT=1 make tables RESULTS_DIR=results_submission_full TABLES_OUT_DIR=tables_submission_full
```

If your change affects release packaging, also run:

```bash
bash scripts/build_replication_bundle.sh --mode submission_full_strong --repro-mode frozen_artifacts --skip-run --results-dir results_submission_full --tables-dir tables_submission_full
```

## Pull Requests

Each PR should include:

- intent summary
- files changed
- exact validation commands and outcomes
- rationale for regenerated artifacts (if any)

Use focused PRs; avoid mixing unrelated manuscript, dataset, and runtime changes.
