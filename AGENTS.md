# Repository Guidelines

## Project Structure & Module Organization
- `aom/`: core package (data loading, metrics, interventions, model backends, reporting).
- Top-level CLIs: `aom_eval.py`, `aom_cf_patching.py`, `aom_coh_patching.py`.
- `scripts/`: orchestration and reproducibility tooling (`run_paper.py`, `run_submission_full_strong.sh`, evidence checks).
- `tests/`: pytest suite for evidence-contract integrity.
- `data_paper_hardened_v2/`: canonical dataset bundle plus `DATASET_MANIFEST.json`.
- `AoM_JoLLLI/`: manuscript (`AoM_paper.md`) and claim ledger (`AoM_evidence_contract.md`).
- `docs/context/`: project context notes; start with `docs/context/CONTEXT-INDEX.md`, then `docs/context/CONTEXT-PLAN.md`, then read the required `docs/context/CONTEXT-*.md` files for your task.
- For mechanistic patching or intervention tasks, always read `docs/context/CONTEXT-003-mechanistic-patching.md`.
- Generated outputs are expected in `results*/`, `tables*/`, `submission/`, and `dist/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt`: create the reviewer / CI environment.
- `python -m pip install --upgrade pip==24.2 && python -m pip install -r requirements.pip.lock.txt`: create the strict paper-facing clean-room environment.
- `python -m pytest -q`: run the default offline smoke/evidence tests.
- `python scripts/run_paper.py smoke`: end-to-end smoke run (no model downloads required).
- `bash scripts/run_submission_full_strong.sh`: regenerate canonical submission artifacts in `results_submission_full/` and `tables_submission_full/`.
- `MAKE_TABLES_STRICT=1 make tables RESULTS_DIR=results_submission_full TABLES_OUT_DIR=tables_submission_full`: strict table rebuild; fails on missing/inconsistent inputs.
- `python scripts/check_evidence_contract.py && python scripts/check_evidence_contract_fields.py --results_dir results_submission_full`: verify evidence IDs and referenced artifact fields.

## Coding Style & Naming Conventions
- Use Python conventions already present in the repo: 4-space indentation, type hints, and deterministic CLI behavior.
- Prefer `pathlib.Path`, `argparse`, and small helper functions over inline shell-heavy logic in Python files.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep research logic in `aom/`; keep orchestration and one-off automation in `scripts/`.

## Testing Guidelines
- Framework: `pytest` with files named `tests/test_*.py` and test functions `test_*`.
- `tests/test_evidence_contract_fields.py` requires a strict-results directory containing `RUN_MANIFEST.json`; set `AOM_RESULTS_DIR=/path/to/results_submission_full` if needed.
- For changes to metrics, manifests, or paper claims, run `python -m pytest -q` and both evidence-check scripts before opening a PR.

## Commit & Pull Request Guidelines
- Follow existing commit style: short, imperative, descriptive subjects (for example, `Freeze paper submission and release gate`).
- Keep commit scope focused; avoid mixing dataset, code, and manuscript changes without justification.
- PRs should include: summary of intent, key paths changed, exact validation commands run, and rationale for any regenerated artifacts or evidence-contract updates.
