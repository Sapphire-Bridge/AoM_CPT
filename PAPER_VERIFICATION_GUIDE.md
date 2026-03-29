# Paper verification guide (AoM_CPT)

This guide documents the current verification surface of this repository only.
It does not attempt to describe every historical script, local experiment path, or earlier exploratory layout.

This repository is the compact AoM/CPT reproducibility surface. Use this guide for the current paper-facing verification path.

## Primary sources in this repo

- Manuscript: `AoM_JoLLLI/AoM_paper.md`
- Claim ledger / evidence contract: `AoM_JoLLLI/AoM_evidence_contract.md`
- Repo routing: `README.md`
- Minimal reviewer runner: `scripts/run_paper.py`
- Canonical submission runner: `scripts/run_submission_full_strong.sh`
- Clean-room reproduction: `scripts/final_repro_cleanroom.sh`
- Artifact bundle / frozen-artifact verifier: `scripts/build_replication_bundle.sh`
- Strict table generation: `scripts/make_tables.py`
- Evidence ID check: `scripts/check_evidence_contract.py`
- Evidence artifact/field check: `scripts/check_evidence_contract_fields.py`

## Fast verification path

Use this path to establish that the repository is coherent and runnable without regenerating the full paper artifact surface:

1. install pinned dependencies:
   `python -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt`
2. run the offline smoke check:
   `python scripts/run_paper.py smoke`
3. run the default offline test suite:
   `python -m pytest -q`
4. inspect:
   `results/paper_smoke/aom_eval.csv`,
   `results/paper_smoke/results_report.md`,
   `results/paper_smoke/RUN_MANIFEST.json`
5. expect the field-level evidence-contract test to skip unless a full `results_submission_full/` artifact set is present

## Strict reproduction path

Use this path for paper-facing artifact regeneration and evidence verification:

1. preferred clean-room path:
   `bash scripts/final_repro_cleanroom.sh --repro-mode full_recompute`
2. in-place canonical run:
   `bash scripts/run_submission_full_strong.sh`
3. strict tables:
   `MAKE_TABLES_STRICT=1 make tables RESULTS_DIR=results_submission_full TABLES_OUT_DIR=tables_submission_full`
4. evidence checks:
   `python scripts/check_evidence_contract.py`
   `python scripts/check_evidence_contract_fields.py --results_dir results_submission_full`
   `python -m pytest -q tests/test_evidence_contract_ids.py tests/test_evidence_contract_fields.py`

## Artifact surface to inspect

After the strict run, inspect:

- `results_submission_full/*.csv`
- `results_submission_full/*.manifest.json`
- `results_submission_full/results_report.md`
- `results_submission_full/RUN_MANIFEST.json`
- `tables_submission_full/*.tex`

---

## One-command clean-room reproduction (recommended)

This runs from a pinned git commit without relying on your local working tree state:

```bash
bash scripts/final_repro_cleanroom.sh --repro-mode full_recompute
```

What it does:
1) `git archive` export to a fresh clean-room directory
2) creates a venv, pins `pip==24.2`, and installs `requirements.pip.lock.txt`
3) runs the canonical submission suite + strict tables
4) runs:
   `python scripts/check_evidence_contract.py`
   `python scripts/check_evidence_contract_fields.py`
   `python -m pytest -q tests/test_evidence_contract_ids.py tests/test_evidence_contract_fields.py`
5) builds a replication tarball

Use `bash scripts/final_repro_cleanroom.sh --help` for options like `--git-rev`, `--clean-dir`, `--repro-mode`, `--local-files-only`, and HF revision flags.

### Repro mode semantics

- `full_recompute`: reruns model evaluations/patching then builds the archive.
- `frozen_artifacts`: reuses an existing artifact set and only validates/packages.

Frozen-artifact example:

```bash
bash scripts/build_replication_bundle.sh \
  --mode submission_full_strong \
  --repro-mode frozen_artifacts \
  --results-dir results_submission_full \
  --tables-dir tables_submission_full \
  --archive aom_replication_bundle.tar.gz
```

---

## In-place reproduction (manual)

### 0) Install (pinned environment)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.2
python -m pip install -r requirements.pip.lock.txt
```

Dependency surfaces:

- Reviewer / smoke validation: `requirements.txt`
- Strict paper-facing reproduction: `requirements.pip.lock.txt`
- Provenance-only reference snapshot: `requirements.lock.txt`

`requirements.lock.txt` contains Conda-only packages such as `anaconda-anon-usage` and `conda`, so it is not the pip install surface.

### 1) Verify the hardened dataset bundle (hashes + stable bundle ID)

The canonical paper dataset bundle is checked in at `data_paper_hardened_v2/` with a deterministic manifest:
- `data_paper_hardened_v2/DATASET_MANIFEST.json`

Expected `paper_hardened_v2` IDs:
- stable bundle id: `fa2f39387339d26abd45912e31eede3b5f88aac4ed7bff20660262fcb46787ff`
- per-file SHA-256:
  - `disamb_pairs.jsonl`: `a9587c185bde2ca83a1fd2d22a62d9d961b2105e7626117240e8aeff207adee1`
  - `counterfactual.jsonl`: `e269fbd4fb819591d0bd5932137fcd18037a9ede9e743b7528916ead9f9374bf`
  - `coherence.jsonl`: `04e122738582a3d5ea2a7e7f8e38a3c7841c3bab2cc04cf4d24640d5a902fc64`

Verification snippet:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
from aom.data.bundle_manifest import compute_bundle_id

d = Path("data_paper_hardened_v2")
m = json.loads((d / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
print("bundle_id(manifest):   ", m.get("bundle_id", "<missing>"))
print("bundle_id(recomputed): ", compute_bundle_id(m))
for fn in ["disamb_pairs.jsonl", "counterfactual.jsonl", "coherence.jsonl"]:
    h = hashlib.sha256((d / fn).read_bytes()).hexdigest()
    print(fn, h, "match", h == m["files"][fn]["sha256"])
PY
```

### 2) Run the canonical submission suite

This regenerates the paper-cited artifact set into the canonical output directories:

```bash
bash scripts/run_submission_full_strong.sh
```

Expected paper-facing outputs:
- `results_submission_full/aom_eval.csv`
- `results_submission_full/aom_eval.manifest.json`
- `results_submission_full/cpt_specificity_disamb_only.csv`
- `results_submission_full/cpt_specificity_disamb_only.manifest.json`
- `results_submission_full/cf_patching.csv`
- `results_submission_full/cf_patching.manifest.json`
- `results_submission_full/cf_patching_shift_vs_subinv.csv`
- `results_submission_full/cf_patching_shift_vs_subinv.manifest.json`
- `results_submission_full/CF_SHIFT_SUBINV_RUN_MANIFEST.json`
- `results_submission_full/coh_patching.csv` (merged from `coh_patching_gpt2.csv`, `coh_patching_qwen.csv`, `coh_patching_qwen15.csv`, and `coh_patching_qwen3b.csv`; the merged CSV has no companion manifest)
- `results_submission_full/coh_patching_gpt2.csv`
- `results_submission_full/coh_patching_gpt2.manifest.json`
- `results_submission_full/coh_patching_qwen.csv`
- `results_submission_full/coh_patching_qwen.manifest.json`
- `results_submission_full/coh_patching_qwen15.csv`
- `results_submission_full/coh_patching_qwen15.manifest.json`
- `results_submission_full/coh_patching_qwen3b.csv`
- `results_submission_full/coh_patching_qwen3b.manifest.json`
- `results_submission_full/results_report.md`
- `results_submission_full/RUN_MANIFEST.json`
- `tables_submission_full/*.tex`

### 3) Regenerate paper tables (strict mode)

The canonical runner already runs strict tables, but you can rerun them explicitly:

```bash
MAKE_TABLES_STRICT=1 make tables RESULTS_DIR=results_submission_full TABLES_OUT_DIR=tables_submission_full
```

Strict mode fails fast if required inputs/provenance are missing or inconsistent across inputs (prevents mixing artifact sets).

### 4) Evidence checks (paper tags → real artifacts/fields)

```bash
python scripts/check_evidence_contract.py
python scripts/check_evidence_contract_fields.py --results_dir results_submission_full
python -m pytest -q tests/test_evidence_contract_ids.py tests/test_evidence_contract_fields.py
```

### 5) Release-gate command (strict SHA)

```bash
TOTAL_BUDGET_SEC=28800 FIELDS_TIMEOUT_SEC=1200 STRICT_SHA=1 ALLOW_DIRTY=0 RESULTS_DIR=results_submission_full TABLES_DIR=tables_submission_full ARCHIVE_NAME=aom_replication_bundle_fast8h.tar.gz bash scripts/release_gate_fast_8h.sh
```

---

## Notes on pinning models and determinism

- All runners support `--local_files_only` (offline) and HF `--revision` / `--tokenizer_revision` pins.
- Each canonical CSV row logs HF metadata (requested revision, effective tokenizer revision, and observed commit hash) plus runtime provenance (git commit, argv hash, platform, dtypes, bootstrap config, dataset bundle id).
- Inference runs in `model.eval()` without sampling; bootstraps are seeded. On CPU you should generally expect exact metric stability across reruns with identical configs. MPS/CUDA may show small floating-point differences depending on backend determinism.

---

## Troubleshooting

- If `import torch` fails with an OpenMP shared-memory error (e.g. `OMP: Error #179: Function Can't open SHM2 failed`), run in a less-restricted environment or consult your system’s OpenMP runtime settings (this is environment-specific and not a repo bug).

---

## Zenodo handoff

After creating a validated replication bundle and tagged GitHub release, follow:

- `docs/release/ZENODO_RELEASE_CHECKLIST.md`
