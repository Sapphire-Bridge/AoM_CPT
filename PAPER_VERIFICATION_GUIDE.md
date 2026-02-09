# Paper verification guide (AoM Prototype repo)

This guide is for reviewers and collaborators who want to (a) regenerate the **paper-cited artifacts** end-to-end from a clean checkout and (b) verify that manuscript evidence tags resolve to concrete files/fields.

Canonical paper sources in this repo:
- Manuscript: `AoM_JoLLLI/AoM_paper.md`
- Claim ledger / evidence contract: `AoM_JoLLLI/AoM_evidence_contract.md`
- Canonical submission runner: `scripts/run_submission_full_strong.sh`
- Strict table regeneration: `make tables` / `scripts/make_tables.py`
- Clean-room replay (one command): `scripts/final_repro_cleanroom.sh`

---

## One-command clean-room reproduction (recommended)

This runs from a pinned git commit without relying on your local working tree state:

```bash
bash scripts/final_repro_cleanroom.sh
```

What it does:
1) `git archive` export to a fresh clean-room directory
2) creates a venv and installs pinned dependencies (prefers `requirements.lock.txt` when present)
3) runs the canonical submission suite + strict tables
4) runs evidence checks (`scripts/check_evidence_contract*.py`) + `pytest -q`
5) builds a replication tarball

Use `bash scripts/final_repro_cleanroom.sh --help` for options like `--git-rev`, `--clean-dir`, `--local-files-only`, and HF revision flags.

---

## In-place reproduction (manual)

### 0) Install (pinned environment)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

If you prefer a smaller top-level install set, `requirements.txt` is pinned but not transitively locked.

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

Expected outputs:
- `results_submission_full/aom_eval.csv` (+ `*.manifest.json`)
- `results_submission_full/cpt_specificity_disamb_only.csv` (+ `*.manifest.json`)
- `results_submission_full/cf_patching.csv` (+ `*.manifest.json`)
- `results_submission_full/coh_patching.csv` (+ `*.manifest.json`)
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
pytest -q
```

---

## Notes on pinning models and determinism

- All runners support `--local_files_only` (offline) and HF `--revision` / `--tokenizer_revision` pins.
- Each canonical CSV row logs HF metadata (requested revision, effective tokenizer revision, and observed commit hash) plus runtime provenance (git commit, argv hash, platform, dtypes, bootstrap config, dataset bundle id).
- Inference runs in `model.eval()` without sampling; bootstraps are seeded. On CPU you should generally expect exact metric stability across reruns with identical configs. MPS/CUDA may show small floating-point differences depending on backend determinism.

---

## Troubleshooting

- If `import torch` fails with an OpenMP shared-memory error (e.g. `OMP: Error #179: Function Can't open SHM2 failed`), run in a less-restricted environment or consult your system’s OpenMP runtime settings (this is environment-specific and not a repo bug).

