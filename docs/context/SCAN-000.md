# SCAN-000 - Broad Repo Scan (Evidence-Based)

Date: 2026-02-22
Repo root: `/Users/felixb/Desktop/Coding/AoM_CPT`
HEAD commit: `ea24714787ac14d6641f15fc08483a463dfb70b9`
Scan mode: local-only (`rg`, `git`, `find`, `du`, `wc`, `ls`), no dependency installs, no network.

## Step 0 - Human guidance (fast read)

Files checked:
- `README.md:1`
- `AGENTS.md` not found in repo root
- `CLAUDE.md` not found in repo root

Conventions extracted:
- Primary workflows are CLI-driven (`pytest -q`, `python aom_eval.py`, `python scripts/run_paper.py`, `bash scripts/run_submission_full_strong.sh`).
- Reproducibility is explicit: per-run CSV + `*.manifest.json` + `RUN_MANIFEST.json` (`README.md:73`, `README.md:119`).
- Security-sensitive model loading is gated by `--trust_remote_code` (default off) and `--local_files_only` for offline mode (`README.md:96`, `README.md:101`).
- Canonical paper dataset bundle is hash-pinned (`data_paper_hardened_v2/DATASET_MANIFEST.json:1`).

## Step 1 - Repo size and structure

### Core metrics
- Tracked files: `82`
- Total size (`du -sh .`): `2.3M`
- Top-level tracked-file concentration:
  - `aom`: 50 files
  - `scripts`: 9 files
  - `data_paper_hardened_v2`: 4 files
  - `tables`: 3 files
  - `tests`: 2 files
  - `AoM_JoLLLI`: 2 files

### Top-level disk footprint
- `aom`: `792K`
- `data_paper_hardened_v2`: `248K`
- `results_submission_full`: `236K`
- `scripts`: `196K`
- `AoM_JoLLLI`: `84K`
- `results_submission_full_mps`: `56K`
- `results_quickcheck`: `52K`

### Directory map (maxdepth 2)
Key roots present:
- `aom/`
- `scripts/`
- `tables/`
- `tests/`
- `data_paper_hardened_v2/`
- `AoM_JoLLLI/`

Not present (important for template adaptation):
- No `backend/` tree
- No `frontend/` tree
- No `.github/workflows/` in this checkout

## Step 2 - Languages and LOC (fallback method)

`cloc` and `tokei` are not installed; fallback computed from tracked files.

Totals:
- Total tracked LOC: `18,909`
- Total tracked files: `82`

Top extensions by LOC:
- `.py`: 64 files, 16,011 LOC
- `.md`: 6 files, 980 LOC
- `.sh`: 3 files, 915 LOC
- `.txt`: 3 files, 471 LOC
- `.jsonl`: 3 files, 432 LOC
- `.json`: 1 file, 80 LOC

## Step 3 - Entrypoints and architectural anchors

This repository is a research evaluation toolkit, not a service backend/frontend app.

### Anchor map

| Area | Anchor files/dirs | Why this is an anchor |
|---|---|---|
| Core eval runner | `aom_eval.py:710`, `aom_eval.py:984` | Main AoM behavioral + CPT CLI, provenance fields, manifest writing, and sweep/error policy logic. |
| Model loading + security | `aom/models/loader.py:32` | Central Hugging Face load path; applies `local_files_only`, `trust_remote_code`, revisions, dtype, device map behavior. |
| Config + repro + manifests | `aom/config.py:15`, `aom/repro.py:99`, `aom/run_manifest.py:49`, `aom/run_summary.py:72` | Config ingestion, deterministic seeding policy, redacted run manifests, run status accounting. |
| Data schema + validation | `aom/data/schemas.py:7`, `aom/data/validate.py:8`, `aom/data/loaders.py:21`, `aom/data/dataset_manifest.py:75`, `aom/data/bundle_manifest.py:62` | Enforces JSONL shape, boundary hygiene, invalid-row accounting, and dataset SHA gates. |
| Core metrics | `aom/metrics/disamb.py:152`, `aom/metrics/counterfactual.py:14`, `aom/metrics/coherence.py:14`, `aom/metrics/composite.py:9` | Defines primary reported metrics and bootstrap semantics. |
| Patching framework | `aom/interventions/activation_patching.py:11`, `aom/interventions/patching/base.py:70`, `aom/interventions/patching/cf_protocol.py:22`, `aom/interventions/patching/coh_protocol.py:37` | Defines activation patching architecture, protocol abstractions, skip logic, and stratified outputs. |
| Specialized patching CLIs | `aom_cf_patching.py:75`, `aom_coh_patching.py:73` | Dedicated CF/COH causal patching pipelines with run manifests. |
| Dataset generation | `scripts/generate_data.py:25`, `scripts/generate_data.py:1289` | Canonical JSONL synthesis with controls (`expected_effect`, `group`). |
| Paper orchestration | `scripts/run_paper.py:640`, `scripts/run_paper.py:741`, `scripts/run_paper.py:910`, `scripts/run_submission_full_strong.sh:215` | Hardware presets and canonical submission artifact pipeline. |
| Tables + evidence checks | `scripts/make_tables.py:17`, `scripts/check_evidence_contract.py:36`, `scripts/check_evidence_contract_fields.py:184`, `tests/test_evidence_contract_fields.py:34` | Strong provenance cross-checks and paper-contract validation gates. |
| Packaging / clean-room | `scripts/build_replication_bundle.sh:127`, `scripts/final_repro_cleanroom.sh:205` | Repro bundle generation and clean-room execution from pinned commit. |
| Mechanistic backend layer | `aom/mechanistic/backends/base.py:40`, `aom/mechanistic/backends/hf_eager.py:27`, `aom/mechanistic/backends/transformer_lens.py:251` | Optional deeper benchmark/perf and attention-pattern instrumentation interfaces. |

## Step 4 - Hotspots and complexity signals

### Largest files by LOC (top hotspots)
- `aom_eval.py` (1,392)
- `scripts/generate_data.py` (1,312)
- `scripts/run_paper.py` (1,143)
- `aom/metrics/disamb.py` (903)
- `aom/interventions/sae_adapter.py` (833)
- `aom/reporting.py` (801)
- `aom_cf_patching.py` (611)
- `aom_coh_patching.py` (587)
- `aom/metrics/sae_decomposition.py` (572)
- `aom/mechanistic/backends/transformer_lens.py` (502)

### Cheap architecture checks (adapted from service template)
- API route surface (`APIRouter`, `@router`, etc.): no matches in this repo.
- Worker/cron queue patterns (`cron`, `enqueue_job`, `arq`, etc.): no scheduler runtime detected in tracked source (only dependency references in lockfile).
- Frontend runtime transport search (`EventSource`, `SSE`, `/api/*`): no frontend tree in this repo.

### Security and provenance enforcement points
- HF custom code gate: `aom_eval.py:731`, `aom_cf_patching.py:92`, `aom_coh_patching.py:90`
- Offline gate: `aom_eval.py:769`, `aom_cf_patching.py:97`, `aom_coh_patching.py:95`
- Dataset SHA gate: `aom_eval.py:634`, `aom_cf_patching.py:340`, `aom_coh_patching.py:325`
- Argv redaction + hash: `aom/run_manifest.py:20`, `aom_eval.py:627`, `aom_eval.py:1133`
- Strict JSON manifest writing (NaN/Inf sanitized): `aom/run_manifest.py:92`

### Deep-dive candidates (heuristic)
Marked deep-dive candidate if any of: >500 LOC, core integration boundary, or security/provenance critical.
- `aom_eval.py` - core integration boundary + largest CLI.
- `scripts/run_paper.py` - orchestration boundary for all paper presets.
- `scripts/generate_data.py` - dataset generation controls and schema assumptions.
- `aom/metrics/disamb.py` - includes both core metric and CPT specificity logic.
- `aom/reporting.py` - broad artifact scanner used in release reporting.
- `aom_cf_patching.py` and `aom_coh_patching.py` - publication-facing causal patching CLIs.

Optional TODO deep dives (not required for first context pass):
- `aom/interventions/sae_adapter.py` and `aom/metrics/sae_decomposition.py`
- `aom/mechanistic/backends/transformer_lens.py`

## Step 5 - Proposed context-doc layering

- Layer 0: `docs/context/SCAN-000.md` (this file) + `docs/context/CONTEXT-PLAN.md`.
- Layer 1 (partition docs):
  - `docs/context/CONTEXT-001-core-runtime-provenance.md`
  - `docs/context/CONTEXT-002-datasets-metrics-eval.md`
  - `docs/context/CONTEXT-003-patching-mechanistic.md`
  - `docs/context/CONTEXT-004-repro-packaging-verification.md`
  - `docs/context/CONTEXT-005-benchmark-perf-harness.md`
- Layer 2 (recommended future deep dives):
  - `docs/context/DEEPDIVE-cpt-specificity.md`
  - `docs/context/DEEPDIVE-sae-patching.md`
  - `docs/context/DEEPDIVE-mechanistic-backends.md`

## Step 6 - Scan limitations

- No dependency install or runtime execution was performed in this scan.
- Churn analysis is weak because git history is shallow in this checkout.
- Service-oriented sections from the original generic template (routes/authz/tenants/CSRF/upload APIs) are mostly not applicable to this research CLI repository.
