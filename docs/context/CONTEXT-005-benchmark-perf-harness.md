# Project Context 005 - Benchmark and Performance Harness (mechanistic layer)

## Scope (files covered)

- `aom/mechanistic/backends/base.py:40` - Backend protocol interfaces for induction experiments.
- `aom/mechanistic/backends/hf_eager.py:27` - Hugging Face eager backend implementation.
- `aom/mechanistic/backends/transformer_lens.py:251` - TransformerLens backend implementation and optional cross-check extras.
- `aom/mechanistic/attention_recorder.py:18` - Hook-based attention pattern capture and induction statistics.
- `aom/mechanistic/induction.py:9` - Core induction-score math and validation routines.
- `aom/mechanistic/logit_lens.py:119` - Logit-lens trace computation for selected token pair.
- `aom/mechanistic/synthetic.py:27` - Synthetic token-sequence generation for induction probing.
- `aom/reporting.py:502` - Results directory scanner and markdown summary report.
- `scripts/report_results.py:14` - CLI wrapper around reporting module.
- `AoM_JoLLLI/AoM_evidence_contract.md:59` - Optional evidence IDs for induction/logit-lens outputs.

## Architecture Overview

- Mechanistic benchmarking is library-driven (protocol interfaces), not currently exposed as a first-class top-level CLI in tracked files.
- Two interchangeable backend implementations exist: native HF eager and TransformerLens.
- Attention recorder computes induction statistics inside forward hooks to avoid storing full attention traces by default.
- Synthetic repeated-sequence generation is used to create controlled induction-test inputs.
- Logit-lens utilities operate on hidden-state trajectories with optional final-norm handling policy.
- Reporting module classifies arbitrary CSV files and generates unified markdown inventory/summaries.

## Key Flows

- Flow A: induction backend load
- Input: model id/path, device, dtype, HF safety flags.
- Validation: model/tokenizer/architecture discovery and vocab sizing.
- Side effects: loads model and tokenizer into chosen backend representation.
- Output: `BackendLoadResult` with architecture, layer count, vocab info, and backend version.
- Entry: `aom/mechanistic/backends/hf_eager.py:30`, `aom/mechanistic/backends/transformer_lens.py:254`.

- Flow B: synthetic induction batch execution
- Input: `base_len`, `repeats`, `batch_size`, `n_batches`, seed.
- Validation: allowed token IDs and sequence constraints.
- Side effects: repeated/control forward passes under recorder.
- Output: collector samples for per-layer/head induction stats.
- Entry: `aom/mechanistic/backends/hf_eager.py:89`.

- Flow C: hook-based attention capture
- Input: target layers and model architecture.
- Validation: attention tensor shape, finiteness, and row-sum checks.
- Side effects: records score/mass/fraction samples per mode (`repeat` or `control`).
- Output: sample tensors used for bootstrap summaries.
- Entry: `aom/mechanistic/attention_recorder.py:176`.

- Flow D: logit-lens trace extraction
- Input: prompt, token pair IDs, lens mode, analysis position.
- Validation: prompt non-empty, token IDs distinct, single-token selection helper if needed.
- Side effects: none (pure read/compute over model states).
- Output: `LogitLensTrace` with per-state-point deltas and final diff.
- Entry: `aom/mechanistic/logit_lens.py:119`.

- Flow E: result scanning/reporting
- Input: results directory containing heterogeneous CSVs.
- Validation: classify file kinds from headers and names.
- Side effects: none (reads files, writes report in wrapper).
- Output: markdown inventory and per-kind summaries.
- Entry: `aom/reporting.py:502`, `scripts/report_results.py:29`.

## Security / Invariants

- Induction attention tensors must be rank-4, finite, non-negative, and row-normalized.
- Enforced in `validate_attention_weights` (`aom/mechanistic/induction.py:9`).

- Current induction metric assumes `repeats == 2`.
- Enforced in recorder constructor (`aom/mechanistic/attention_recorder.py:40`).

- Synthetic token exclusion cannot remove all tokens.
- Enforced in `build_allowed_ids` (`aom/mechanistic/synthetic.py:16`).

- Single-token continuation requirements are explicit for logit-lens helpers.
- Enforced in `encode_single_token_id` and `select_single_token_continuation` (`aom/mechanistic/logit_lens.py:42`).

- Backend loaders pass through HF safety flags (`local_files_only`, `trust_remote_code`).
- Enforced by backend load signatures and kwargs wiring (`aom/mechanistic/backends/base.py:43`, `aom/mechanistic/backends/hf_eager.py:44`).

## Interfaces

### Backend protocol interfaces

- `InductionBackend.load(...) -> BackendLoadResult`
- `InductionBackend.run_batches(...) -> BackendRunResult`
- Protocol defined at `aom/mechanistic/backends/base.py:40`.

### Core dataclass interfaces

- `BackendLoadResult` fields include model/tokenizer handles plus architecture, layer count, vocab, dtype/device metadata.
- `BackendRunResult` returns collector + optional extras (`aom/mechanistic/backends/base.py:35`).
- `LogitLensTrace` and `LogitLensPoint` define logit-lens output structure (`aom/mechanistic/logit_lens.py:19`).

### Reporting interfaces

- `generate_results_report(results_dir, max_rows_per_csv, include_file_inventory) -> str`.
- CSV kind classification recognizes:
- AoM eval files via `aom_composite`.
- induction files via `induction_advantage` + `layer` + `head`.
- logit-lens files via `lens`/`logit_diff` signatures.
- Source: `aom/reporting.py:192`.

## Observability

- `aom/reporting.py` provides model-level summary tables from raw CSVs without requiring schema-specific scripts for every output.
- Logit-lens and induction summaries include range/mean stats useful for regression checks (`aom/reporting.py:680`, `aom/reporting.py:722`).
- Evidence contract tracks optional mechanistic artifact IDs (`E10`, `E11`) as non-core claims (`AoM_JoLLLI/AoM_evidence_contract.md:53`).

## Operational Notes / Gotchas

- TransformerLens backend requires optional dependency; backend throws explicit runtime error when missing (`aom/mechanistic/backends/transformer_lens.py:266`).
- Attention recorder toggles `_attn_implementation` to `eager` during capture and restores it afterward (`aom/mechanistic/attention_recorder.py:75`).
- `baseline_mode` semantics differ (`shuffle` vs `offset0`); interpretation of control distribution depends on this setting.
- Report generation is file-system-driven; stale CSV artifacts in a results directory can affect summaries if not cleaned.

## Risks / Limitations

- No tracked top-level CLI currently wires the full induction/logit-lens pipeline end-to-end in this scan.
- `TODO(verify):` identify canonical script(s) that generate `results/induction_*.csv` and `results/logit_lens_*` artifacts for open-source release docs.
- Hook-based implementations can break with upstream model API changes in attention forward signatures.
- Cross-backend comparability (HF vs TransformerLens) is partially addressed, but not guaranteed for every architecture/version combination.
