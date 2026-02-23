# Project Context 002 - Datasets, Validation, and Metric Computation (AoM domain)

## Scope (files covered)

- `aom/data/schemas.py:7` - Dataclass contracts for `DisambPair`, `CounterfactualPair`, `CoherenceItem`.
- `aom/data/validate.py:8` - Prompt/continuation boundary checks and semantic constraints.
- `aom/data/loaders.py:21` - JSONL loading + validation + manifest return path.
- `aom/data/dataset_manifest.py:75` - Per-dataset digest/count/error manifest builder.
- `aom/data/bundle_manifest.py:62` - Bundle-manifest SHA validation and stable bundle-id computation.
- `aom/metrics/disamb.py:152` - AoM-DISAMB metric and patching-specific metric helpers.
- `aom/metrics/counterfactual.py:14` - AoM-CF metric splits and preference-shift logic.
- `aom/metrics/coherence.py:14` - AoM-COH metric plus group/paired-delta reporting.
- `aom/metrics/composite.py:9` - Composite metric from primary sub-metrics.
- `aom/metrics/cue_leakage.py:119` - Cue-overlap diagnostic features for DISAMB stratification.
- `scripts/generate_data.py:25` - Canonical JSONL generator for DISAMB/CF/COH suites.
- `data/README.md:11` - Human-readable dataset field contract.
- `data_paper_hardened_v2/DATASET_MANIFEST.json:1` - Pinned canonical bundle metadata.

## Architecture Overview

- Domain data is JSONL-first, with explicit dataclasses and validation rather than implicit schema inference.
- Loading path returns both parsed items and dataset manifests (hashes, row counts, invalid samples).
- Metric modules are independent and return flat dicts suitable for CSV row merging.
- Confidence intervals use shared bootstrap helpers (`MetricValue` semantics, empty/non-finite handling).
- Composite metric intentionally depends only on three primary metrics (`disamb.accuracy`, `cf.shift_direction_accuracy`, `coh.constraint_accuracy`).
- Dataset bundles are content-addressed via per-file SHA map, not manifest-file metadata alone.

## Key Flows

- Flow A: JSONL ingest and row-level validation
- Input: dataset path + role (`disamb`/`cf`/`coh`).
- Validation: JSON object type, schema-specific semantic checks, prompt-boundary checks.
- Side effects: invalid row sample capture (capped) and dataset digest/count calculation.
- Output: `(items, DatasetManifest)` or `DatasetLoadError` when `error_policy=raise`.
- Entry: `aom/data/loaders.py:21`.

- Flow B: metric computation per suite
- Input: loaded items + model/tokenizer/device.
- Validation: metrics return `MetricValue` validity flags and reasons for empty/non-finite samples.
- Side effects: none (pure compute modules).
- Output: dict of scalar metrics + CI fields.
- Entrypoints: `compute_aom_disamb`, `compute_aom_cf`, `compute_aom_coh`.

- Flow C: composite aggregation
- Input: disamb/cf/coh metric dicts.
- Validation: missing-policy behavior (`fail`, `nan`, `ignore`).
- Side effects: none.
- Output: `MetricValue` for `aom_composite`.
- Entry: `aom/metrics/composite.py:9`.

- Flow D: dataset-bundle integrity gate
- Input: `DATASET_MANIFEST.json` + selected dataset file paths.
- Validation: file SHA-256 equality and optional bundle-id consistency.
- Side effects: none.
- Output: bundle info fields (`dataset_bundle_manifest_sha256`, `dataset_bundle_id`, etc.).
- Entry: `aom/data/bundle_manifest.py:62`.

## Security / Invariants

- Prompt/continuation token-boundary hygiene is enforced during validation.
- Enforcement: `_validate_prompt_cont_boundary` (`aom/data/validate.py:8`).

- `CounterfactualPair.expected_effect` is constrained to `shift|invariant|graded`.
- Enforcement: `validate_counterfactual_pairs` (`aom/data/validate.py:65`).

- Shift items must align label semantics (`base`, `cf`, and `contrast_labels`).
- Enforcement: `aom/data/validate.py:86`.

- Dataset-manifest count arithmetic is invariant.
- Enforcement in run-manifest validation path: `n_rows_total == n_rows_valid + n_rows_invalid` (`aom/run_manifest.py:301`).

- Bundle SHA checks are strict string-length and equality comparisons.
- Enforcement: `_get_file_sha` and `validate_bundle_manifest` (`aom/data/bundle_manifest.py:26`, `aom/data/bundle_manifest.py:107`).

- Bootstrap helper never silently treats invalid samples as valid values.
- Enforcement: `MetricValue(valid=False, reason=...)` on empty/non-finite sample (`aom/stats.py:43`).

## Interfaces

### Input schema interfaces

- DISAMB row schema (`DisambPair`):
- `pair_id`, `target`, `target_occurrence`, `a`, `b`, `choices`, optional `metadata`.
- Source: `aom/data/schemas.py:14`, `data/README.md:13`.

- CF row schema (`CounterfactualPair`):
- `item_id`, `base`, `cf`, `choices`, `intervention_type`, optional `contrast_labels`, `expected_effect`.
- Source: `aom/data/schemas.py:30`, `data/README.md:38`.

- COH row schema (`CoherenceItem`):
- `item_id`, `context`, `valid_continuations`, `invalid_continuations`, `constraint_type`, optional `group`.
- Source: `aom/data/schemas.py:53`, `data/README.md:55`.

### Output metric interfaces

- DISAMB primary fields:
- `accuracy`, `mean_margin`, plus cue-overlap splits and CI fields.
- Source: `aom/metrics/disamb.py:232`.

- CF primary fields:
- `shift_direction_accuracy`, split label accuracies, shift magnitudes, and CI fields.
- Source: `aom/metrics/counterfactual.py:117`.

- COH primary fields:
- `constraint_accuracy`, `pairwise_auc`, `mean_gap`, group-level and paired-delta fields.
- Source: `aom/metrics/coherence.py:130`.

- Composite field:
- `aom_composite` with `valid/n/reason` semantics in caller.
- Source: `aom/metrics/composite.py:9`.

### Dataset manifest interfaces

- Per-dataset manifest fields:
- `role`, `path`, `sha256`, `size_bytes`, row counts, `schema_name`, `error_policy`, `invalid_samples`.
- Source: `aom/data/dataset_manifest.py:47`.

- Bundle-manifest-derived fields:
- `dataset_bundle_manifest_sha256`, `dataset_bundle_manifest_name`, `dataset_bundle_id`.
- Source: `aom/data/bundle_manifest.py:115`.

## Observability

- Loader path captures first invalid samples (line + error type + message) for operator triage (`aom/data/loaders.py:33`).
- Evaluator emits dataset warnings when invalid rows exist but valid rows remain (`aom_eval.py:1122`).
- Per-run manifests embed dataset manifests under `datasets.*` for auditability (`aom_eval.py:1380`).
- `data_paper_hardened_v2/DATASET_MANIFEST.json` includes suite counts and per-file hashes for published artifact checks.

## Operational Notes / Gotchas

- `error_policy=warn_skip` can keep runs alive while still producing `WARN` status; do not treat this as clean PASS by default.
- `expected_effect=graded` exists in dataset generation but may be excluded by protocol configuration in CF patching runs.
- COH paired deltas require triplet grouping via `item_id` base prefix (`__` split); malformed ids break paired analyses.
- Continuations should include tokenizer-safe leading spaces where relevant (`scripts/generate_data.py:48`, `data/README.md:29`).
- `compute_bundle_id` is stable against manifest metadata churn, but `dataset_bundle_manifest_sha256` is not.

## Risks / Limitations

- Generated templates can carry lexical cues despite hardening efforts; keyword baseline is diagnostic, not exhaustive.
- Metric dicts are intentionally wide and evolving; downstream consumers should not hardcode a minimal field set without validation.
- COH and CF split metrics are interpretation-sensitive; claims should cite exact fields and item counts.
- `TODO(verify):` if future datasets add new `expected_effect` values, validation and context docs must be updated together.
