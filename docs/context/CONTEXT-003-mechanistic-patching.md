# Project Context 003 - Causal Patching and Mechanistic Controls (intervention runtime)

## Scope (files covered)

- `aom/interventions/activation_patching.py:11` - Patch-site dataclasses, architecture registry, and hook-based patch forward pass.
- `aom/interventions/patching/base.py:70` - Generic protocol abstraction and aggregate patching runner.
- `aom/interventions/patching/cf_protocol.py:22` - CF intervention-swap case builder and skip rules.
- `aom/interventions/patching/coh_protocol.py:37` - COH constraint/irrelevant pseudo-ablation case builder.
- `aom/token_spans.py:21` - Substring-to-token-span mapping with contiguity requirement.
- `aom/token_diff.py:36` - Minimal divergent token-region detection between sequences.
- `aom/metrics/disamb.py:485` - DISAMB CPT context-swap patching summary.
- `aom/metrics/disamb.py:652` - Target-specificity off-target control protocol.
- `aom_cf_patching.py:75` - Standalone AoM-CF patching CLI and manifest path.
- `aom_coh_patching.py:73` - Standalone AoM-COH patching CLI and manifest path.
- `aom_eval.py:421` - Integrated CPT patching path (`cpt_*`) and specificity path (`cpt_spec_*`).
- `aom/metrics/sae_patching.py:100` - SAE feature-space patching entrypoint (optional path).

## Architecture Overview

- Low-level patching is hook-based hidden-state replacement at decoder block outputs (`resid_post` semantics).
- Protocol classes own case construction (donor/receiver/span/strata); base runner owns repeated layer execution and aggregation.
- CF and COH protocols are explicit and separately versionable, with reason-coded skips.
- DISAMB patching has two families:
- context-swap sweep over layers (`cpt_*` fields)
- fixed-depth target-specificity with off-target controls (`cpt_spec_*` fields)
- Patching execution assumes deterministic scoring pipeline from base evaluator modules.
- SAE patching path is opt-in and requires explicit external bundle configuration.

## Key Flows

- Flow A: DISAMB context-swap CPT sweep (`aom_eval.py --run_patching`)
- Input: DISAMB minimal pairs + optional layer list.
- Validation: token-span alignment and optional token-id match requirement.
- Side effects: per-layer effect/sham aggregates and direction-level maxima.
- Output: `cpt_mean_max_effect`, `cpt_effect_layer_*`, skip counts.
- Core path: `aom/metrics/disamb.py:485`.

- Flow B: target-specificity control (`aom_eval.py --run_patching_specificity`)
- Input: fixed layer (or `depth_frac`), buffers, `position_window`, `selection_seed`.
- Validation: deterministic fallback strategy for control span selection.
- Side effects: stratified metrics by control strategy.
- Output: `cpt_spec_mean_signed_delta*`, `cpt_spec_n_directions_ctrl_patched_*`.
- Core path: `aom/metrics/disamb.py:652`.

- Flow C: CF protocolized activation patching (`aom_cf_patching.py`)
- Input: CF dataset and protocol config (`span_mode`, `include_expected_effects`, `max_total_len_delta`).
- Validation: divergence-span extraction, length constraints, type checks.
- Side effects: protocol skip-reason accounting and stratified outputs.
- Output: `cf_patch_*` row namespace + run manifest.
- Core path: `aom/interventions/patching/cf_protocol.py:47`, `aom_cf_patching.py:424`.

- Flow D: COH protocolized activation patching (`aom_coh_patching.py`)
- Input: COH triplets (`main`, `ablate_relevant`, `ablate_irrelevant`).
- Validation: triplet presence, divergence spans, pseudo-donor alignment.
- Side effects: condition-stratified effects with `effect_sign=-1.0` (degradation expected).
- Output: `coh_patch_*` row namespace + run manifest.
- Core path: `aom/interventions/patching/coh_protocol.py:64`, `aom_coh_patching.py:403`.

## Security / Invariants

- Patching requires eager attention implementation.
- Enforced by hard runtime check in patching CLIs (`aom_cf_patching.py:305`, `aom_coh_patching.py:290`).

- Patch-site layer/token shape constraints are explicit.
- Enforced in `forward_with_patched_block_output_span` (`aom/interventions/activation_patching.py:211`).

- Target span must map to a contiguous token interval.
- Enforced in `token_span_for_substring` (`aom/token_spans.py:46`).

- CF protocol skip reasons are deterministic and auditable.
- Enforced via `CaseSkip` reasons (`aom/interventions/patching/cf_protocol.py:57`).

- COH protocol requires complete triplets per `base_id`.
- Enforced by `missing_triplet` skip (`aom/interventions/patching/coh_protocol.py:91`).

- Target-specificity control selection is deterministic, not effect-optimized.
- Enforced by seeded RNG (`aom/metrics/disamb.py:769`).

## Interfaces

### Patching control interfaces

- `aom_eval.py` patching flags:
- `--run_patching`, `--patch_layers`, `--patch_allow_token_id_mismatch`.
- `--run_patching_specificity`, `--patch_specificity_layer`, `--patch_specificity_depth_frac`.
- `--patch_specificity_buffer`, `--patch_specificity_donor_buffer`, `--patch_specificity_position_window`, `--patch_specificity_seed`.
- Definitions: `aom_eval.py:855` onward.

- CF patching CLI flags:
- `--span_mode {divergent_only,divergent_plus_downstream,left_aligned_truncated}`.
- `--include_expected_effects`, `--max_total_len_delta`.
- Definitions: `aom_cf_patching.py:132` onward.

- COH patching CLI flags:
- `--max_total_len_delta`, `--patch_layers`.
- Definitions: `aom_coh_patching.py:130` onward.

### Output field interfaces

- DISAMB CPT sweep outputs under `cpt_*`:
- `mean_max_effect`, `mean_sham_max_effect`, `flip_rate_at_best_layer`, `effect_layer_*`, `sham_effect_layer_*`, direction counts.
- Source: `aom/metrics/disamb.py:629`.

- Specificity outputs under `cpt_spec_*`:
- signed/absolute deltas, win rates, strategy-specific submetrics, control coverage counters.
- Source: `aom/metrics/disamb.py:845`.

- CF protocol outputs under `cf_patch_*` from generic runner:
- protocol metadata, skip-reason digest/json, stratum-prefixed summaries, comparison effect sizes.
- Source: `aom/interventions/patching/base.py:255`.

- COH protocol outputs under `coh_patch_*` mirror same base-runner schema with condition strata.

## Observability

- Patching runners emit per-model/seed summary lines with key effect metric (`aom_cf_patching.py:489`, `aom_coh_patching.py:467`).
- Skip reasons are exported as both human-readable JSON and SHA digest for compact traceability (`aom/interventions/patching/base.py:262`).
- Case counts distinguish protocol-skips vs span-mismatch skips (`aom/interventions/patching/base.py:260`).
- Full run manifests include policy settings, dataset hashes, and status summary in the same schema used by base evaluator.

## Operational Notes / Gotchas

- `device_map` with patching is intentionally constrained in A100 preset runner (`scripts/run_paper.py:911`).
- `require_token_id_match=True` improves semantic comparability but can reduce coverage.
- `left_aligned_truncated` CF span mode trades strict alignment for higher case retention.
- COH protocol uses pseudo-ablated donor IDs built by token replacement; it does not run full donor forward on ablated text directly.
- Off-target specificity interpretation depends strongly on strategy composition; always inspect strategy counts.

## Risks / Limitations

- Hook-based patching may be fragile under future transformer architecture changes not covered by registry.
- Layerwise sweeps are computationally expensive and can dominate runtime on larger models.
- `TODO(verify):` SAE patching and decomposition outputs are present in code paths but not part of the main canonical run in this scan.
- There is conceptual overlap between `aom/metrics/disamb.py` patching helpers and protocolized patching framework; future refactors should avoid divergence.
