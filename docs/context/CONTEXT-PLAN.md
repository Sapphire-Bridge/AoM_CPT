# CONTEXT-PLAN - Execution Plan for Context Docs

Date: 2026-02-22
Scope: Open-source research companion repo (`AoM_CPT`)

## Goal

Produce maintainable context files under `docs/context/` that let a new engineer or LLM make safe, provenance-aware changes without reverse-engineering the entire codebase.

## Required read routing (for humans and agents)

Read order:
1. `docs/context/CONTEXT-INDEX.md`
2. `docs/context/CONTEXT-PLAN.md`
3. Task-specific files from `docs/context/CONTEXT-INDEX.md`

Mandatory rule:
- If a task includes patching, intervention, counterfactual/coherence patching, or mechanistic analysis, read `docs/context/CONTEXT-003-mechanistic-patching.md` before editing code.

## Non-negotiables

- No invention: every concrete claim must be traceable to code or versioned artifacts.
- Security/provenance first: document gates for model loading, dataset integrity, and manifest redaction before feature details.
- Stable identifiers only: use file paths, line anchors, CLI flags, field names, and module symbols.
- Keep docs skimmable: avoid narrative prose; keep sectioned bullets.
- Mark uncertainty explicitly as `TODO(verify): ...`.

## Which draft is better for this repo?

Short answer:
- Use the second prompt as the first pass (scan + evidence-based planning).
- Use the first prompt as the per-file output schema.

Why:
- The second prompt forces broad, cheap discovery and prevents hallucinated architecture.
- The first prompt gives better section shape for maintainable context files.
- This repo is not backend/frontend; the second prompt catches that mismatch early.

Recommended merged workflow:
1. Scan report (`SCAN-000`) from prompt 2.
2. Partition plan (`CONTEXT-PLAN`) from prompt 2.
3. Context files using prompt 1 section schema, adapted to actual repository domains.

## Partition adaptation

Original partition labels were backend/frontend-heavy. Adapted partitions for this repository:
- 001: Core runtime/config/provenance/security/observability.
- 002: Dataset schemas/loaders/metric computation pipeline.
- 003: Causal patching + mechanistic internals.
- 004: Repro scripts, table generation, evidence checks, packaging.
- 005: Benchmark/perf harness and mechanistic backend interfaces.

## Proposed doc set

Already generated in this pass:
- `docs/context/CONTEXT-INDEX.md`
- `docs/context/SCAN-000.md`
- `docs/context/CONTEXT-PLAN.md`
- `docs/context/CONTEXT-001-core-runtime-provenance.md`
- `docs/context/CONTEXT-002-datasets-metrics-eval.md`
- `docs/context/CONTEXT-003-mechanistic-patching.md`
- `docs/context/CONTEXT-004-repro-packaging-verification.md`
- `docs/context/CONTEXT-005-benchmark-perf-harness.md`

Optional Layer-2 follow-ups (only if repo grows):
- `docs/context/DEEPDIVE-cpt-specificity.md`
- `docs/context/DEEPDIVE-sae-patching.md`
- `docs/context/DEEPDIVE-mechanistic-backends.md`

## Standard context file template (optimized)

Use this exact section order for each `CONTEXT-00X` file:

1. `Project Context 00X — <Area Name> (<scope>)`
2. `Scope (files covered)`
- `path:line — responsibility`
3. `Architecture Overview`
- 1 to 8 bullets only
4. `Key Flows`
- Flow-style bullets with input -> checks -> outputs
5. `Security / Invariants`
- Explicit invariant + enforcement site
6. `Interfaces`
- CLI flags, input schemas, output fields, manifest fields, error/status codes
7. `Observability`
- logs, manifests, summaries, traceability fields
8. `Operational Notes / Gotchas`
- common failure patterns, performance caveats, defaults that surprise users
9. `Risks / Limitations`
- what can regress or become invalid quickly

## Context extraction function (pseudocode)

```text
function build_context_doc(area, anchors):
  doc = init_sections(template)

  for file in anchors:
    read target ranges only
    extract:
      - responsibilities
      - flow entrypoints
      - invariants and enforcement points
      - interfaces (flags/schemas/outputs)
      - observability hooks
      - operational caveats

  for claim in doc:
    if claim lacks explicit code/artifact support:
      replace with TODO(verify)

  run consistency checks:
    - no duplicated ownership across docs
    - no contradictory defaults/flags
    - invariant ownership points to one enforcing location

  keep concise; prefer bullets over prose
  write docs/context/CONTEXT-00X-*.md
```

## Per-doc build plans

### CONTEXT-001 (core runtime/provenance)

Scope anchors:
- `aom_eval.py:710`
- `aom/models/loader.py:32`
- `aom/repro.py:99`
- `aom/run_manifest.py:49`
- `aom/run_summary.py:72`
- `aom/config.py:15`
- `aom/prompting.py:79`

Capture:
- Config precedence (`--config` defaults vs CLI overrides).
- Model load/security flags (`--trust_remote_code`, `--local_files_only`).
- Run status semantics (`PASS/WARN/FAIL`) and manifest schema.
- UTC timestamps and argv redaction/hash behavior.

### CONTEXT-002 (datasets/metrics)

Scope anchors:
- `aom/data/schemas.py:7`
- `aom/data/validate.py:8`
- `aom/data/loaders.py:21`
- `aom/data/dataset_manifest.py:75`
- `aom/data/bundle_manifest.py:62`
- `aom/metrics/disamb.py:152`
- `aom/metrics/counterfactual.py:14`
- `aom/metrics/coherence.py:14`
- `aom/metrics/composite.py:9`

Capture:
- JSONL contracts and invalid-row behavior.
- Primary metric definitions and bootstrap units.
- Composite-metric missing policy behavior.
- Dataset SHA gate and bundle-id semantics.

### CONTEXT-003 (patching/mechanistic)

Scope anchors:
- `aom/interventions/activation_patching.py:11`
- `aom/interventions/patching/base.py:70`
- `aom/interventions/patching/cf_protocol.py:22`
- `aom/interventions/patching/coh_protocol.py:37`
- `aom/metrics/disamb.py:485`
- `aom_cf_patching.py:75`
- `aom_coh_patching.py:73`
- `aom/token_spans.py:21`
- `aom/token_diff.py:36`

Capture:
- Span alignment logic and skip reasons.
- Target vs sham/off-target controls.
- Patching preconditions (`eager` attention).
- Stratified outputs used by paper tables.

### CONTEXT-004 (repro/verification/release)

Scope anchors:
- `scripts/run_paper.py:640`
- `scripts/run_submission_full_strong.sh:215`
- `scripts/make_tables.py:17`
- `scripts/check_evidence_contract.py:36`
- `scripts/check_evidence_contract_fields.py:184`
- `scripts/build_replication_bundle.sh:127`
- `scripts/final_repro_cleanroom.sh:205`
- `tests/test_evidence_contract_fields.py:34`

Capture:
- Canonical artifact pipelines.
- Strict table provenance cross-check gate.
- Evidence-contract checks and required fields.
- Bundle packaging and clean-room workflow constraints.

### CONTEXT-005 (benchmark/perf harness)

Scope anchors:
- `aom/mechanistic/backends/base.py:40`
- `aom/mechanistic/backends/hf_eager.py:27`
- `aom/mechanistic/backends/transformer_lens.py:251`
- `aom/mechanistic/attention_recorder.py:18`
- `aom/mechanistic/induction.py:9`
- `aom/mechanistic/logit_lens.py:119`
- `aom/reporting.py:502`

Capture:
- Backend abstraction and attention recording assumptions.
- Synthetic sequence induction constraints and validation.
- Logit lens trace API and token-selection constraints.
- Reporting layer that scans outputs for summary dashboards.

## Work order (risk-first)

1. Security and provenance gates (loader flags, dataset SHA checks, manifest redaction).
2. Dataset validation and primary metric semantics.
3. Causal patching protocols (highest conceptual complexity).
4. Repro orchestration + strict evidence/table gates.
5. Optional benchmark/perf internals (mechanistic backends).

## Consistency checklist for future refreshes

- `dataset_bundle_id`, `git_commit`, and `argv_sha256` still required in CSVs used for tables/evidence checks.
- `run_status` and `run_summary` manifest schema unchanged.
- `trust_remote_code` default remains opt-in.
- `strict_data` and `data_error_policy` interaction remains documented correctly.
- Target-specificity strategy names remain exactly: `matched_token`, `same_index`, `same_index_relaxed`.
