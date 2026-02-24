# CONTEXT-INDEX - Required Read Order and Task Routing

Date: 2026-02-23
Scope: Open-source research companion repo (`AoM_CPT`)

## Required start sequence

1. `docs/context/CONTEXT-INDEX.md` (this file)
2. `docs/context/CONTEXT-PLAN.md`
3. Task-specific `docs/context/CONTEXT-*.md` files listed below

## Task routing

- Core runtime/config/provenance/security work:
  - Read `docs/context/CONTEXT-001-core-runtime-provenance.md`
- Dataset/schema/loader/metric work:
  - Read `docs/context/CONTEXT-002-datasets-metrics-eval.md`
- Patching/intervention/mechanistic work:
  - Always read `docs/context/CONTEXT-003-mechanistic-patching.md`
- Repro/table/evidence/packaging work:
  - Read `docs/context/CONTEXT-004-repro-packaging-verification.md`
- Benchmark/perf/mechanistic backend harness work:
  - Read `docs/context/CONTEXT-005-benchmark-perf-harness.md`

## Mandatory mechanistic trigger keywords

If a task includes any of the following, `docs/context/CONTEXT-003-mechanistic-patching.md` is required:
- `patching`
- `intervention`
- `activation patching`
- `counterfactual`
- `coherence patching`
- `mechanistic`

## Fallback when scope is mixed

- Read all matching context files before editing code.
- If uncertain, read `docs/context/CONTEXT-001-core-runtime-provenance.md`, `docs/context/CONTEXT-003-mechanistic-patching.md`, and `docs/context/CONTEXT-004-repro-packaging-verification.md`.
