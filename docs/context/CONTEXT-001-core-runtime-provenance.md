# Project Context 001 - Core Runtime, Config, and Provenance (evaluation runtime)

## Scope (files covered)

- `aom_eval.py:710` - Main AoM evaluator CLI surface and config ingestion.
- `aom_eval.py:984` - Multi-model/multi-seed execution loop and partial artifact flushing.
- `aom_eval.py:1345` - Final manifest assembly and run-status handling.
- `aom/models/loader.py:32` - Central Hugging Face model/tokenizer load function.
- `aom/config.py:15` - JSON/YAML config loader and path resolution.
- `aom/repro.py:99` - Seeding and determinism mode enforcement (`strict`, `best_effort`, `off`).
- `aom/run_manifest.py:49` - Run-manifest builder and argv redaction helpers.
- `aom/run_manifest.py:108` - Run-manifest schema validator.
- `aom/run_summary.py:72` - Run success/failure/skip accounting and threshold evaluation.
- `aom/prompting.py:79` - Prompt rendering modes and prompt/system hashing.
- `aom/utils.py:19` - Logprob computation config (`dtype`, finite checks).

## Architecture Overview

- `aom_eval.py` owns orchestration; package modules provide single-purpose helpers (loading, metrics, manifests, repro).
- Runtime is CLI-first; no HTTP service layer and no long-lived worker process.
- Provenance is emitted both per-row (CSV) and per-run (`*.manifest.json`) with overlapping audit fields.
- Determinism and data-policy controls are first-class runtime knobs and are persisted to manifests.
- Prompt processing is explicit (`raw` vs `chat_template`) and writes hashes, not raw prompt text, into provenance.
- Run status is derived from explicit counters plus configured thresholds, not ad hoc exceptions.

## Key Flows

- Flow A: `aom_eval.py` run loop
- Input: CLI args/config (`aom_eval.py:710`).
- Validation: config keys and relative path resolution (`aom_eval.py:964`, `aom/config.py:37`, `aom/config.py:43`).
- Side effects: loads datasets/models, computes metrics, writes CSV incrementally (`aom_eval.py:1048`, `aom_eval.py:1257`).
- Output: final CSV + run manifest + process exit (`aom_eval.py:1292`, `aom_eval.py:1385`, `aom_eval.py:1387`).

- Flow B: model load with security gates
- Input: model id/path + flags (`--revision`, `--tokenizer_revision`, `--local_files_only`, `--trust_remote_code`).
- Validation: device-map/accelerate import path (`aom/models/loader.py:77`).
- Side effects: AutoTokenizer/AutoModel load and architecture detection (`aom/models/loader.py:59`, `aom/models/loader.py:99`).
- Output: `LoadedModel` with effective tokenizer revision and model commit hash (`aom/models/loader.py:23`).

- Flow C: determinism + seed behavior
- Input: `ReproConfig(seed, determinism)` from CLI defaults/flags (`aom_eval.py:1169`).
- Validation: strict determinism failure when unsupported (`aom_eval.py:1173`).
- Side effects: seeds Python/NumPy/Torch and toggles deterministic algorithms (`aom/repro.py:99`).
- Output: reproducibility metadata persisted in manifest (`aom_eval.py:1374`).

- Flow D: manifest generation and status
- Input: collected row(s), summary counters, dataset/provenance metadata.
- Validation: statuses from `RunSummary.evaluate` + dataset warning promotion (`aom_eval.py:1312`, `aom_eval.py:1316`).
- Side effects: redacted argv and strict-JSON serialization (`aom/run_manifest.py:20`, `aom/run_manifest.py:92`).
- Output: `run_status` in `{PASS,WARN,FAIL}` and reason list (`aom_eval.py:1382`).

## Security / Invariants

- Remote-code execution is opt-in only.
- Enforced by CLI default: `--trust_remote_code` unset by default (`aom_eval.py:731`).
- Logged in outputs: `hf_trust_remote_code` (`aom_eval.py:378`).

- Offline/no-download mode is explicit.
- Enforced by `--local_files_only` flag (`aom_eval.py:769`, `aom/models/loader.py:50`).

- Sensitive prompt inputs are redacted from argv provenance.
- Enforced by `DEFAULT_REDACTED_FLAGS` and `redact_argv` (`aom/run_manifest.py:13`, `aom/run_manifest.py:20`).

- Manifest JSON must remain strict JSON (no NaN/Inf tokens).
- Enforced by `_to_jsonable` + `allow_nan=False` (`aom/run_manifest.py:75`, `aom/run_manifest.py:97`).

- Run-summary accounting invariants are checked explicitly.
- Enforced in `RunSummary._check_invariants` (`aom/run_summary.py:82`).

- Timestamps are UTC ISO strings.
- Enforced by `datetime.now(timezone.utc).isoformat()` usage (`aom_eval.py:531`, `aom/run_manifest.py:16`).

- Git provenance can be required.
- Enforced by default `--require_git` true (`aom_eval.py:958`, `aom/repro.py:49`).

## Interfaces

### CLI interfaces

- Primary CLI: `python aom_eval.py ...` (`aom_eval.py:710`).
- Config file input: `--config` supports JSON/YAML (`aom_eval.py:713`, `aom/config.py:15`).
- Error policy inputs:
- `--error_policy {raise,warn_skip,skip_silent}` (`aom_eval.py:781`)
- `--data_error_policy {raise,warn_skip}` (`aom_eval.py:788`)
- `--strict_data` and `--strict_errors` toggles (`aom_eval.py:795`, `aom_eval.py:801`).

### Output shapes

- CSV row is a flat key/value object containing:
- Model/provenance fields (`model`, `git_commit`, `argv_sha256`, dataset hashes).
- Metric namespaces prefixed `disamb_`, `cf_`, `coh_`, plus `aom_composite`.
- Optional CPT fields (`cpt_*`, `cpt_spec_*`, `sae_cpt_*`).
- See assembly site: `aom_eval.py:340` and updates at `aom_eval.py:389`, `aom_eval.py:435`, `aom_eval.py:515`.

- Manifest output is JSON object with:
- `manifest_version`, `created_at_utc`, `argv_redacted`, `results_row`, CSV provenance.
- Run policy/status fields (`error_policy`, `run_status`, `run_summary`, data policy).
- Assembly site: `aom_eval.py:1345`.

### Error/status interface

- Exit code `1` when `run_status == FAIL` (`aom_eval.py:1387`).
- Exit code `2` for parse/config fatal before runtime loop (`aom_eval.py:987`).
- `run_status` values: `PASS`, `WARN`, `FAIL` (`aom/run_summary.py:10`).

## Observability

- Runtime emits structured progress/failure prints to stdout/stderr (`aom_eval.py:541`, `aom_eval.py:1258`, `aom_eval.py:1282`).
- Partial CSV flush happens during long sweeps, reducing crash-loss risk (`aom_eval.py:1048`, `aom_eval.py:1257`).
- CSV row includes reproducibility keys (`git_commit`, `argv_sha256`, software versions, dataset bundle id).
- Manifest captures failure/skip type distributions through `run_summary` (`aom/run_summary.py:108`).

## Operational Notes / Gotchas

- Strict determinism is not supported on MPS; strict mode can hard-fail (`aom/repro.py:139`, `aom_eval.py:1173`).
- `device_map` loading may require `accelerate`; loader raises actionable ImportError (`aom/models/loader.py:80`).
- `strict_metrics` forces composite missing policy to `fail`, which can change pass/fail behavior (`aom_eval.py:393`).
- Manifest stores only the first row in `results_row` when CSV has multiple rows (`aom_eval.py:1331`).
- CSV schema is dynamic (union of row keys), so downstream tooling should not assume fixed column order (`aom_eval.py:684`).

## Risks / Limitations

- High coupling in `aom_eval.py` increases regression risk when adding new metric families.
- Similar orchestration logic is duplicated in `aom_eval.py`, `aom_cf_patching.py`, and `aom_coh_patching.py`; drift risk is non-trivial.
- Logging is print-based (no structured logger); machine parsing relies on CSV/manifests rather than log streams.
- `TODO(verify):` whether additional redaction flags beyond system prompts will be needed for future CLI options.
