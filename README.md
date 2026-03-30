# AoM_CPT

Compact public companion repository for the Appearance of Meaning behavioral and CPT intervention framework.

## What this repo is

`AoM_CPT` is the public reproducibility and verification surface for the AoM behavioral and CPT intervention strand. It contains the manuscript and evidence contract, the hardened paper dataset bundle, the maintained smoke/full reproduction runners, and the packaging/check scripts used to verify paper-facing artifacts.

## What this repo is not

- It is not the full mechanics-focused research stack for every related AoM analysis.
- It is not a complete archive of historical experiment layouts, exploratory scripts, or legacy config routes.
- It is not the only repository you should use to infer the full breadth of supporting AoM work.

## Relation to `AoM_mechanism`

`AoM_CPT` is the narrower companion repository for the behavioral and CPT intervention strand. `AoM_mechanism` is the broader mechanics-focused artifact and verification repository with a larger mechanistic evidence surface.

## How to check this repo

A minimal reviewer pass is:

1. create/activate a venv, then install dependencies with `python -m pip install -r requirements.txt`
2. run `python scripts/run_paper.py smoke`
3. run `python -m pytest -q`
4. inspect `results/paper_smoke/aom_eval.csv`, `results/paper_smoke/results_report.md`, and `results/paper_smoke/RUN_MANIFEST.json`
5. use [`PAPER_VERIFICATION_GUIDE.md`](PAPER_VERIFICATION_GUIDE.md) for the strict paper-facing verification path; field-level evidence-contract checks apply to full `results_submission_full/` artifacts, not the smoke run

## Paper

The repository companion manuscript and verification sources are:

- **The Appearance of Meaning: Context-Dependence and Semantic Competence in Transformer Architectures**
- Manuscript: [`AoM_JoLLLI/AoM_paper.md`](AoM_JoLLLI/AoM_paper.md)
- Evidence contract: [`AoM_JoLLLI/AoM_evidence_contract.md`](AoM_JoLLLI/AoM_evidence_contract.md)
- Verification guide: [`PAPER_VERIFICATION_GUIDE.md`](PAPER_VERIFICATION_GUIDE.md)
- Zenodo preprint: <https://zenodo.org/records/18907020>
- Abstract: in the manuscript section `Abstract` near the top of the file

## Canonical paper reproduction

Regenerate the paper-cited artifacts (results + manifests + strict tables):

```bash
bash scripts/run_submission_full_strong.sh
```

For a clean-room verification pass from a fresh exported commit:

```bash
bash scripts/final_repro_cleanroom.sh --repro-mode full_recompute
```

## Reproducibility modes

Replication bundling supports two explicit modes:

- `full_recompute`: rerun model evaluations/patching before building the archive.
- `frozen_artifacts`: reuse an already generated `results/` + `tables/` set and only validate/package.

Full recompute bundle build:

```bash
bash scripts/build_replication_bundle.sh \
  --mode submission_full_strong \
  --repro-mode full_recompute \
  --results-dir results_submission_full \
  --tables-dir tables_submission_full \
  --archive aom_replication_bundle.tar.gz
```

Frozen-artifact verification + packaging:

```bash
bash scripts/build_replication_bundle.sh \
  --mode submission_full_strong \
  --repro-mode frozen_artifacts \
  --results-dir results_submission_full \
  --tables-dir tables_submission_full \
  --archive aom_replication_bundle.tar.gz
```

## Environment (pinned release runtime)

The canonical full run records its effective runtime in `results_submission_full/RUN_MANIFEST.json` when you regenerate the paper artifact set. The strict paper-facing reproduction envelope is:

- Ubuntu `24.04`
- Python `3.12.7`
- `pip==24.2`
- `torch==2.5.1`
- `transformers==4.57.3`
- `tokenizers==0.22.1`
- `numpy==1.26.4`

Dependency surfaces:

- Reviewer / CI surface: `requirements.txt`
- Strict clean-room / paper reproduction surface: `requirements.pip.lock.txt`
- Provenance-only reference snapshot: `requirements.lock.txt`

Strict paper-facing environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.2
python -m pip install -r requirements.pip.lock.txt
```

`requirements.lock.txt` is a reference-machine snapshot retained for provenance recording. It contains Conda-only packages such as `anaconda-anon-usage` and `conda`, so it is not the pip install surface.

## Canonical dataset manifest

Use the hardened paper dataset bundle and manifest:

- Dataset directory: [`data_paper_hardened_v2/`](data_paper_hardened_v2/)
- Manifest: [`data_paper_hardened_v2/DATASET_MANIFEST.json`](data_paper_hardened_v2/DATASET_MANIFEST.json)
- Canonical bundle id: `fa2f39387339d26abd45912e31eede3b5f88aac4ed7bff20660262fcb46787ff`

## Quickstart

Run the minimal offline reviewer check (builds a tiny local model; no downloads):

```bash
python scripts/run_paper.py smoke
```

Run the default offline test suite:

```bash
python -m pytest -q
```

Generate a larger templated dataset (starting point; you should still curate for publication):

```bash
python scripts/generate_data.py --out_dir data --seed 0
```

To reproduce the original (cue-heavy) DISAMB templates:

```bash
python scripts/generate_data.py --out_dir data --seed 0 --disamb_mode easy
```

To include CF sham controls and COH ablation controls:

```bash
python scripts/generate_data.py --out_dir data --seed 0 --cf_include_shams --coh_include_controls
```

Run an evaluation on a local HuggingFace model (requires model files present):

```bash
python aom_eval.py \
  --model_name_or_path gpt2 \
  --local_files_only \
  --disamb_path data/disamb_pairs.jsonl \
  --cf_path data/counterfactual.jsonl \
  --coh_path data/coherence.jsonl \
  --run_patching \
  --csv_path results/aom_results.csv
```

Qwen example (downloads unless `--local_files_only` is set):

```bash
python aom_eval.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --csv_path results/qwen_behavioral.csv
```

Sweep multiple models into one CSV:

```bash
python aom_eval.py \
  --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B \
  --device mps \
  --torch_dtype float32 \
  --attn_implementation eager \
  --csv_path results/qwen_behavioral.csv
```

Notes:
- Metrics are log-prob based and deterministic in `model.eval()`; uncertainty is estimated via bootstrap over items.
- Log-probs are computed via `log_softmax` in `--logprobs_dtype` (default `float32`) for numeric stability; `--strict_finite` (default) fails fast on NaN/Inf.
- `scripts/generate_data.py` can generate sham CF controls and coherence ablation controls; see `data/README.md`.
- For publication-grade results, expand the datasets in `data/` (templates, paraphrases, lexical diversity) and report CIs.

## Reproducible runs (pinned artifacts + manifests)

`aom_eval.py` writes a JSON run manifest next to the CSV (`*.manifest.json`) to make runs replayable and auditable. The manifest records CSV provenance (`csv_sha256`, `csv_n_rows`) so you can detect post-hoc edits.

Pin Hugging Face artifacts (optional):

```bash
python aom_eval.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --revision <branch|tag|commit> \
  --tokenizer_revision <branch|tag|commit> \
  --csv_path results/qwen_pinned.csv
```

Offline / air-gapped use (no downloads):

```bash
python aom_eval.py --local_files_only --csv_path results/offline.csv
```

Security: `--trust_remote_code` is **off by default**. Only enable it if you trust the model repository, since it may execute custom code during loading. Run manifests redact `--system_prompt` and `--system_prompt_file` argv values by default.

## Token boundary hygiene (warn-only by default)

If prompts don’t end in whitespace/newline, tokenization at the prompt→continuation boundary can change when scoring separately (“token healing” risk). By default, `aom_eval.py` warns; you can make it strict with `--boundary_check error` or opt into deterministic normalization with `--normalize_boundaries`.

## Prompt modes (raw vs chat templates)

Default AoM datasets use full prompt strings:

- `--prompt_input full_prompt --prompt_mode raw` (default)

For chat models, you can interpret dataset prompt fields as user messages and render via the tokenizer’s chat template:

- `--prompt_input user_message --prompt_mode chat_template`
- Optional system prompt via `--system_prompt_file` (preferred) or `--system_prompt` (redacted in run manifests)

Run manifests store hashes only (e.g., `system_prompt_sha256`, `chat_template_sha256`) and never write raw prompt/system text to disk.

## Paper submission reproduction (canonical artifacts)

Regenerate the paper-cited artifact set (AoM eval + specificity + CF/COH patching + report + run manifest + strict tables) into the canonical directories (`results_submission_full/`, `tables_submission_full/`):

```bash
bash scripts/run_submission_full_strong.sh
```

Clean-room / “one command” reproducibility check (exports a clean repo snapshot, installs the strict pip lock from `requirements.pip.lock.txt`, runs the suite, runs evidence checks, and builds a replication bundle):

```bash
bash scripts/final_repro_cleanroom.sh --repro-mode full_recompute
```

Release-ready 8h gate (exact copy-paste command, from repo root):

```bash
TOTAL_BUDGET_SEC=28800 FIELDS_TIMEOUT_SEC=1200 STRICT_SHA=1 ALLOW_DIRTY=0 RESULTS_DIR=results_submission_full TABLES_DIR=tables_submission_full ARCHIVE_NAME=aom_replication_bundle_fast8h.tar.gz bash scripts/release_gate_fast_8h.sh
```

For frozen-artifact packaging from existing validated outputs:

```bash
bash scripts/build_replication_bundle.sh \
  --mode submission_full_strong \
  --repro-mode frozen_artifacts \
  --results-dir results_submission_full \
  --tables-dir tables_submission_full \
  --archive aom_replication_bundle.tar.gz
```

## Paper-mode runner (smoke / M1Max / CUDA preset)

This repo includes a convenience runner that generates a hardened “paper dataset” (with CF shams + COH controls) and runs reproducible evaluation presets:

```bash
# Offline end-to-end smoke check (builds a tiny local model; no downloads)
python scripts/run_paper.py smoke

# Long local run tuned for Apple Silicon (MPS)
python scripts/run_paper.py m1max

# Full high-memory CUDA preset (historical preset name: `a100`)
python scripts/run_paper.py a100 --attn_behavioral flash_attention_2
```

The `a100` mode name is a retained preset label for the high-memory CUDA path. Successful NVIDIA runs on other hardware should be reported as CUDA validation on the tested GPU, not as A100-specific validation.

Additional ad hoc CUDA validation was performed on an NVIDIA RTX A4500 using cached Hugging Face artifacts and `aom_eval.py` with `--device cuda`; the behavioral evaluation completed successfully for a 10-model non-8B subset. This is a CUDA operability check only. It does not replace the documented reviewer path (`python scripts/run_paper.py smoke`) or the strict paper-facing path (`bash scripts/final_repro_cleanroom.sh --repro-mode full_recompute`).

## Standalone causal patching CLIs

`aom_cf_patching.py` and `aom_coh_patching.py` remain available for experimentation, but the maintained public verification path goes through `scripts/run_paper.py`, `scripts/run_submission_full_strong.sh`, and `scripts/final_repro_cleanroom.sh`.

## Tables

Generate LaTeX tables from `results/` artifacts:

```bash
make tables
```

## Public release metadata

- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Zenodo checklist: [`docs/release/ZENODO_RELEASE_CHECKLIST.md`](docs/release/ZENODO_RELEASE_CHECKLIST.md)
- Zenodo metadata scaffold: [`.zenodo.json`](.zenodo.json)

## License

MIT. See [`LICENSE`](LICENSE).
