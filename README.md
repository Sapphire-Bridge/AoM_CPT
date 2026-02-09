# AoM Prototype (Appearance of Meaning)

Research-prototype evaluator for AoM competences and CPT-style causal tests:

- AoM-DISAMB: context-sensitive disambiguation (minimal pairs)
- AoM-CF: minimal-pair intervention sensitivity (directional preference shift)
- AoM-COH: discourse-level coherence / constraint tracking
- CPT (optional): context-swap activation patching on internal states

## Quickstart

Run a small offline smoke test (random small GPT-2 config; no downloads):

```bash
pytest -q
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

Clean-room / “one command” reproducibility check (exports a clean repo snapshot, installs pinned deps from `requirements.lock.txt`, runs the suite, runs evidence checks, and builds a replication bundle):

```bash
bash scripts/final_repro_cleanroom.sh
```

## Paper-mode runner (smoke / M1Max / A100)

This repo includes a convenience runner that generates a hardened “paper dataset” (with CF shams + COH controls) and runs reproducible evaluation presets:

```bash
# Offline end-to-end smoke check (builds a tiny local model; no downloads)
python scripts/run_paper.py smoke

# Long local run tuned for Apple Silicon (MPS)
python scripts/run_paper.py m1max

# Full run intended for CUDA GPUs (e.g. A100s)
python scripts/run_paper.py a100 --attn_behavioral flash_attention_2
```

## CF / COH causal patching runners

Separate CLIs implement the causal interventions for AoM-CF and AoM-COH:

```bash
python aom_cf_patching.py --config configs/cf_patching_gpt2_paper.yaml
python aom_coh_patching.py --config configs/coh_patching_gpt2_paper.yaml
```

These write `results/*.csv` plus `results/*.manifest.json` with provenance (git commit, HF commit hash, versions, argv hash, dataset SHA-256s, wall time, seeds).

## Tables

Generate LaTeX tables from `results/` artifacts:

```bash
make tables
```
