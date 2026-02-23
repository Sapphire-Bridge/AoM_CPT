# Paper modes / runners

There are two supported ways to run “paper-style” suites in this repo:

## 1) Canonical submission suite (paper-cited artifacts)

This is the canonical end-to-end runner referenced by the evidence contract (`AoM_JoLLLI/AoM_evidence_contract.md`):

```bash
bash scripts/run_submission_full_strong.sh
```

Default outputs:
- `results_submission_full/` (CSVs + per-artifact `*.manifest.json` + `results_report.md` + `RUN_MANIFEST.json`)
- `tables_submission_full/` (strictly regenerated LaTeX tables)

This suite includes:
- Behavioral AoM eval + CPT (core paper models)
- CPT target-specificity control run (8-model suite)
- AoM-CF intervention-span patching (canonical CF suite)
- AoM-CF shift vs substitution-invariant patching (E10e; `cf_patching_shift_vs_subinv.csv`)
- AoM-COH pseudo-ablation patching (GPT-2 + Qwen2.5 per-checkpoint artifacts; `coh_patching.csv` + `coh_patching_qwen*.csv`)
- Strict table regeneration

## 2) Preset runner (`run_paper.py`)

`run_paper.py` provides hardware-tuned presets:

```bash
python scripts/run_paper.py smoke
python scripts/run_paper.py m1max
python scripts/run_paper.py a100 --attn_behavioral flash_attention_2
```

`run_paper.py` writes `RUN_MANIFEST.json` in its results directory and logs the same provenance fields into the generated CSVs.

## Dataset bundle

The canonical hardened paper dataset is checked in at `data_paper_hardened_v2/` and is SHA-gated by:
- `data_paper_hardened_v2/DATASET_MANIFEST.json`

---

For clean-room reproduction from a pinned commit, use:

```bash
bash scripts/final_repro_cleanroom.sh
```
