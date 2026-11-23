# Materials for the first revision (JLLI, R1)

The revised manuscript reports the same archived runs as the original submission. The paper-cited result files (`results_submission_full/`) are in the replication bundle attached to release [v1.0.2](https://github.com/Sapphire-Bridge/AoM_CPT/releases/tag/v1.0.2) (`aom_replication_bundle_fast8h.tar.gz`).

For the revision, existing GPT‑2 items were rescored, the exploratory effect sizes in Appendix F were recalculated, and the SDH control positions were checked (Section 4.7). No new items were added. This folder holds those materials.

## `coh_gpt2_rerun/`

The 80 original COH stories rescored with GPT‑2 (revision `607a30d`), CPU, float32. The rerun reproduces the archived COH contrast. It also yields the 15 stories whose two replacement prompts have the same number of GPT‑2 tokens (Section 4.4 and Section 6 of the revised manuscript).

- `run.py`: the script that was run
- `triplets.csv`: margins, token counts and contrasts for each story
- `summary.json`: group values for all 80 stories, the 15 equal-length stories and the 5 state stories, plus the comparison with the archived value
- `selection.json`: the selection rule and the IDs in each subset (fixed from the tokenizer before scoring)
- `prompt_scores.json`: candidate scores for all 240 prompts
- `manifest.json`: command, code versions, data hash and model revision. Local paths are replaced by `<repo>`, `<bundle>` and `<hf_cache>`.

Command:

```bash
python revision_R1/coh_gpt2_rerun/run.py --repo . \
  --archive <bundle>/results_submission_full/aom_eval.csv \
  --snapshot <hf_cache>/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --out <output_dir>
```

## `single_item_scores/`

`score_examples.py` scores the two worked examples in the revised manuscript with the repository's own label scorer: the DISAMB pair `bank-lex-0` (Section 1.1) and the CF item `role-swap-5` (Section 2). Run it from the repository root:

```bash
python revision_R1/single_item_scores/score_examples.py
```

Expected output: for `bank-lex-0`, finance minus river is 5.308 after the finance passage and −2.148 after the river passage (ratios of about 200 and 9). For `role-swap-5`, the preference for "passenger" over "pilot" moves by 1.077.

## `appendix_f_check/`

`check_appendix_f.py` recomputes Cohen's d for the CF patching comparison after excluding the three noun-substitution controls (Appendix F). It reads only the archived `cf_patching_shift_vs_subinv.csv` and needs no model. With three excluded items, their individual values follow from the archived bootstrap interval and group mean. The script prints both the earlier values (pooled SD of the full 60:20 comparison) and the values reported in the revision (pooled SD recomputed for 60:17).

```bash
python revision_R1/appendix_f_check/check_appendix_f.py <bundle>/results_submission_full/cf_patching_shift_vs_subinv.csv
```

## `sdh_control_check/`

Checks which control positions the archived SDH protocol selected (Section 4.7, Appendix Table D1). Settings are those of the archived run: fixed depth l* = round(0.25·(L−1)), `matched_token` control, `buffer=2`, `donor_buffer=2`, `position_window=8`, `selection_seed=0`.

- `check_control_positions.py`: uses the repository's own selection function and needs only a tokenizer. For GPT‑2, Qwen2.5, Qwen3, Llama‑3.2 and Llama‑3.1 tokenizers it gives the same result. All 104 controls come from `matched_token`. In 44 directions the two passages are identical up to and including the control position, at the same index, so the control patch changes nothing. In 10 directions the control position is the final token of the prompt ("the"), where the next word is predicted.
- `sdh_reimplementation.py` (with `selection_lib.py`): reruns target and control patching at l* and writes one row per direction. The means reproduce Table D1 to four decimals for GPT‑2 (revision `607a30d`), Qwen2.5‑0.5B (`060db64`) and Llama‑3.2‑1B (`4e20de3`), CPU, float32.
- `rows_gpt2.json`, `rows_llama-3.2-1b.json`, `rows_qwen2.5-0.5b.json`: the per-direction values (`et` = E_target, `ec` = E_ctrl, `base` = unpatched margin, `cls` = control class).
- `breakdown.py` and `summary.txt`: means by control type. The bootstrap here uses 2000 resamples, so intervals differ slightly from Table D1. The means are identical.

The paper cites the GPT‑2 and Llama‑3.2‑1B rows. In these two models the mean control effect is 4.214 and 1.514 in the 10 final-token directions and −0.033 and −0.029 in the other 94.

```bash
python revision_R1/sdh_control_check/check_control_positions.py gpt2
python revision_R1/sdh_control_check/sdh_reimplementation.py <model snapshot path> gpt2 rows_gpt2.json
python revision_R1/sdh_control_check/breakdown.py revision_R1/sdh_control_check/rows_gpt2.json
```
