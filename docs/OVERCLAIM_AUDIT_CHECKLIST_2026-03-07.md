# Overclaim Audit Checklist

Date: 2026-03-07
Scope: `AoM_JoLLLI/`, root docs, and `docs/context/` hits for:

- `mean_max_effect` / display-label variant `Mean max effect`
- `mean_argmax_layer` / display-label variant `Mean argmax layer`
- `best layer`
- `peak layer`
- `strongest layer`
- `semantic specificity`
- `specificity`
- `cross-model`
- `cross-family`
- `GPT-2`
- `Qwen`
- `FP64`
- `float64`
- `precision`

Method notes:

- I treated label variants like `Mean max effect` and `Mean argmax layer` as hits for the underscored field names when they clearly refer to the same metric.
- For manuscript tables, I list the table as the unit rather than every row.
- For the evidence contract, I list affected table rows because those are the claim-bearing units.

## Findings

1. High: the SDH manuscript/table interpretation does not match the canonical checked-in artifact.
- `AoM_JoLLLI/AoM_paper.md:121-122`, `AoM_JoLLLI/AoM_paper.md:170`, `AoM_JoLLLI/AoM_paper.md:322`, and Table 3 rows at `AoM_JoLLLI/AoM_paper.md:324-331` describe SDH as mostly near-zero/negative, with only `Qwen2.5-3B` robustly positive and `N=100` directions per model.
- The cited artifact `results_submission_full/cpt_specificity_disamb_only.csv` instead records `cpt_spec_n_directions_target_patched=104` for every model and positive `cpt_spec_mean_signed_delta` for every non-`gpt2` row:
  - `gpt2`: `-0.2897`
  - `Qwen2.5-0.5B`: `+0.6605`
  - `Qwen2.5-1.5B`: `+0.5438`
  - `Qwen2.5-3B`: `+0.9708`
  - `Qwen3-4B`: `+0.9188`
  - `Llama-3.2-1B`: `+0.2577`
  - `Llama-3.2-3B`: `+0.2129`
  - `Llama-3.1-8B`: `+0.3324`
- This changes the qualitative story from "only one model robust positive" to "the canonical run mostly favors target over control."

2. High: the paper claims strategy-sensitive SDH results, but the canonical SDH artifact uses only one control-selection strategy.
- `AoM_JoLLLI/AoM_paper.md:122`, `AoM_JoLLLI/AoM_paper.md:235`
- In `results_submission_full/cpt_specificity_disamb_only.csv`, every model has:
  - `cpt_spec_n_directions_ctrl_patched_matched_token=104`
  - `cpt_spec_n_directions_ctrl_patched_same_index=0`
  - `cpt_spec_n_directions_ctrl_patched_same_index_relaxed=0`
- So the checked-in canonical run does not show strategy composition varying within the cited artifact. If supplementary runs exist, the manuscript should cite them directly instead of implying that Table 3 is strategy-mixed.

3. Medium: `best layer` wording is ambiguous relative to the implemented metric.
- `AoM_JoLLLI/AoM_paper.md:227`, `AoM_JoLLLI/AoM_paper.md:310`
- The implementation computes a per-direction argmax layer (`l_i*`) and then reports `cpt_mean_argmax_layer` as the mean of those per-direction maxima. `flip_rate_at_best_layer` also uses each direction's own maximizing layer.
- Readers can easily misread `best layer` as one shared model-level optimum.

4. Medium: `specificity` is used for distinct constructs that are not the same thing.
- `AoM_JoLLLI/AoM_paper.md:318-319` labels CF and COH relevance-controlled contrasts as `causal specificity`, even though SDH `target-specificity` is a different protocol and artifact family.
- This is a conceptual broadening rather than a numeric error, but it makes the claims sound more localization-specific than the evidence warrants.

5. Low: `cross-family` sounds broader than the actual comparison set.
- `AoM_JoLLLI/AoM_paper.md:3`
- The line is not false, but the new comparison named there is only `GPT-2` plus `Qwen2.5`, so the phrase reads broader than the empirical span.

## Checklist

### Manuscript sentences

- [ ] `AoM_JoLLLI/AoM_paper.md:3`
  - Terms: `cross-family`, `GPT-2`, `Qwen`
  - Risk: Low
  - Note: valid but rhetorically broad for a two-family comparison.

- [ ] `AoM_JoLLLI/AoM_paper.md:11`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: supported by Table 2 / Figure 1 style evidence.

- [ ] `AoM_JoLLLI/AoM_paper.md:13`
  - Terms: `specificity`
  - Risk: Medium
  - Note: derivative of the SDH interpretation that is not supported by the current canonical Table 3 artifact.

- [ ] `AoM_JoLLLI/AoM_paper.md:30`
  - Terms: `specificity`
  - Risk: Low
  - Note: protocol-and-constraint framing is conservative.

- [ ] `AoM_JoLLLI/AoM_paper.md:33`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: support present in `aom_eval.csv`.

- [ ] `AoM_JoLLLI/AoM_paper.md:34`
  - Terms: `specificity`
  - Risk: Medium
  - Note: the contribution claim depends on the SDH interpretation that conflicts with the checked-in SDH artifact.

- [ ] `AoM_JoLLLI/AoM_paper.md:58-61`
  - Terms: `specificity`
  - Risk: Low
  - Note: protocol description only.

- [ ] `AoM_JoLLLI/AoM_paper.md:70`
  - Terms: `specificity`
  - Risk: Medium
  - Note: as a prediction this is acceptable, but it is not what the canonical artifact now shows for most models.

- [ ] `AoM_JoLLLI/AoM_paper.md:94`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: simple methods description.

- [ ] `AoM_JoLLLI/AoM_paper.md:109`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: supported by Table 1 / `aom_eval.csv`.

- [ ] `AoM_JoLLLI/AoM_paper.md:117`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: supported by `aom_eval.csv`.

- [ ] `AoM_JoLLLI/AoM_paper.md:119`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: descriptive scaling summary; numerically supported.

- [ ] `AoM_JoLLLI/AoM_paper.md:121`
  - Terms: `Qwen`
  - Risk: High
  - Note: contradicted by `cpt_specificity_disamb_only.csv`, which shows positive deltas for most non-`gpt2` models.

- [ ] `AoM_JoLLLI/AoM_paper.md:122`
  - Terms: `specificity`, `best layer`
  - Risk: High
  - Note: the strategy-sensitivity clause is unsupported by the canonical artifact; the `best sweep layer` wording is also easy to overread given the per-direction argmax implementation.

- [ ] `AoM_JoLLLI/AoM_paper.md:124`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: supported by COH patching artifacts.

- [ ] `AoM_JoLLLI/AoM_paper.md:126`
  - Terms: `Qwen`
  - Risk: Low
  - Note: supported by Table 4.

- [ ] `AoM_JoLLLI/AoM_paper.md:129`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: matches `cf_patching_shift_vs_subinv.csv`.

- [ ] `AoM_JoLLLI/AoM_paper.md:130`
  - Terms: `Qwen`
  - Risk: Low
  - Note: qualified and artifact-backed.

- [ ] `AoM_JoLLLI/AoM_paper.md:131`
  - Terms: `Qwen`, `mean max effect`
  - Risk: Low
  - Note: Appendix F table supports the type-specific heterogeneity claim.

- [ ] `AoM_JoLLLI/AoM_paper.md:137`
  - Terms: `Qwen`
  - Risk: Low
  - Note: summary sentence; CF and COH clauses are supported.

- [ ] `AoM_JoLLLI/AoM_paper.md:146`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: supported by COH patching results and explicitly protocol-bounded.

- [ ] `AoM_JoLLLI/AoM_paper.md:170`
  - Terms: `specificity`
  - Risk: High
  - Note: "near zero or negative under some control-selection strategies" is not supported by the canonical SDH artifact set checked in here.

- [ ] `AoM_JoLLLI/AoM_paper.md:191`
  - Terms: `Qwen`
  - Risk: Low
  - Note: limitation statement; conservative.

- [ ] `AoM_JoLLLI/AoM_paper.md:199`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Medium
  - Note: the CPT clause is supported, but the SDH clause inherits the Table 3 / canonical-artifact mismatch.

- [ ] `AoM_JoLLLI/AoM_paper.md:203`
  - Terms: `Qwen`
  - Risk: Low
  - Note: future-work sentence, not an empirical overclaim.

- [ ] `AoM_JoLLLI/AoM_paper.md:227`
  - Terms: `mean max effect`, `best layer`
  - Risk: Medium
  - Note: metric definition is accurate, but `best layer` should be clarified as per-direction rather than global.

- [ ] `AoM_JoLLLI/AoM_paper.md:235`
  - Terms: `specificity`
  - Risk: High
  - Note: protocol definition is fine, but the "strategy-stratified diagnostics" clause overstates what the canonical artifact actually contains.

- [ ] `AoM_JoLLLI/AoM_paper.md:243`
  - Terms: `mean max effect` (table context immediately below)
  - Risk: Low
  - Note: supported by Appendix F table.

- [ ] `AoM_JoLLLI/AoM_paper.md:252`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: qualified sensitivity check; not outcome-based by its own description.

### Manuscript captions and tables

- [ ] `AoM_JoLLLI/AoM_paper.md:300` — Table 1 caption
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: descriptive only.

- [ ] `AoM_JoLLLI/AoM_paper.md:299-305` — Table 1
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: values match `aom_eval.csv`.

- [ ] `AoM_JoLLLI/AoM_paper.md:309` — Table 2 caption
  - Terms: `GPT-2`, `Qwen`, `Mean max effect`
  - Risk: Low
  - Note: caption is faithful to the metric construction.

- [ ] `AoM_JoLLLI/AoM_paper.md:310-314` — Table 2
  - Terms: `best layer`, `Mean argmax layer`, `Mean max effect`, `Qwen`
  - Risk: Medium
  - Note: numbers match `aom_eval.csv`, but `best layer` can be misread as a single shared optimum.

- [ ] `AoM_JoLLLI/AoM_paper.md:318` — Figure 2 title
  - Terms: `specificity`
  - Risk: Medium
  - Note: broadens `specificity` beyond SDH target-specificity.

- [ ] `AoM_JoLLLI/AoM_paper.md:319` — Figure 2 caption
  - Terms: `specificity`
  - Risk: Medium
  - Note: same conceptual broadening; numerically okay for CF/COH.

- [ ] `AoM_JoLLLI/AoM_paper.md:321-322` — Table 3 title + caption
  - Terms: `specificity`
  - Risk: High
  - Note: `N directions per model = 100` conflicts with `cpt_spec_n_directions_target_patched=104` in the canonical CSV.

- [ ] `AoM_JoLLLI/AoM_paper.md:321-331` — Table 3
  - Terms: `specificity`, `Qwen`
  - Risk: High
  - Note: the qualitative pattern does not match `results_submission_full/cpt_specificity_disamb_only.csv`.

- [ ] `AoM_JoLLLI/AoM_paper.md:332-344` — Table 4
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: values match CF/COH patching artifacts.

- [ ] `AoM_JoLLLI/AoM_paper.md:245-250` — Appendix F table
  - Terms: `Mean max effect`, `GPT-2`, `Qwen`
  - Risk: Low
  - Note: supports the Qwen2.5-1.5B heterogeneity discussion.

### Evidence contract rows

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:25` — `C4`
  - Terms: `specificity`
  - Risk: Low
  - Note: code-level fallback description is accurate.

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:30` — `C9`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low
  - Note: row is narrowly phrased.

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:33-34` — `R0`, `R3`
  - Terms: `specificity`, `Qwen`, `GPT-2`
  - Risk: Low
  - Note: operational provenance claims only.

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:35` — `E1`
  - Terms: `Qwen`
  - Risk: Low

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:37` — `E3`
  - Terms: `mean_max_effect`, `mean_argmax_layer`
  - Risk: Low
  - Note: this row correctly frames `cpt_mean_argmax_layer` as a depth proxy, not a literal universal best layer.

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:38` — `E4a`
  - Terms: `Qwen`
  - Risk: Low

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:39-44` — `E5a` to `E5f`
  - Terms: `specificity`, `Qwen`
  - Risk: Low
  - Note: these rows point to the correct artifact fields; the overclaim problem is in the paper's interpretation, not the contract wording.

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:48-52` — `E10a` to `E10e`
  - Terms: `mean_max_effect`, `GPT-2`, `Qwen`
  - Risk: Low
  - Note: rows are appropriately artifact-bounded.

- [ ] `AoM_JoLLLI/AoM_evidence_contract.md:53` — `E11b`
  - Terms: `GPT-2`, `Qwen`
  - Risk: Low

### Other docs

- [ ] `README.md:96`
  - Terms: `GPT-2`
  - Risk: None
  - Note: operational example only.

- [ ] `README.md:133-171`
  - Terms: `Qwen`
  - Risk: None
  - Note: command examples only.

- [ ] `README.md:201`
  - Terms: `specificity`
  - Risk: None
  - Note: references the artifact set name only.

- [ ] `PAPER_VERIFICATION_GUIDE.md:101`
  - Terms: `specificity`
  - Risk: None
  - Note: file path only.

- [ ] `PAPER_VERIFICATION_GUIDE.md:105-108`
  - Terms: `Qwen`
  - Risk: None
  - Note: expected-output inventory only.

- [ ] `AoM_JoLLLI/GITHUB_SUPPLEMENTARY_INDEX.md:9`
  - Terms: `Qwen`
  - Risk: Low
  - Note: interpretive-scope note is appropriately cautious.

- [ ] `docs/context/CONTEXT-003-mechanistic-patching.md:12,15,25,35,38,76,85-86,101,104,127`
  - Terms: `specificity`, `mean_max_effect`
  - Risk: Low
  - Note: internal engineering documentation. The "inspect strategy counts" advice is correct in principle, but the current canonical SDH artifact does not actually vary strategy.

- [ ] `docs/context/CONTEXT-004-repro-packaging-verification.md:5,113`
  - Terms: `specificity`
  - Risk: None

- [ ] `docs/context/CONTEXT-PLAN.md:66,227`
  - Terms: `specificity`
  - Risk: None

- [ ] `docs/context/SCAN-000.md:125,143`
  - Terms: `specificity`
  - Risk: None

## No-hit terms

- `peak layer`
- `strongest layer`
- `semantic specificity`
- `cross-model`
- `FP64`
- `float64`
- `precision`
