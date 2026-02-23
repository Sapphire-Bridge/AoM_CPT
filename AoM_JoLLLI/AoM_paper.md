The Appearance of Meaning: Context-Dependence and Semantic Competence in Transformer Architectures
Author: [Felix Borck / Goethe University / felix.borck@gmail.com]Acknowledgements / Funding: self funded
ESSLLI note: This manuscript develops and extends an ESSLLI 2025 Student Session presentation. The operational AoM evaluation framework, the CPT-style activation-patching protocol, and the cross-family empirical results reported here (GPT‑2 and Qwen2.5) are new to this submission (i.e., not previously published).
Keywords: large language models; transformers; context dependence; semantic competence; appearance of meaning; causal interpretability; activation patching; Kaplan; Braun; Wittgenstein; Kant

Evidence tags (for auditability): bracketed IDs like [E3], [R0], [C2] refer to rows in `AoM_evidence_contract.md`.

Abstract (≈200 words)
Large language models (LLMs) produce text that human interlocutors routinely treat as meaningful and context-sensitive. This paper does not address whether LLMs possess meaning proper. It isolates an empirical target: appearance of meaning (AoM), operationalized as three measurable competences: (i) context-sensitive disambiguation, (ii) sensitivity to meaning-altering vs. meaning-preserving interventions, and (iii) discourse-level constraint coherence. We then state a mechanistic hypothesis, the Context-Primacy Thesis (CPT): in transformer LMs, causal control of AoM-relevant preference margins is exerted by contextualized token-in-context states at specific depths, in the sense that intervening on those states (with weights fixed) produces donor-directed changes beyond sham baselines.
The framework supports sham counterfactual (CF) items, relevance-controlled coherence (COH) ablations, and CPT sham patching. [C6, C7, C3]
We evaluate GPT‑2 (124M) and Qwen2.5-{0.5B, 1.5B, 3B} using log-probability scoring and report suite metrics with bootstrap confidence intervals. [E2a, E1, C8]
AoM composite scores range from 0.8489 (Qwen/Qwen2.5-0.5B) to 0.9026 (Qwen/Qwen2.5-3B). [E2a, E1]
Context-swap activation patching on disambiguation minimal pairs yields donor-directed effects with sham baselines near zero (mean max effect: 0.3946 [0.3010, 0.5010] for GPT‑2; 1.6511 [1.2414, 2.1279], 1.8066 [1.4607, 2.2301], and 1.6861 [1.3043, 2.1170] across Qwen2.5-0.5B/1.5B/3B, respectively). [E3, E4a]
Intervention-span patching on a relevance-controlled counterfactual patching set (shift N=60 vs substitution-invariant N=20) yields shift > invariant separation for GPT‑2, Qwen2.5‑0.5B, and Qwen2.5‑3B (d = 0.68–1.18), but not for Qwen2.5‑1.5B (d ≈ 0), indicating heterogeneous relevance control across checkpoints under this protocol. [E10e, C10]
Constraint-sentence pseudo-ablation patching on the coherence suite yields a robust constraint–irrelevant degradation asymmetry across all four models (Cohen's d ranging from 1.27 to 1.53), providing causal evidence for coherence sensitivity under this intervention regime. [E10c, E10d, C9]
To probe the spatial structure of causal control, we test a supplementary Spatial Distribution Hypothesis (SDH) via a fixed-layer target-specificity protocol (depth_frac=0.25; position_window=8) across eight models (adding Qwen3‑4B and Llama‑3.x checkpoints). [C4, E5a, E5b, E5c, E5d, E5e, E5f]
Specificity is heterogeneous: only Qwen2.5‑3B shows a robust positive target–control delta, while most models show deltas near zero or negative. This constrains token-atomic interpretations of causal control and is consistent with a local, relational integration picture within the tested window. [E5a, E5b, E5c, E5d, E5e, E5f]

1. Introduction
Transformer LLMs produce language that competent speakers often judge as coherent and context-responsive. This behavioral profile motivates a question that does not require settling metaphysical debates: what mechanisms explain systematic meaning-like behavior under controlled contextual variation?
1.1. From “meaning” to an empirical target: appearance of meaning (AoM)
Appearance of meaning is a term of art. AoM is a competence profile that (i) tracks core aspects of semantic competence in human language use and (ii) is measurable under controlled context manipulations. We operationalize AoM as:
1. Context-sensitive disambiguation (AoM-DISAMB): systematic selection among ambiguous senses conditioned on surrounding cues.
2. Minimal-pair intervention sensitivity (AoM-CF): systematic output shifts under meaning-altering interventions, and relative invariance under meaning-preserving shams.
3. Discourse-level coherence (AoM-COH): constraint tracking across extended contexts, with stronger degradation under relevant ablations than under length-matched irrelevant ablations.
All three are computed from log-probability comparisons over labeled continuations. AoM targets controlled sensitivity patterns, not free-form generation quality. This corresponds to what Mahowald et al. (2024) term formal linguistic competence—text-internal statistical mastery of language structure—while explicitly bracketing functional linguistic competence involving world knowledge, reasoning, and grounding.
1.2. Mechanistic hypothesis: Context-Primacy Thesis (CPT)
Transformers compute contextualized states at every position. That fact alone does not decide between semantic theories. This paper states a causal hypothesis about proximate controllers:
Context-Primacy Thesis (CPT): In transformer language models, causal control of AoM-relevant preference margins is exerted by contextualized token-in-context states at specific depths, in the sense that intervening on those states (with weights fixed) produces donor-directed changes beyond sham baselines.
CPT is an architectural claim about causal control under intervention. It is not a claim that LLMs have meaning proper.^1
1.3. Contributions
1. Operationalization: an AoM competence profile (DISAMB/CF/COH) with explicit scoring and uncertainty.
2. Mechanistic thesis: CPT stated as a causal-control claim testable by intervention.
3. Method: controlled evaluation with shams and relevance-controlled ablations; deterministic log-probability scoring.
4. Evidence: AoM and DISAMB/CF patching results for GPT‑2 and Qwen2.5, plus COH pseudo-ablation patching across all four models; and an SDH target-specificity stress test across eight models.
1.4. Philosophical payoff and discipline
The paper is methodologically conservative. It treats AoM as an explanandum and asks which causal mechanisms in transformer computation control it. If AoM is granted as robust behavior, then dismissals that classify it as “mere surface” must identify an additional property, argue that it is competence-relevant, and state detectable signatures. This paper supplies an empirical anchor that makes those demands concrete.
The interpretive contrast is between a stability ideal—stable, context-independent meaning carriers plus context as parameter—and an architectural reality in which representations are contextualized throughout computation. The question here is mechanistic: what variables causally control AoM behavior in these systems?
^1 Footnote 1. Following Kant’s phenomena/noumena discipline (KrV A51/B75), we treat AoM as an empirical explanandum and bracket claims about meaning proper. The target is conditions for the appearance of competence, not metaphysical semantics.
2. Formal targets: AoM, CPT, and SDH
2.1. Preliminaries: scoring labeled continuations
Let (M) be a causal language model defining a conditional distribution over token sequences. For a prompt (x) and a candidate continuation (y=(y_1,\dots,y_T)), define the raw log-probability score:
[S_M(x,y) ;=; \sum_{t=1}^{T} \log P_M\!\left(y_t \mid x \oplus y_{<t}\right).]
Define the length-normalized score:
[\overline{S}_M(x,y) ;=; \frac{1}{T} S_M(x,y).]
Scoring used in reported runs. Reported results use length-normalized scoring (the evaluation flag no_length_norm=False), i.e., label scores are based on (\overline{S}_M) unless stated otherwise.
When a label corresponds to multiple candidate continuations (Y={y^{(1)},\dots,y^{(k)}}), label scores are aggregated via log-mean-exp:
[s(Y) ;=; \log\left(\frac{1}{k}\sum_{j=1}^{k} e^{\overline{S}_M(x,y^{(j)})}\right).]
In the reported AoM suites, labels are single-candidate (so log-mean-exp reduces to the single candidate’s score). The evaluator supports multi-candidate labels.
2.2. AoM-DISAMB: context-sensitive disambiguation
A disambiguation item (i) consists of an ambiguous word type (w_i), two contexts (x_i^{(A)}), (x_i^{(B)}), and sense-diagnostic continuations (y_i^{(A)}), (y_i^{(B)}).
Define the model’s predicted sense label in context (x) by:
[\widehat{s}_M(x) ;=; \arg\max_{s \in \{A,B\}} \overline{S}_M(x, y_i^{(s)}).]
DISAMB accuracy is computed over both sides of each minimal pair.
2.3. AoM-CF: minimal-pair intervention sensitivity
A counterfactual item (i) consists of a base prompt (x_i), an intervention prompt (x_i'), and two labeled continuations (a_i), (b_i), with an annotation indicating whether the intervention should flip the model’s preference (shift item) or preserve it (invariant/sham item).
Define the preference margin:
[\Delta_M(x_i) ;=; \overline{S}_M(x_i,a_i) - \overline{S}_M(x_i,b_i), \qquad\Delta_M(x_i') ;=; \overline{S}_M(x_i',a_i) - \overline{S}_M(x_i',b_i).]
Define the predicted preferred label as (\operatorname{pref}(x)=a) iff (\Delta_M(x_i) > 0), else (b).
Primary CF metric (reported): shift-direction accuracy computed on shift items only (N=60 in the canonical `aom_eval.csv` artifact). A shift item is correct iff (\operatorname{pref}(x_i)\neq \operatorname{pref}(x_i')). Invariant items (\operatorname{pref}(x_i)= \operatorname{pref}(x_i')) and graded items are reported separately via split metrics and perturbation-magnitude fields (N=60 invariant; N=20 graded).
We also report mean absolute preference changes (|\Delta_M(x_i')-\Delta_M(x_i)|) separately for shift vs invariant items to verify that shams induce smaller perturbations.
2.4. AoM-COH: discourse-level coherence and constraint tracking
A coherence item (i) consists of a context (x_i), a valid continuation (v_i), an invalid continuation (u_i), and constraint metadata.
The coherence decision uses mean-per-token scoring:
[\text{COH-ACC}_i(M) ;=;\mathbf{1}\!\left[\overline{S}_M(x_i,v_i) > \overline{S}_M(x_i,u_i)\right].]
Items are grouped into three matched conditions: main, ablate-relevant, and ablate-irrelevant (length-matched control). The ablations test whether degradation tracks informational role rather than context length reduction alone.
2.5. AoM composite
For compact reporting:
[\text{AoM}(M) ;=; \frac{1}{3}\left(\text{DISAMB}(M) + \text{CF}(M) + \text{COH}(M)\right).]
This composite is instrumental. It is not a definition of meaning.
2.6. CPT stated, predicted, and falsifiable
Let (h_{M,\ell,p}(x)) denote the hidden state vector at transformer block (\ell) and token position (p) when processing prompt (x). CPT predicts that interventions that replace token-in-context states at meaning-relevant locations—holding weights fixed—should change meaning-relevant output preferences.
We test CPT using context-swap activation patching on disambiguation pairs by patching the receiver’s hidden state at the ambiguous target span from a donor run, sweeping (\ell) across layers.
Supplementary hypothesis: spatial distribution (SDH)
In addition to CPT, we test a hypothesis about the spatial structure of causal control at a fixed depth:
SDH (Spatial Distribution Hypothesis): At a consolidation depth, disambiguation-relevant causal control is distributed across a local positional neighborhood around the ambiguous span, such that patching nearby off-target positions yields non-trivial donor-directed effects.
SDH is independent of CPT. CPT can be true with SDH false (token-local control). CPT can be true with SDH true (distributed local control). The target-specificity protocol in §4.7 and results in §5.7 test SDH directly.
Testable predictions (within this operational program)
* P1 (Causal control via patching; CPT). Patching a receiver’s target-span state with the donor’s corresponding state shifts the receiver’s preference toward the donor’s sense label, beyond sham/no-op baselines.
* P2 (Layer localization; CPT). Patching-induced effects show a reproducible depth profile with early-to-mid peaks rather than random diffuse changes.
* P3 (Informational role over length; AoM-COH control). Removing relevant constraints degrades coherence more than removing length-matched irrelevant context.
* P4 (Locality within window; SDH stress test). At fixed depth (\ell^*), off-target patches within the local window produce effects that are often comparable to target patches, and control strategy materially affects measured specificity.
Falsifiers (in the current experimental regime)
* F1 (Causal null; CPT). Patching effects indistinguishable from sham/no-op controls across layers.
* F2 (No localization; CPT). No stable layerwise structure; effects are non-replicable.
* F3 (Length dominance; AoM-COH control failure). Coherence degradation tracks truncation length more than relevance-controlled ablations.
* F4 (Global diffusion; SDH). If patches far outside the tested position window yield effects comparable to within-window patches, SDH (as a local distribution claim) is falsified. (Outside-window effects are not tested in this submission.)
2.7. What AoM, CPT, and SDH are not
* AoM is not defined as truth-tracking, grounding, normativity, or consciousness.
* CPT does not claim lexical information is absent. It is a claim about proximate causal control under intervention for these operational targets.
* SDH is not a claim of global positional symmetry. It is a local-window claim tied to an explicit protocol (§4.7).
* None of these settles reference, grounding, or normativity without additional detectable competence signatures.

3. Related work (minimal load-bearing set)
3.1. Context sensitivity: formal semantics and contextualized representations
Formal semantics provides explicit models of how context parameters and evolving discourse states affect interpretation (Kaplan 1989; Heim & Kratzer 1998; Groenendijk & Stokhof 1991; Veltman 1996). In transformer LMs, token representations vary strongly across contexts; embedding geometry is anisotropic and complicates naive similarity interpretations (Ethayarajh 2019). These are correlational observations; they motivate causal tests of whether and how context-sensitive information is used in generation.
3.2. Probing versus causal interpretability
Probing can show information is decodable, not that it is used (Belinkov & Glass 2019; Rogers et al. 2020). Causal mediation analysis applies counterfactual interventions on network-internal components to distinguish decodable from causally active information (Vig et al. 2020). Interchange interventions formalize this as counterfactual substitution on aligned internal variables across inputs, enabling causal abstraction tests (Geiger et al. 2021). The CPT patching protocol in §4.6 is an interchange intervention specialized to token-in-context hidden states and AoM preference margins; it does not perform a full mediation decomposition. Mechanistic interpretability work develops patching-style interventions and circuit frameworks (Elhage et al. 2021; Wang et al. 2023). The transformer-circuits view treats the residual stream as a shared communication channel, with attention heads iteratively writing context-conditioned information to token representations. Related work also shows that intervening on intermediate representations can redirect downstream generation in predictable ways (Li et al. 2021; Lindsey et al. 2025). Editing approaches intervene on internal associations (Meng et al. 2022). To connect causal effects to the “surface-statistics” objection, we report keyword baselines as a diagnostic. A stratified analysis in §5.2 evaluates model DISAMB accuracy on pairs where the keyword baseline fails, providing a direct test of whether performance depends on cue availability.
3.3. Surface-statistics objections, grounding, and operational methodology
A prominent worry is that LLM success reflects surface form statistics rather than semantic competence (Bender & Koller 2020; Bender et al. 2021). Lappin (2024) evaluates LLM strengths and weaknesses from a logic-and-semantics perspective, arguing that current models exhibit substantial but incomplete linguistic competence; this assessment within the present journal motivates finer-grained causal investigation of specific competence components. Relatedly, the vector grounding problem challenges inferences from vector representations to reference or world-involving content (Mollo & Milliere 2023). In contrast, inferential and functional accounts argue that substantial aspects of meaning can be realized without direct reference (Piantadosi & Hill 2022) and that current debates are continuous with classic disputes in philosophy of language and mind (Milliere & Buckner 2024a, 2024b). When metaphysical disputes stall, Turing-style methodology recommends specifying testable competence profiles (Turing 1950). Recent work documents strong LLM performance on complex evaluations (Katz et al. 2024; Jannai et al. 2023). This paper targets context-sensitivity signatures that bear directly on these debates, rather than general benchmark ranking.
3.4. Philosophical frameworks: Kaplan, Braun, Wittgenstein, Kant
Kaplan (1989) formalizes character/content using stable meaning rules plus contextual parameters; this is a semantic framework, not an implementation claim. Its decomposition nevertheless suggests a natural empirical question for mechanistic analysis: whether models route and integrate context variables in a stable, expression-linked manner. Braun (1995) critiques overly extensional or insufficiently structured character theories and emphasizes relational structure. Wittgenstein (1953) and Cavell (1979) stress use and normativity. Kant’s phenomena/noumena distinction supports methodological discipline: empirical competence profiles can be studied without treating them as decisive about meaning “in itself.”

4. Methods
This section reflects the AoM prototype evaluation framework and the logged run metadata in the provided result artifacts. Where a fact is not recorded, the paper states that explicitly.
4.1. Models evaluated
We evaluate four causal LMs spanning two transformer families for the full AoM + CPT layer-sweep protocol:
* openai-community/gpt2 (GPT‑2, 124M parameters; 12 transformer blocks)
    * Radford et al. (2019), Language Models are Unsupervised Multitask Learners (GPT‑2 technical report; OpenAI).
    * Checkpoint distributed via Hugging Face: https://huggingface.co/openai-community/gpt2
* Qwen/Qwen2.5-0.5B
    * Model card / checkpoint: https://huggingface.co/Qwen/Qwen2.5-0.5B
* Qwen/Qwen2.5-1.5B
    * Model card / checkpoint: https://huggingface.co/Qwen/Qwen2.5-1.5B
* Qwen/Qwen2.5-3B
    * Model card / checkpoint: https://huggingface.co/Qwen/Qwen2.5-3B
For the SDH target-specificity stress test (§4.7, §5.7), we additionally run the fixed-layer specificity protocol on:
* Qwen/Qwen3-4B — https://huggingface.co/Qwen/Qwen3-4B
* meta-llama/Llama-3.2-1B — https://huggingface.co/meta-llama/Llama-3.2-1B
* meta-llama/Llama-3.2-3B — https://huggingface.co/meta-llama/Llama-3.2-3B
* meta-llama/Meta-Llama-3.1-8B — https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
All evaluations run in deterministic inference mode (model.eval()), and all reported metrics use log-probabilities rather than sampled generations.
Software + hardware provenance (captured in the canonical run artifacts).
* Per-row provenance fields in `results_submission_full/*.csv` include platform, device/dtypes, bootstrap settings, dataset hashes, git commit, and Hugging Face revision/commit metadata. [R3]
* Per-artifact manifests (`results_submission_full/*.manifest.json`) record redacted argv, run status/summary, dataset validity counts/samples, and determinism metadata. [R0, C2]
* Paper tables are regenerated in strict mode from a single results directory; strict mode fails on missing/mismatched provenance to prevent mixing artifact sets. [C11]
4.2. Datasets
All datasets are JSONL with schema validation and tokenization boundary checks.
[C1]
4.2.1. Disambiguation dataset (AoM-DISAMB)
The disambiguation suite contains 52 minimal pairs (104 sides) across six ambiguous word types: [E2a, E1]
* bank, bat, spring, match, pitcher, mole.
Each item provides two contexts and a sense-diagnostic continuation for each sense.
Tokenization validation. The evaluator checks alignment of the ambiguous target span across minimal-pair contexts for scoring and patching. Misaligned directions are skipped. [C3]
Keyword baseline. A keyword baseline predicts senses from cue words. Its role is diagnostic, not competitive. [C5]
4.2.2. Counterfactual dataset (AoM-CF)
The counterfactual suite contains 140 base/intervention pairs: [E2a, E1]
* 60 shift items (expected preference flip)
* 60 invariant items (meaning-preserving shams; expected no flip)
* 20 graded items (not used in the primary AoM-CF shift-direction accuracy in Table 1; excluded from CF patching evaluation by configuration, but included in the coverage accounting in §5.8)
Each item contains two labeled continuations ((a_i,b_i)) and the expected flip/no-flip annotation.
4.2.3. Coherence dataset (AoM-COH)
The coherence suite contains 80 main items, each with matched ablate-relevant and ablate-irrelevant variants (total 240 contexts across the three conditions). The irrelevant ablation removes an unrelated sentence of comparable length. [E2a, E1]
4.3. Scoring and aggregation
Scoring uses log-softmax probabilities for numerical stability. Continuations are scored by mean per-token log-probability when length normalization is enabled (no_length_norm=False as recorded in the canonical result artifacts). [R3]
The evaluator uses:
* deterministic evaluation (model.eval()),
* log-prob computations in float32,
* strict finiteness checking.
Strict finiteness checks. In the logged runs reported here, strict_finite=True (fail on NaN/Inf). [R3]
Run provenance (canonical artifacts). Reported behavioral AoM and DISAMB patching numbers come from deterministic CPU runs with strict metric/data enforcement: platform `macOS-15.7.3-arm64-arm-64bit`; Python `3.12.7`; PyTorch `2.5.1`; Transformers `4.57.3`; Tokenizers `0.22.1`; `logprobs_dtype=float32`; `no_length_norm=False`; `strict_finite=True`; `bootstrap_n=1000`; `bootstrap_seed=42`; `ci=0.95`; `git_commit=ea24714787ac14d6641f15fc08483a463dfb70b9`. Dataset bundle name is `paper_hardened_v2` with bundle id `fa2f39387339d26abd45912e31eede3b5f88aac4ed7bff20660262fcb46787ff` and bundle manifest sha256 `deb9b00c165a9b8fafe9a7aa9080bdb1fc7f5d637ad9c148e825d2155a0e4a31` (`data_paper_hardened_v2/DATASET_MANIFEST.json`). Model checkpoint revisions logged per run are: `gpt2` `607a30d783dfa663caf39e06633721c8d4cfcd7e`; `Qwen/Qwen2.5-0.5B` `060db6499f32faf8b98477b0a26969ef7d8b9987`; `Qwen/Qwen2.5-1.5B` `8faed761d45a263340a0528343f099c05c9a4323`; `Qwen/Qwen2.5-3B` `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`. Dataset file checksums are anchored in `DATASET_MANIFEST.json`: `disamb_pairs.jsonl` `a9587c...adee1`; `counterfactual.jsonl` `e269fb...74bf`; `coherence.jsonl` `04e122...fc64`. Caveat: in the current `aom_eval.csv`, the Qwen2.5-3B row has blank per-row dataset sha256 fields; `DATASET_MANIFEST.json` is the authoritative checksum source.
4.4. Statistical uncertainty: bootstrap confidence intervals
Metrics report 95% bootstrap confidence intervals.
* Bootstrap replicates: 1000 in logged runs (bootstrap_n=1000 where recorded). [C8, R3]
* Resampling units:
    * DISAMB: minimal pairs (pair-averaged across the two sides).
    * CF / COH: items.
    * CPT and SDH: donor→receiver directions.
Legacy CSVs may omit some bootstrap configuration fields; the framework defaults above were used for the logged runs.
Seeds. Forward passes are run in evaluation mode with no sampling. Reported seed values therefore do not index stochastic model behavior; they affect bootstrap resampling and, where applicable, deterministic control-span selection procedures.
4.5. Controls
The framework includes:
* AoM-CF shams (invariant items should flip less than shift items),
* AoM-COH relevance-controlled ablations,
* CPT sham patching (no-op replacement),
* tokenization/alignment checks with explicit skipping.
4.6. CPT-style activation patching
The framework runs context-swap activation patching on the disambiguation suite.
What is patched. Hidden states at the ambiguous target span are recorded from a donor run and injected into a receiver run at transformer block (\ell) via forward hooks (weights fixed).
Layer sweep. The patch is applied across all transformer blocks (\ell = 0,1,\dots,L-1), producing a per-layer effect curve.
Patch-effect definition (reported units)
For a disambiguation direction (donor→receiver) and layer (\ell), let (s(\text{label})) be the label score computed as log-mean-exp over the label’s continuation set, where each continuation is scored by mean per-token log-probability(length-normalized by default; --no_length_norm disables this).
Define the margin for an expected label (y) as:
[\operatorname{margin}(y) ;=; s(y) ;-; \max_{y'\neq y} s(y').]
For CPT context-swap patching, define the donor-directed effect at layer (\ell) as:
[\operatorname{effect}_\ell ;=; \operatorname{margin}_{\text{patched}}(y_{\text{donor}});-;\operatorname{margin}_{\text{base}}(y_{\text{donor}}),]
where (y_{\text{donor}}) is the donor side’s gold sense label, and “patched” replaces the receiver’s post-block hidden states at the target span with the donor’s corresponding span states.
We report:
* Mean max effect. Per direction, compute (\max_\ell \operatorname{effect}_\ell) over the swept layers. Report the bootstrap mean and CI over directions.
* Flip rate at best layer. Per direction, at the argmax layer for (\operatorname{effect}_\ell), count a flip iff the unpatched receiver prediction is not (y_{\text{donor}}) and the patched prediction equals (y_{\text{donor}}). Report bootstrap mean and CI.
* Sham baseline. Same patch site, but replace with the receiver’s own hidden states (no-op). Reported analogously.
4.7. Target-specificity stress test (SDH)
This protocol tests SDH by comparing donor-directed effects from target-span patching to effects from off-target control-span patching at a fixed depth.
Fixed layer choice
For each model with (L) transformer blocks, we set a fixed “consolidation” layer index:
[\ell^* ;=; \operatorname{round}\!\big(\text{depth_frac} \times (L-1)\big),\quad \text{with } \text{depth_frac}=0.25 \text{ in reported runs.}]
This ensures cross-model comparability. It does not guarantee (\ell^*) equals the empirically best layer from a full sweep.
Position window and exclusion buffers
Control spans are selected within a local position neighborhood around the ambiguous span, operationalized as:
* cpt_spec_position_window = 8 tokens (setting used in reported runs).
Candidate control positions are filtered by exclusion buffers:
* receiver exclusion buffer: buffer = 2 tokens (exclude positions within ±2 tokens of the target span),
* donor exclusion buffer: donor_buffer = 2 tokens (default in reported runs).
Deterministic control selection and strategy fallback
Off-target controls are selected deterministically (selection_seed=0). Control selection uses a fallback strategy:
1. matched_token
2. same_index
3. same_index_relaxed
The strategy used materially affects measured specificity; results are therefore reported both aggregated (Table 3) and stratified (Table 3b). [C4, E5a, E5b, E5c, E5d, E5e, E5f]
Per-direction effects and summary statistics
At layer (\ell^*), for each donor→receiver direction (i), compute:
* (E_{\text{target},i} = \operatorname{effect}_{\ell^*}) with target-span patching,
* (E_{\text{ctrl},i} = \operatorname{effect}_{\ell^*}) with control-span patching,
* (\Delta_i = E_{\text{target},i} - E_{\text{ctrl},i}),
* (\text{win}_i = \mathbf{1}[\Delta_i > 0]).
Aggregate over directions using bootstrap means and 95% CIs:
* (E_{\text{target}}), (E_{\text{ctrl}}), (\Delta), and win rate.

4.8. Causal patching beyond disambiguation: COH and CF protocols
To test whether CPT-style causal leverage extends beyond disambiguation, we specify two additional activation patching protocols for the coherence and counterfactual suites. [C9, C10]

COH pseudo-ablation patching. For each COH item, we construct two pseudo-ablated variants that preserve length and alignment: (i) constraint-pseudo: replace the constraint-relevant sentence with length-matched padding tokens; (ii) irrelevant-pseudo: replace the irrelevant sentence with length-matched padding tokens. Coherence is measured as the margin between valid and invalid continuation scores. For constraint-span patching, we patch hidden states at the constraint sentence positions from constraint-pseudo into the main run; for irrelevant-span patching, we patch hidden states at the irrelevant sentence positions from irrelevant-pseudo into the main run. Sham patching replaces main-run states with main-run states (no-op). The primary diagnostic is the constraint–irrelevant degradation asymmetry. [C9]

CF intervention-span patching. For each CF item, the base prompt x and intervention prompt x' are tokenized, and the minimal contiguous token-level divergence (intervention span) is identified. In the reported runs, span alignment uses span_mode=left_aligned_truncated: when donor and receiver intervention spans differ in token count, we left-align and truncate to a shared span length. Hidden states at the aligned intervention span are cached from the x' (donor) run and patched into the x (receiver) run at each transformer block ℓ. The effect metric is the donor-directed preference margin shift (analogous to §4.6). Sham patching replaces receiver states with receiver states. Items with empty receiver (base) spans are skipped (empty_span); graded items are excluded by configuration. Skip counts are reported. For the relevance-controlled CF patching analysis (§5.8.2), we use a patching-specific dataset combining the 60 shift items from the canonical CF suite with 20 substitution-invariant controls designed to ensure non-empty base spans (E10e); this differs from the canonical 140-item CF suite used for behavioral AoM-CF metrics in §5.3. [C10]

Both protocols use the same scoring, bootstrap, and finiteness-checking infrastructure as the DISAMB patching described in §4.3–4.6. [C9, C10]

5. Results
Results are reported for:
* AoM + CPT sweep: GPT‑2 and Qwen2.5-{0.5B, 1.5B, 3B} (Tables 1–2). [E2a, E1, E3, E4a]
* SDH specificity stress test (fixed layer): eight models total (Table 3, Table 3b). [E5a, E5b, E5c, E5d, E5e, E5f]
* CF intervention-span patching with substitution-invariant controls (§5.8.2; shift N=60 vs invariant N=20; zero skips) for GPT‑2 and Qwen2.5-{0.5B, 1.5B, 3B} (Table 4). [E10e, C10]
* COH pseudo-ablation patching (§5.8) for GPT‑2 and Qwen2.5-{0.5B, 1.5B, 3B} (Table 4). [E10c, E10d, C9]
Unless stated otherwise, confidence intervals are 95% bootstrap CIs.
5.1. Behavioral AoM metrics
AoM composite scores:
* GPT‑2 (124M): 0.869
* Qwen2.5‑0.5B: 0.849
* Qwen2.5‑1.5B: 0.869
* Qwen2.5‑3B: 0.903
[E2a, E1]
[Table 1 about here] [E2a, E1]
5.2. AoM-DISAMB: high accuracy with a baseline caveat
DISAMB accuracies:
* GPT‑2 (124M): 0.856 [0.788, 0.914] (N = 52 pairs; 104 sides)
* Qwen2.5‑0.5B: 0.913 [0.856, 0.962]
* Qwen2.5‑1.5B: 0.933 [0.885, 0.971]
* Qwen2.5‑3B: 0.933 [0.885, 0.981]
[E2a, E1]
A keyword baseline attains 0.615 [0.558, 0.673] with pair-bootstrap CIs aligned to the DISAMB bootstrap unit (minimal pairs). [E2a, E1, C5]
Keyword stratification analysis. Of the 52 pairs, the keyword heuristic gets both sides correct on 12 and at least one side wrong on 40. On the 40 keyword-incorrect pairs, model DISAMB accuracy remains well above chance with CI lower bounds above 0.75: GPT‑2 0.838 [0.762, 0.912]; Qwen2.5‑0.5B 0.912 [0.850, 0.963]; Qwen2.5‑1.5B 0.900 [0.838, 0.963]; Qwen2.5‑3B 0.925 [0.863, 0.975]. The Qwen models show minimal degradation on the keyword-incorrect subset relative to overall accuracy (drops of 0.001–0.023), indicating that their disambiguation performance does not depend on cue-word availability. This is the strongest rebuttal available to the cue-leakage objection: models disambiguate accurately even on pairs where a keyword heuristic fails. [E2a, E1, C5, E11b]
DISAMB accuracy alone is therefore not the main philosophical load-bearer in this submission, but the keyword stratification demonstrates that the competence captured is not reducible to surface-cue exploitation. CPT patching and CF/COH controls carry the weight.
Mean DISAMB margins increase with capacity within Qwen2.5 (reported in the CSV artifacts), consistent with increased separation between senses under these templates. [E1]
5.3. AoM-CF: controlled sensitivity with sham separation
CF shift-direction accuracy:
* GPT‑2: 0.850 [0.750, 0.933] (N = 60 shift items)
* Qwen2.5‑0.5B: 0.733 [0.617, 0.833]
* Qwen2.5‑1.5B: 0.800 [0.700, 0.883]
* Qwen2.5‑3B: 1.000 [1.000, 1.000]
[E2a, E1]
Sham separation is visible in mean absolute preference changes (|\Delta(x')-\Delta(x)|): invariant items shift less than shift items across models, including GPT‑2. [E8]
5.4. AoM-COH: coherence tracks informational role, not only length
Main-group coherence accuracies:
* GPT‑2: 0.900 [0.838, 0.963]
* Qwen2.5‑0.5B: 0.900 [0.838, 0.963]
* Qwen2.5‑1.5B: 0.875 [0.800, 0.938]
* Qwen2.5‑3B: 0.775 [0.688, 0.863]
[E2a, E1]
Ablations show an informational-role pattern: ablate-relevant degrades more than length-matched ablate-irrelevant across all four models, with varying magnitude. [E7]
5.5. CPT-style activation patching: causal leverage for token-in-context control
Activation patching is run on the disambiguation suite for GPT‑2 and Qwen2.5 models reported in Table 2. Each has 104 patching directions total, with all 104 patched and no misalignment skips in these hardened runs. [E3, E4a, C3]
Primary CPT aggregates:
* Mean max effect (defined in §4.6):
    * GPT‑2: 0.395 [0.301, 0.501]
    * Qwen2.5‑0.5B: 1.651 [1.241, 2.128]
    * Qwen2.5‑1.5B: 1.807 [1.461, 2.230]
    * Qwen2.5‑3B: 1.686 [1.304, 2.117]
* Flip rate at best layer:
    * GPT‑2: 0.000 [0.000, 0.000]
    * Qwen2.5‑0.5B: 0.058 [0.019, 0.106]
    * Qwen2.5‑1.5B: 0.115 [0.058, 0.183]
    * Qwen2.5‑3B: 0.048 [0.010, 0.087]
* Mean argmax layer:
    * GPT‑2: 5.47 / (12-1) (≈50% depth)
    * Qwen2.5‑0.5B: 8.19 / (24-1) (≈36%)
    * Qwen2.5‑1.5B: 7.72 / (28-1) (≈29%)
    * Qwen2.5‑3B: 13.53 / (36-1) (≈39%)
* Sham patching baseline: near zero in all reported runs. [E3, E4a]
These results support CPT as a causal-control claim under this intervention regime. They do not establish exclusivity of the target span as the only causal locus. [E3, E4a]
[Table 2 about here] [E3, E4a]
5.6. Scaling summary (AoM + CPT sweep)
* CPT patching effects are present in GPT‑2 and Qwen2.5; within Qwen2.5, effect magnitudes are non-monotonic in these hardened runs. [E3, E4a]
* CF shift-direction accuracy scales strongly within Qwen2.5 on these items; GPT‑2 is competitive despite smaller size. [E2a, E1]
* COH ablations show an informational-role effect across models; main coherence does not show uniform monotonic scaling on this suite. [E7]
5.7. Target-specificity stress test: SDH results
This section tests SDH using the fixed-layer target-specificity protocol (§4.7). Results are measured at:
* (\ell^* = \operatorname{round}(0.25\times(L-1))),
* cpt_spec_position_window=8,
* exclusion buffers buffer=2, donor_buffer=2,
* deterministic selection (selection_seed=0).
[C4, E5a, E5b, E5c, E5d, E5e, E5f]
Table 3. Target-specificity analysis across models (SDH stress test). Reported effects are donor-directed margin shifts (\operatorname{effect}_{\ell^*}) as defined in §4.6, evaluated at fixed (\ell^*). [E5a, E5b, E5c, E5d, E5e, E5f]
[Table 3 about here] [E5a, E5b, E5c, E5d, E5e, E5f]
Findings.
1. Specificity is heterogeneous. Only Qwen2.5‑3B shows a robust positive (\Delta = E_{\text{target}} - E_{\text{ctrl}}) with CI excluding zero. Most models show (\Delta \approx 0) and win rates near 0.5. GPT‑2 and Llama‑3.1‑8B show negative deltas (control patches stronger than target patches at (\ell^*) under this protocol). [E5a, E5b, E5c, E5d, E5e, E5f]
2. No monotonic capacity claim is supported for specificity. Qwen2.5‑3B is positive; Qwen3‑4B is not. Llama deltas are non-positive across 1B–8B in these runs. [E5a, E5b, E5c, E5d, E5e, E5f]
3. Control strategy is a first-order factor. Stratified deltas differ sharply by control selection strategy (Table 3b). The same_index strategy yields consistently negative point-estimate deltas across models, though CIs include zero for most; same_index_relaxed yields positive deltas across most models. The qualitative contrast between strategies is robust even where individual CIs are wide. [E5a, E5b, E5c, E5d, E5e, E5f, C4]
Table 3b. Specificity deltas stratified by control selection strategy (means with 95% CIs; N per strategy shown).
[Table 3b about here] [E5a, E5b, E5c, E5d, E5e, E5f]
Interpretation. The fixed-layer specificity results constrain a token-atomic reading of causal control. Under a local-window protocol (position_window=8 with exclusion buffers), donor-directed effects are often not unique to target-span patching. This is consistent with local distribution of disambiguation-relevant control across nearby positions at (\ell^*). The result is protocol-conditional: it establishes local non-exclusivity within the tested window, not global diffusion. [E5a, E5b, E5c, E5d, E5e, E5f, C4]
Caveats (explicit).
* The test is at fixed (\ell^*), not at each model’s empirically best layer from a sweep. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
* Control selection is not guaranteed semantically neutral. The same_index strategy appears to select meaning-relevant positions in these templates, inflating control effects. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
* Effects outside the tested position window are not measured here. SDH is not tested as a global claim in this submission. [C4]

5.8. Causal patching beyond disambiguation: COH and CF
Sections 5.5–5.7 establish causal control via activation patching on the disambiguation suite. This section extends patching-style interventions beyond disambiguation to coherence and counterfactual sensitivity. We report COH pseudo-ablation patching and CF intervention-span patching as specified in §4.8 (summarized in Table 4). [C9, C10]

5.8.1. COH causal patching: constraint-directed degradation under pseudo-ablation patching
To test whether CPT-style causal leverage extends to coherence constraint tracking, we apply COH pseudo-ablation patching to the coherence suite (§4.8). [C9]

Method (summary; see §4.8). For each COH item, we form two pseudo-ablated variants that preserve length and alignment: constraint-pseudo (constraint sentence replaced with padding) and irrelevant-pseudo (irrelevant sentence replaced with padding). For constraint-span patching, we patch hidden states at the constraint-sentence positions from constraint-pseudo into the main run; for irrelevant-span patching, we patch hidden states at the irrelevant-sentence positions from irrelevant-pseudo into the main run, sweeping across layers ℓ. The effect metric is the (positive) degradation in the coherence margin (valid vs invalid continuation), and the primary diagnostic is the constraint–irrelevant degradation asymmetry. Sham patching is a no-op baseline. [C9]

Results (N = 80 constraint-span cases; 80 irrelevant-span cases; total 160 per model).
* GPT‑2 (124M): mean max effect 0.753 [0.651, 0.858]; constraint-span 1.094 [0.970, 1.219]; irrelevant-span 0.411 [0.305, 0.515]; Δ = 0.683 (d = 1.266); sham ≈ 0. [E10c]
* Qwen2.5‑0.5B: mean max effect 1.010 [0.837, 1.193]; constraint-span 1.687 [1.452, 1.945]; irrelevant-span 0.332 [0.255, 0.428]; Δ = 1.355 (d = 1.526); sham 0.009. [E10d]
* Qwen2.5‑1.5B: mean max effect 1.150 [0.964, 1.355]; constraint-span 1.838 [1.595, 2.133]; irrelevant-span 0.463 [0.354, 0.593]; Δ = 1.375 (d = 1.368); sham 0.019. [E10d]
* Qwen2.5‑3B: mean max effect 0.869 [0.725, 1.025]; constraint-span 1.403 [1.195, 1.627]; irrelevant-span 0.336 [0.276, 0.400]; Δ = 1.067 (d = 1.377); sham 0.010. [E10d]

Interpretation. Across the tested model families (GPT‑2 and Qwen2.5‑{0.5B, 1.5B, 3B}), patching pseudo-ablated states at constraint-relevant positions yields substantially larger coherence-margin degradation than patching at irrelevant positions. The resulting constraint–irrelevant asymmetry is large in every model (d = 1.27–1.53, computed over per-case max-over-layers effects for constraint vs irrelevant patching), while sham baselines remain near zero. This supports a relevance-controlled causal effect on coherence under this alignment-preserving pseudo-ablation protocol. [E10c, E10d, C9]

Scale note. Absolute effect magnitudes vary non-monotonically within Qwen2.5, so cross-model comparisons should emphasize the asymmetry direction and standardized effect sizes rather than raw deltas.

Padding caveat. COH pseudo-ablation uses padding-token replacements to preserve alignment, which is off-distribution relative to fluent text. Accordingly, we interpret the observed asymmetry as evidence of span-local causal sensitivity under this intervention protocol, not as definitive isolation of a discourse-constraint circuit. A priority robustness test is to repeat the analysis with fluent, semantically irrelevant distractor replacements matched for length and syntactic profile.

Device note. The GPT‑2 COH patching row was computed on CPU; Qwen2.5 rows were computed on MPS (Apple Silicon GPU). Sham baselines are near zero on both devices and effects are large; exact floating-point values may differ by device without changing the qualitative pattern.

5.8.2. CF causal patching: shift vs substitution-invariant controls under intervention-span patching
To test whether CPT-style causal leverage extends to counterfactual sensitivity with relevance controls, we apply intervention-span patching to a combined CF set containing shift items (N=60; meaning-altering) and substitution-invariant items (N=20; label-preserving substitution controls expected not to flip the preferred continuation under these templates). [C10]

Method (summary; see §4.8). For each CF item, base prompt x and intervention prompt x' are tokenized and the minimal contiguous token-level divergence (intervention span) is identified. In the reported runs, we use span_mode=left_aligned_truncated to align donor and receiver spans when they differ in token count by left-aligning and truncating to a shared span length. Hidden states at the aligned intervention span are cached from the x' (donor) run and patched into the x (receiver) run at each transformer block ℓ. The primary metric is the donor-directed preference margin shift; sham patching replaces receiver states with receiver states (no-op). Stratified results by intervention type are reported in the result artifacts. [C10]

Dataset. The patching set comprises 60 shift items (negation / quantifier / role-swap; from the canonical CF suite) and 20 substitution-invariant items (verb-synonym, adjective-synonym, name-substitution, noun-synonym), designed so that the divergence span is non-empty for both base and intervention prompts. These controls are patching invariants (label-preserving under the template), not guaranteed full semantic equivalences; the goal is to test whether patching effects preferentially track meaning-altering interventions relative to plausible same-label substitutions under this protocol. Some substitution controls (notably certain noun substitutions) are not tight lexical synonyms; we treat this as a control-calibration issue and report type-stratified diagnostics below. All 80 items were evaluated with zero skips across all four models. [E10e]

Results (per model: N=60 shift, N=20 substitution-invariant; total N=80; zero skips).
* GPT‑2 (124M): shift 0.701 [0.574, 0.850]; invariant 0.133 [0.031, 0.260]; Δ = 0.568 (d = 1.179); sham ≈ 0. [E10e]
* Qwen2.5‑0.5B: shift 1.478 [1.072, 1.911]; invariant 0.402 [0.123, 0.873]; Δ = 1.076 (d = 0.677); sham ≈ 0. [E10e]
* Qwen2.5‑1.5B: shift 0.491 [0.370, 0.632]; invariant 0.496 [0.224, 0.797]; Δ = −0.005 (d ≈ 0); sham ≈ 0. [E10e]
* Qwen2.5‑3B: shift 1.515 [1.151, 1.929]; invariant 0.356 [0.070, 0.777]; Δ = 1.159 (d = 0.798); sham ≈ 0. [E10e]

Intervention-type stratification (shift items; N = 20 per type) is heterogeneous across models. [E10e]

| Model | Negation mean max effect (CI) | Quantifier mean max effect (CI) | Role-swap mean max effect (CI) |
|---|---|---|---|
| GPT‑2 (124M) | 0.508 [0.452, 0.569] | 1.132 [0.902, 1.368] | 0.463 [0.287, 0.692] |
| Qwen2.5‑0.5B | 0.557 [0.483, 0.630] | 3.538 [2.859, 4.150] | 0.341 [0.197, 0.506] |
| Qwen2.5‑1.5B | 0.806 [0.700, 0.925] | 0.159 [0.106, 0.231] | 0.506 [0.219, 0.881] |
| Qwen2.5‑3B | 0.753 [0.663, 0.839] | 0.338 [0.278, 0.397] | 3.453 [2.903, 4.034] |

Interpretation. Three models show the expected relevance-controlled pattern: intervention-span patching produces substantially larger donor-directed effects on meaning-altering shift items than on label-preserving substitution controls (d = 0.68–1.18), with sham baselines near zero. Qwen2.5‑1.5B is a genuine negative: shift and invariant effects are statistically indistinguishable (d ≈ 0). This qualifies any blanket generalization claim for CF patching: relevance control holds in 3/4 tested checkpoints under this protocol, not 4/4. [E10e, C10]

Interpretive scope of the 1.5B null. The Qwen2.5-1.5B shift≈invariant result is treated as a model-specific protocol failure, not as a direct falsification of CPT. CPT predicts donor-directed causal leverage under meaning-relevant interventions, but does not entail uniform shift>invariant separation at every checkpoint under one intervention design. This null therefore motivates heterogeneity hypotheses (e.g., training-mixture effects, lexical calibration, or span-alignment sensitivity) for targeted follow-up tests.

Type note (diagnostic). The 1.5B null is driven by two factors visible in the intervention-type strata: unusually weak shift effects for quantifier items (0.159 vs 1.132 in GPT‑2) and elevated invariant effects for synonym substitutions. We treat this as a model-specific constraint on CF relevance control under patching and report intervention-type strata in the artifacts for follow-up. [E10e]

Sensitivity: excluding noun-substitution controls (criterion-based). Three noun-substitution items produce invariant effects 4–22× larger than the remaining 17 controls across all Qwen models, indicating poor lexical calibration (the "synonyms" are not close distributional equivalents). Excluding these 3 items and holding the pooled SD constant yields adjusted d values: GPT‑2 1.25 (+0.08); Qwen2.5‑0.5B 0.81 (+0.13); Qwen2.5‑1.5B 0.28 (+0.29); Qwen2.5‑3B 0.99 (+0.19). The 1.5B null partially lifts (d: −0.01 → 0.28) but remains weak, confirming that the 1.5B result is not solely an artifact of poorly calibrated noun controls. This exclusion is criterion-based (lexical calibration), not outcome-based. [E10e]

5.8.3. Summary: causal evidence across AoM components
Table 4 summarizes causal patching evidence across AoM components. DISAMB, CF (with substitution-invariant controls), and COH patching are reported for all four models. [E3, E4a, E10e, E10c, E10d, C9, C10]

[Table 4 about here]


6. Discussion: from metrics to mechanism to philosophy
6.1. What the results establish (and what they do not)
Within the limits of templated datasets and a small number of seeds, the results establish an AoM competence profile under controlled variation:
* context-sensitive disambiguation and coherence behavior. [E2a, E1]
* stronger response to meaning-altering interventions than to shams. [E8, E7]
* causal leverage of token-in-context states via patching. [E3, E4a, C3]
* constraint-directed coherence degradation under COH pseudo-ablation patching (constraint vs irrelevant spans, relevance-controlled). [E10c, E10d, C9]
* donor-directed causal effects on counterfactual sensitivity under intervention-span patching, with shift > invariant separation in 3/4 tested checkpoints (d = 0.68–1.18) and a genuine null for Qwen2.5‑1.5B (d ≈ 0). [E10e, C10]
The results do not settle reference, grounding, normativity, or consciousness.
6.2. From behavior to mechanism: what patching adds
Behavioral AoM metrics alone underdetermine mechanism. CPT patching adds an intervention:
1. Define a meaning-relevant behavioral readout on minimal pairs: relative preference between sense-diagnostic labels.
2. Intervene on internal variables: (h_{M,\ell,p}(x)) at the ambiguous target span.
3. Measure donor-directed change via (\operatorname{effect}_\ell) (margin shift; §4.6).
4. Compare to controls (sham patching; alignment checks).
This yields causal evidence that token-in-context states at specific depths control disambiguation behavior. [E3, E4a, C3]
COH pseudo-ablation patching extends this intervention logic to discourse constraints: across the tested model families (GPT‑2 and Qwen2.5‑{0.5B, 1.5B, 3B}), injecting pseudo-ablated states at constraint-relevant spans degrades coherence substantially more than patching length-matched irrelevant spans (d = 1.27–1.53), supporting a relevance-controlled causal effect on coherence under this protocol. [E10c, E10d, C9]
Spatial structure (SDH constraint). The SDH stress test adds a second constraint: at fixed depth (\ell^*) and within a local position window (position_window=8, buffers ±2), off-target control patches often produce donor-directed effects comparable to target patches. This rules out a general claim that causal control at (\ell^*) is uniquely anchored to the target span. The natural mechanistic reading is local relational distribution within the tested neighborhood, not isolated token storage. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
6.3. Anticipating hostile reviewers: strongest objections and responses
Objection 1: “AoM is only surface.”
AoM is an empirical target. It is not a metaphysical thesis. If meaning requires more than AoM, the burden is to specify competence-relevant necessity and detectable signatures (§6.6).
Objection 2: "Templates are too easy; keyword baseline is strong."
The keyword baseline attains 0.615, confirming that surface cues are present in these templates. However, stratified analysis (§5.2) shows that models maintain accuracy well above 0.75 even on the 40/52 pairs where the keyword heuristic fails—with Qwen models dropping by at most 0.023. DISAMB accuracy is not reducible to cue exploitation. [E2a, E1, E9, C5]
Objection 3: “Patching could be artifact.”
Sham patching is near zero and layerwise profiles are structured. Stronger controls (random-span patching, outside-window patching, activation norm diagnostics) remain desirable and are listed as future work. [E3, E4a, C3]
Objection 4: “Tokenization confounds.”
Span alignment checks skip misaligned directions. The paper reports skipped counts. This remains a validity condition. [E3, E4a, C3]
Objection 5: “COH ablations reflect length.”
Ablate-irrelevant controls for length reduction. Ablate-relevant degrades more. That is the intended diagnostic asymmetry. [E7]
6.4. Relation to Kaplan, Braun, Wittgenstein, and Kant (without overreach)
Terminological clarification (scope of "context"). Throughout this paper, "context" in the operational sections denotes linguistic co-text: the surrounding token sequence and discourse material available to the model at inference time. We do not claim access to Kaplanian extra-linguistic context parameters (e.g., speaker, time, place). Kaplan's framework is used here as a comparative stability ideal for implementation-level reasoning, not as the literal variable set manipulated by our interventions.
Kaplan (1989) and Braun (1995): stability ideals, relational character, and the empirical pressure from SDH. Kaplan’s character/content framework assigns to each expression a fixed rule (character) that maps contexts to contents, treating context as a parameter that selects among content values; this is a theory of meaning rather than an implementation claim, and nothing in Kaplan entails a particular computational architecture. Although our DISAMB items concern lexical ambiguity rather than Kaplanian indexicals, Kaplan’s decomposition supplies a useful template for contrasting a stability‑ideal implementation—expression‑linked rules with context as a separable parameter—against relational implementation pictures. The point is not that Kaplan’s framework directly predicts lexical-disambiguation circuitry, but that it offers a generic factorization—stable type-level rule plus context parameterization—against which token-privileged mechanistic pictures can be contrasted. The contrast here is not Kaplan vs Braun as semantic theories, but token‑privileged vs relational implementation glosses that one might overlay on Kaplan-style factorization. Nothing here assumes that a stability ideal must be implemented token-locally; the empirical pressure from SDH is directed at token‑privileged or token‑atomic mechanistic glosses, not at Kaplan’s semantic framework itself. A natural token‑privileged gloss of the stability ideal is that, if an expression’s contribution is implemented as something like an expression‑linked controller, the ambiguous token position should be a privileged causal site for disambiguation, with surrounding context modulating that site rather than participating symmetrically in causal control. Under this gloss, the SDH stress test (§5.7) would predict robust positive target–control deltas: patching the ambiguous span should yield systematically stronger donor‑directed effects than patching nearby off‑target positions, because causal control would be comparatively anchored to the expression’s own site. Braun (1995), by contrast, challenges treating character as an intrinsic, purely type-level property independent of broader syntactic and contextual structure, and emphasizes structured relations among an expression, its co‑text, and the discourse situation. On a relational gloss, the mechanistic prediction differs: disambiguation‑relevant causal control should often be distributed across positions that participate in the relational structure, not uniquely localized at the ambiguous token.

The SDH results bear on this contrast. At fixed depth and within the tested local neighborhood (position_window=8, exclusion buffers ±2), seven of eight models show target–control deltas near zero or negative; only Qwen2.5‑3B exhibits a robust positive delta (Table 3). The stratified analysis (Table 3b) sharpens the point: measured specificity depends strongly on control selection strategy—some strategies yield systematically negative deltas, while same_index_relaxed yields positive deltas across most models. This pattern exerts pressure on token‑privileged glosses at the tested depth and within the tested window: it suggests that whatever controls disambiguation there is not uniquely anchored to the target span. A relational reading predicts exactly this possibility, because positions other than the ambiguous token can participate in the integration that controls disambiguation behavior. At the same time, the strategy sensitivity also signals a methodological warning: “neutral” controls are nontrivial in templated contexts, and some strategies may systematically select informationally salient positions (a hypothesis that can be directly tested by analyzing which tokens/roles are chosen under each strategy).

Two caveats constrain the inference. First, the SDH test operates at a single fixed depth (depth_frac=0.25), not at each model’s empirically best layer from a full sweep; specificity may differ at other depths. Second, the result establishes local non‑exclusivity within the tested window, not global diffusion; positions outside the tested neighborhood are not measured in this submission (§6.5, F4). Within these bounds, the empirical pattern aligns more naturally with Braun‑style relational emphases than with token‑atomic locality: at the tested depth, disambiguation‑relevant causal control is often not uniquely anchored to the ambiguous token position, but appears distributed across a local neighborhood in a way that is sensitive to the relational structure of the context. This does not refute Kaplan’s semantic framework; it constrains token‑atomic mechanistic glosses on stability ideals and motivates more explicit modeling of how context is integrated and routed in transformer computation.
Wittgenstein (1953) and Cavell (1979): meaning as use, and why context is constitutive rather than auxiliary.
On a Wittgensteinian view, “meaning is use” (PI §43): the contribution of an expression is fixed by the role it plays in rule-governed linguistic activity, not by a context-independent semantic atom carried by the word-type. In that sense, context is not merely an external input that selects among pre-existing meanings; it is part of what constitutes the use that makes an utterance intelligible in the first place (a point Cavell develops by stressing that understanding is embedded in forms of life and criteria for correct application). Read with this discipline, AoM is explicitly a *use-profile* rather than a metaphysical semantics: DISAMB and CF probe whether a model’s preferences track controlled changes in use-conditions (surrounding cues; meaning-altering vs meaning-preserving edits), and COH probes whether the model maintains use-constraints across discourse rather than responding only to length or recency. The patching results then provide a mechanistic analogue of the Wittgensteinian moral: the proximate controllers of these use-sensitivities are not static word-types but contextualized token-in-context states, and the SDH stress test further suggests that—at least at the tested depth and within the tested neighborhood—causal control is often distributed across relationally relevant positions rather than uniquely localized at the ambiguous token. This does **not** establish the specifically Wittgensteinian normative dimension (public criteria, correction, rule-following as a social practice): our operationalization remains text-internal and non-interactive. The narrower claim is that, within the model, the *appearance* of meaning-like competence is implemented through context-dependent integration that is empirically isolable by controlled interventions on internal state.
Kant: phenomena and methodological modesty
AoM/CPT/SDH is a phenomena-first program: measure competence signatures and causal controllers without claiming to settle meaning “in itself.”
6.5. Falsifiers and limits
CPT is falsifiable in this framework (F1–F2). SDH is falsifiable as a local distribution claim once outside-window controls are implemented (F4).
Current limits:
* prototype operationalization with small, templated suites: DISAMB (52 pairs) and the current CF/COH sets are intentionally controlled for identifiability in a first-pass causal test. This strengthens internal validity but limits claims about naturalistic generalization; larger and less templated corpora are required in the next phase.
* DISAMB templates contain surface cues (keyword baseline 0.615), though keyword-failure stratification shows accuracy remains high on the 40/52 pairs where the heuristic fails; further adversarial hardening would strengthen generality. [E2a, E1, C5, E11b]
* CF causal patching now includes substitution-invariant controls (N=20) and is relevance-controlled by design; the expected shift > invariant separation holds for 3/4 tested checkpoints but fails for Qwen2.5‑1.5B (d ≈ 0). A criterion-based sensitivity analysis excluding poorly calibrated noun-synonym controls (§5.8.2) partially lifts the 1.5B null (d → 0.28) but does not resolve it. The invariant set is small; strengthening and broadening invariant controls is a concrete next step. [C10, E10e]
* limited seeds in reported AoM+CPT sweeps; [E2a, E1, E3, E4a]
6.6. Burden shift and the context trilemma
The AoM program is intended to change what must be argued, not to settle metaphysics. The results in this paper provide (i) behavioral evidence for a consistent (suite-conditional) pattern of context-sensitivity across DISAMB/CF/COH under controlled manipulations, and (ii) causal evidence that internal token-in-context states at specific depths exert donor-directed control over AoM-relevant preference margins under activation patching, with sham baselines near zero and structured depth profiles. SDH adds a further mechanistic constraint: at a fixed tested depth and within a local neighborhood, causal control is often not uniquely localized to the ambiguous token position. These findings do not establish meaning proper, but they do make purely dismissive “mere surface” moves less informative: to maintain a meaning-beyond-AoM position while conceding AoM-like competence, one must identify an additional property that is (1) competence-relevant, (2) plausibly absent here, and (3) associated with detectable signatures that could in principle be measured.
However, the force of this burden-shift is **conditional** on closing the remaining validation gaps in the present experimental regime. In this revision: DISAMB templates contain surface cues (keyword baseline 0.615), though stratified analysis shows models retain high accuracy on the keyword-failure subset; COH patching now replicates across all four models with large constraint–irrelevant asymmetries (d > 1.2); and CF patching is now relevance-controlled by design (shift vs substitution-invariant), with the expected separation holding for 3/4 checkpoints but failing for Qwen2.5‑1.5B. As remaining gaps are closed—by broadening and hardening the invariant control set, extending COH patching with fluent natural-language distractor replacements (in addition to padding), and adding outside-window SDH controls—the AoM evidence would increasingly constrain meaning-beyond-AoM proposals to be explicit about what more is required and how it would show up empirically.
The present results remain compatible with multiple interpretations: (A) context-dependent semantics realized by distributed relational states; (B) robust competence without traditional semantics; (C) compressed, practice-derived competence without normativity. This paper does not adjudicate among (A)–(C). It aims to supply a shared empirical substrate—operational competence signatures plus causal-control and spatial-structure constraints—on which the five positions surveyed by Søgaard (2025) can make more discriminating commitments.

7. Conclusion
This paper defines appearance of meaning (AoM) as an operational competence profile under controlled context variation and states CPT as a causal-control hypothesis for transformer architectures.
GPT‑2 and Qwen2.5 models achieve AoM composites from 0.8489 to 0.9026 on these suites. [E2a, E1]
CPT-style activation patching yields donor-directed effects with sham baselines near zero and mid-depth peaks (≈29–50% depth across the evaluated checkpoints, using 0-indexed layers and depth_frac=layer/(L-1)). [E3, E4a]
A fixed-layer target-specificity stress test across eight models evaluates SDH. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
Specificity is heterogeneous: in most models tested, donor-directed control at (\ell^*) is not uniquely localized to the target span within the tested local window (position_window=8, buffers ±2). [E5a, E5b, E5c, E5d, E5e, E5f]
Only Qwen2.5‑3B shows a robust positive specificity delta under this protocol. [E5a, E5b, E5c, E5d, E5e, E5f]
Control selection strategy materially affects measured specificity and must be reported. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
This submission is therefore a JoLLLI-oriented theory-plus-protocol paper: it establishes an operational target and causal test framework under tightly controlled templates, while leaving exhaustive mechanistic decomposition and broad naturalistic stress-testing to follow-on work.
Several concrete extensions follow from documented limitations, listed in priority order.
First priority: characterizing and hardening CF relevance control under patching. A shift vs substitution-invariant comparison is now implemented (N=60 shift, N=20 invariant; zero skips); the expected separation holds for three checkpoints (d = 0.68–1.18) but fails for Qwen2.5‑1.5B (d ≈ 0). Next steps include expanding the invariant set beyond synonym substitutions, reporting type-stratified separations, and testing whether the 1.5B null persists under alternative shift subsets or alternative invariant constructions.
Second: replacing padding-based COH pseudo-ablations with fluent natural-language distractor replacements to test robustness to the distribution-shift concern. (COH pseudo-ablation patching now covers all four models with large constraint–irrelevant asymmetries; the padding caveat remains the primary open robustness question for this component.)
Third: hardening the DISAMB suite against cue leakage and adding outside-window controls for SDH would strengthen the most protocol-conditional components.

References
Belinkov, Y., & Glass, J. (2019). Analysis Methods in Neural Language Processing: A Survey. TACL / arXiv versions.
Bender, E. M., & Koller, A. (2020). Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data. ACL.
Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? FAccT.
Braun, D. (1995). What is Character? Journal of Philosophical Logic, 24(3), 227–240.
Cavell, S. (1979). The Claim of Reason: Wittgenstein, Skepticism, Morality, and Tragedy. Oxford University Press.
Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? arXiv:1909.00512.
Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. Technical report / arXiv versions.
Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal Abstractions of Neural Networks. NeurIPS 2021. https://openreview.net/forum?id=RmuXDtjDhG
Groenendijk, J., & Stokhof, M. (1991). Dynamic Predicate Logic. Linguistics and Philosophy.
Heim, I., & Kratzer, A. (1998). Semantics in Generative Grammar. Blackwell.
Jannai, D., Meron, A., Lenz, B., Levine, Y., & Shoham, Y. (2023). Human or Not? A Gamified Approach to the Turing Test. arXiv:2305.20010.
Kaplan, D. (1989). Demonstratives. In Almog, Perry, & Wettstein (Eds.), Themes from Kaplan (pp. 481–563). Oxford University Press.
Kant, I. (1781/1787). Critique of Pure Reason (A/B pagination).
Katz, D. M., Bommarito, M. J., Gao, S., & Arredondo, P. (2024). GPT‑4 Passes the Bar Exam. Philosophical Transactions of the Royal Society A, 382, 20230254.
Lappin, S. (2024). Assessing the Strengths and Weaknesses of Large Language Models. Journal of Logic, Language and Information, 33, 9–20. https://doi.org/10.1007/s10849-023-09409-x
Li, B. Z., Nye, M., & Andreas, J. (2021). Implicit Representations of Meaning in Neural Language Models. ACL-IJCNLP 2021.
Lindsey, J., et al. (2025). On the Biology of a Large Language Model. Transformer Circuits. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
Mahowald, K., Ivanova, A. A., Blank, I. A., Kanwisher, N., Tenenbaum, J. B., & Fedorenko, E. (2024). Dissociating language and thought in large language models. Trends in Cognitive Sciences, 28(6), 517–540.
Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. (ROME; arXiv / workshop versions).
Milliere, R., & Buckner, C. (2024a). A Philosophical Introduction to Language Models—Part I: Continuity With Classic Debates. arXiv:2401.03910.
Milliere, R., & Buckner, C. (2024b). A Philosophical Introduction to Language Models—Part II: The Way Forward. arXiv:2405.03207.
Mollo, D. C., & Milliere, R. (2023). The Vector Grounding Problem. arXiv:2304.01481. (Forthcoming in Philosophy and the Mind Sciences.)
Piantadosi, S. T., & Hill, F. (2022). Meaning without reference in large language models. arXiv / preprint versions.
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI (GPT‑2 technical report).
Rogers, A., Kovaleva, O., & Rumshisky, A. (2020). A Primer in BERTology: What We Know About How BERT Works. TACL.
Søgaard, A. (2025). Do Language Models Have Semantics? On the Five Positions. ACL 2025. https://aclanthology.org/2025.acl-long.1258.pdf
Turing, A. M. (1950). Computing Machinery and Intelligence. Mind, 59(236), 433–460.
Veltman, F. (1996). Defaults in Update Semantics. Journal of Philosophical Logic.
Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020). Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS 2020.
Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. ICLR 2023. arXiv:2211.00593.
Wittgenstein, L. (1953). Philosophical Investigations. Blackwell.
Model cards / checkpoints (provenance, parameterization, licensing) via Hugging Face:
* https://huggingface.co/openai-community/gpt2
* https://huggingface.co/Qwen/Qwen2.5-0.5B
* https://huggingface.co/Qwen/Qwen2.5-1.5B
* https://huggingface.co/Qwen/Qwen2.5-3B
* https://huggingface.co/Qwen/Qwen3-4B
* https://huggingface.co/meta-llama/Llama-3.2-1B
* https://huggingface.co/meta-llama/Llama-3.2-3B
* https://huggingface.co/meta-llama/Meta-Llama-3.1-8B

Tables and Figures
Table 1. Behavioral AoM metrics across models
Table 1. Behavioral AoM metrics for GPT‑2 and Qwen2.5 models on the hardened `paper_hardened_v2` suites. DISAMB accuracy is computed over minimal pairs (52 pairs, pair-bootstrap). CF reports shift-direction accuracy on shift items only (N=60 in the canonical `aom_eval.csv` artifact); invariant (N=60) and graded (N=20) items are reported via split metrics and perturbation-magnitude fields. COH reports constraint accuracy in the main condition (80 items); matched ablation controls are length-matched variants across three conditions (240 contexts total). Brackets denote 95% bootstrap confidence intervals (1000 replicates; seed=42). [E2a, E1, C6, C7]
Model	AoM composite	DISAMB acc (CI)	CF shift-dir acc (CI)	COH main acc (CI)	COH ablate-irrelevant (CI)	COH ablate-relevant (CI)
openai-community/gpt2 (124M)	0.869	0.856 [0.788, 0.914]	0.850 [0.750, 0.933]	0.900 [0.838, 0.963]	0.888 [0.812, 0.950]	0.588 [0.487, 0.700]
Qwen/Qwen2.5-0.5B	0.849	0.913 [0.856, 0.962]	0.733 [0.617, 0.833]	0.900 [0.838, 0.963]	0.888 [0.825, 0.950]	0.362 [0.263, 0.475]
Qwen/Qwen2.5-1.5B	0.869	0.933 [0.885, 0.971]	0.800 [0.700, 0.883]	0.875 [0.800, 0.938]	0.850 [0.775, 0.925]	0.350 [0.263, 0.450]
Qwen/Qwen2.5-3B	0.903	0.933 [0.885, 0.981]	1.000 [1.000, 1.000]	0.775 [0.688, 0.863]	0.825 [0.750, 0.900]	0.375 [0.275, 0.487]
Keyword baseline (DISAMB suite): 0.615 [0.558, 0.673] (pair-bootstrap CI aligned to DISAMB bootstrap unit). [E2a, E1, C5]

Table 2. CPT-style activation patching aggregates
Table 2. CPT-style context-swap activation patching aggregates on the disambiguation suite (GPT‑2 and Qwen2.5). “Mean max effect” averages (over directions) (\max_\ell \operatorname{effect}_\ell) as defined in §4.6. Flip rate reports donor-directed flips at the empirically best layer. Sham patching replaces with the receiver’s own states (no-op). Block counts (L) are inferred from the per-layer effect columns in the cited artifacts (L = 1 + max{i : cpt_effect_layer_i is finite}). [E3, E4a, C3]
Model	Patched / total (skipped)	Mean max effect (CI)	Flip rate @ best layer (CI)	Mean argmax layer	Sham mean max effect (CI)
GPT‑2 (124M; 12 blocks)	104/104 (0)	0.395 [0.301, 0.501]	0.000 [0.000, 0.000]	5.47	2.43×10⁻⁶ [1.23×10⁻⁶, 3.76×10⁻⁶]
Qwen2.5‑0.5B (24 blocks)	104/104 (0)	1.651 [1.241, 2.128]	0.058 [0.019, 0.106]	8.19	7.59×10⁻⁶ [6.13×10⁻⁶, 9.09×10⁻⁶]
Qwen2.5‑1.5B (28 blocks)	104/104 (0)	1.807 [1.461, 2.230]	0.115 [0.058, 0.183]	7.72	5.67×10⁻⁶ [3.69×10⁻⁶, 7.87×10⁻⁶]
Qwen2.5‑3B (36 blocks)	104/104 (0)	1.686 [1.304, 2.117]	0.048 [0.010, 0.087]	13.53	1.61×10⁻⁵ [1.42×10⁻⁵, 1.79×10⁻⁵]
Table 3. SDH target-specificity stress test (fixed layer)
Table 3. Target-specificity results at fixed layer (\ell^*=\operatorname{round}(0.25\times(L-1))) with position_window=8 and exclusion buffers ±2. (E_{\text{target}}) and (E_{\text{ctrl}}) are donor-directed effects (\operatorname{effect}{\ell^*}) (margin shifts; §4.6). (\Delta = E{\text{target}}-E_{\text{ctrl}}). Win rate is (\Pr[\Delta_i>0]). [E5a, E5b, E5c, E5d, E5e, E5f, C4]
Model	Layers	ℓ*	E_target (CI)	E_ctrl (CI)	Δ (CI)	Win rate (CI)
GPT‑2 (124M)	12	3	0.44 [0.28, 0.61]	0.95 [0.64, 1.27]	-0.51 [-0.85, -0.20]	0.41 [0.32, 0.50]
Qwen2.5‑0.5B	24	6	1.25 [0.98, 1.55]	1.37 [0.95, 1.87]	-0.13 [-0.65, 0.36]	0.52 [0.42, 0.62]
Qwen2.5‑1.5B	28	7	1.39 [1.05, 1.75]	1.55 [1.02, 2.21]	-0.16 [-0.88, 0.42]	0.52 [0.42, 0.62]
Qwen2.5‑3B	36	9	2.15 [1.62, 2.77]	1.27 [0.79, 1.79]	+0.87 [0.13, 1.66]	0.59 [0.49, 0.68]
Qwen3‑4B	36	9	1.59 [1.09, 2.12]	1.76 [1.06, 2.56]	-0.17 [-1.04, 0.65]	0.52 [0.42, 0.62]
Llama‑3.2‑1B	16	4	1.07 [0.73, 1.38]	1.23 [0.82, 1.68]	-0.16 [-0.70, 0.33]	0.50 [0.40, 0.59]
Llama‑3.2‑3B	28	7	1.10 [0.78, 1.48]	1.27 [0.80, 1.75]	-0.17 [-0.79, 0.43]	0.52 [0.42, 0.61]
Llama‑3.1‑8B	32	8	0.80 [0.57, 1.07]	1.33 [0.84, 1.89]	-0.52 [-1.13, 0.02]	0.50 [0.40, 0.59]
Table 3b. SDH specificity deltas by control selection strategy
Table 3b. (\Delta) stratified by control selection strategy. Each entry reports mean (\Delta) with 95% CI and N of directions for that strategy. [E5a, E5b, E5c, E5d, E5e, E5f, C4]
Model	matched_token Δ	same_index Δ	same_index_relaxed Δ
GPT‑2 (124M)	-0.74 [-1.17, -0.21] (N=30)	-0.70 [-1.62, 0.18] (N=30)	-0.20 [-0.51, 0.07] (N=40)
Qwen2.5‑0.5B	-0.49 [-0.98, 0.00] (N=30)	-1.02 [-2.64, 0.45] (N=30)	+0.82 [0.37, 1.23] (N=40)
Qwen2.5‑1.5B	-0.30 [-0.84, 0.27] (N=30)	-1.31 [-3.30, 0.50] (N=30)	+0.80 [0.17, 1.48] (N=40)
Qwen2.5‑3B	+0.41 [-0.56, 1.33] (N=30)	-0.17 [-2.06, 1.71] (N=30)	+2.00 [1.06, 3.00] (N=40)
Qwen3‑4B	+0.59 [-0.41, 1.75] (N=30)	-2.20 [-4.77, -0.01] (N=30)	+0.77 [0.02, 1.45] (N=40)
Llama‑3.2‑1B	-0.17 [-0.83, 0.51] (N=30)	-1.24 [-2.75, 0.20] (N=30)	+0.65 [0.21, 1.10] (N=40)
Llama‑3.2‑3B	-0.26 [-0.87, 0.38] (N=30)	-1.63 [-3.30, -0.06] (N=30)	+0.99 [0.46, 1.62] (N=40)
Llama‑3.1‑8B	-0.36 [-0.99, 0.17] (N=30)	-1.99 [-3.75, -0.36] (N=30)	+0.45 [0.12, 0.84] (N=40)

Table 4. Causal patching across AoM components
Table 4. DISAMB context-swap patching rows are reproduced from Table 2 for comparison. COH rows report pseudo-ablation patching on the hardened coherence suite across all four models; for COH, Mean Max Effect is the overall mean across constraint- and irrelevant-span patching conditions, and Cohen's d is for the constraint–irrelevant asymmetry. CF rows report intervention-span patching on shift items (N=60) versus substitution-invariant controls (N=20) under `span_mode=left_aligned_truncated`; for CF, Mean Max Effect reports the shift stratum mean, and Cohen's d is the shift–invariant separation computed over per-case max-over-layers effects. Sham baselines are near zero at the shown precision. Brackets denote 95% bootstrap confidence intervals. [E3, E4a, E10e, E10c, E10d]
| Experiment | Model | Mean Max Effect (CI) | Sham Baseline | Key Contrast | Cohen's d | N Used / Total | Control Status |
|---|---|---|---|---|---|---|---|
| DISAMB context-swap | GPT‑2 | 0.395 [0.301, 0.501] | 2.43×10⁻⁶ | effect vs sham | — | 104/104 | Sham-controlled |
| DISAMB context-swap | Qwen2.5‑0.5B | 1.651 [1.241, 2.128] | 7.59×10⁻⁶ | effect vs sham | — | 104/104 | Sham-controlled |
| DISAMB context-swap | Qwen2.5‑1.5B | 1.807 [1.461, 2.230] | 5.67×10⁻⁶ | effect vs sham | — | 104/104 | Sham-controlled |
| DISAMB context-swap | Qwen2.5‑3B | 1.686 [1.304, 2.117] | 1.61×10⁻⁵ | effect vs sham | — | 104/104 | Sham-controlled |
| COH pseudo-ablation | GPT‑2 | 0.753 [0.651, 0.858] | 0.000 | constraint 1.094 vs irrelevant 0.411 (Δ = 0.683) | 1.266 | 160/160 | Sham + relevance-controlled |
| COH pseudo-ablation | Qwen2.5‑0.5B | 1.010 [0.837, 1.193] | 0.009 | constraint 1.687 vs irrelevant 0.332 (Δ = 1.355) | 1.526 | 160/160 | Sham + relevance-controlled |
| COH pseudo-ablation | Qwen2.5‑1.5B | 1.150 [0.964, 1.355] | 0.019 | constraint 1.838 vs irrelevant 0.463 (Δ = 1.375) | 1.368 | 160/160 | Sham + relevance-controlled |
| COH pseudo-ablation | Qwen2.5‑3B | 0.869 [0.725, 1.025] | 0.010 | constraint 1.403 vs irrelevant 0.336 (Δ = 1.067) | 1.377 | 160/160 | Sham + relevance-controlled |
| CF intervention-span | GPT‑2 | 0.701 [0.574, 0.850] | 0.000 | shift 0.701 vs invariant 0.133 (Δ = 0.568) | 1.179 | 80/80 (60+20) | Sham + relevance-controlled |
| CF intervention-span | Qwen2.5‑0.5B | 1.478 [1.072, 1.911] | 0.000 | shift 1.478 vs invariant 0.402 (Δ = 1.076) | 0.677 | 80/80 (60+20) | Sham + relevance-controlled |
| CF intervention-span | Qwen2.5‑1.5B | 0.491 [0.370, 0.632] | 0.000 | shift 0.491 vs invariant 0.496 (Δ = −0.005) | −0.009 | 80/80 (60+20) | Sham + relevance-controlled |
| CF intervention-span | Qwen2.5‑3B | 1.515 [1.151, 1.929] | 0.000 | shift 1.515 vs invariant 0.356 (Δ = 1.159) | 0.798 | 80/80 (60+20) | Sham + relevance-controlled |
