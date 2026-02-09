The Appearance of Meaning: Context-Dependence and Semantic Competence in Transformer Architectures
Author: [TODO: AUTHOR NAME / AFFILIATION / EMAIL]Acknowledgements / Funding: [TODO]
ESSLLI note: This manuscript develops and extends an ESSLLI 2025 Student Session presentation. The operational AoM evaluation framework, the CPT-style activation-patching protocol, and the cross-family empirical results reported here (GPT‑2 and Qwen2.5) are new to this submission (i.e., not previously published).
Keywords: large language models; transformers; context dependence; semantic competence; appearance of meaning; causal interpretability; activation patching; Kaplan; Braun; Wittgenstein; Kant

Evidence tags (for auditability): bracketed IDs like [E3], [R0], [C2] refer to rows in `AoM_evidence_contract.md`.

Abstract (≈200 words)
Large language models (LLMs) produce text that human interlocutors routinely treat as meaningful and context-sensitive. This paper does not address whether LLMs possess meaning proper. It isolates an empirical target: appearance of meaning (AoM), operationalized as three measurable competences: (i) context-sensitive disambiguation, (ii) sensitivity to meaning-altering vs. meaning-preserving interventions, and (iii) discourse-level constraint coherence. We then state a mechanistic hypothesis, the Context-Primacy Thesis (CPT): in transformer LMs, causal control of AoM-relevant preference margins is exerted by contextualized token-in-context states at specific depths, in the sense that intervening on those states (with weights fixed) produces donor-directed changes beyond sham baselines.
The framework supports sham CF items, relevance-controlled COH ablations, and CPT sham patching. [C6, C7, C3]
We evaluate GPT‑2 (124M) and Qwen2.5-{0.5B, 1.5B, 3B} using log-probability scoring and report suite metrics with bootstrap confidence intervals. [E2a, E1, C8]
AoM composite scores range from 0.855 (GPT‑2) to 0.914 (Qwen2.5‑3B). [E2a, E1]
Context-swap activation patching on disambiguation minimal pairs yields strong donor-directed effects with sham baselines near zero (mean max effect: 0.667 [0.499, 0.856] for GPT‑2; 1.861 → 2.448 → 3.114 across Qwen2.5). [E3, E4a]
Intervention-span patching on the counterfactual suite provides coverage-limited evidence (shift items only under the reported alignment protocol) that donor-directed causal control extends beyond disambiguation. [E10a, E10b, C10]
Constraint-sentence pseudo-ablation patching on the coherence suite yields a robust constraint–irrelevant degradation asymmetry, providing causal evidence for coherence sensitivity under this intervention regime. [E10c, E10d, C9]
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
4. Evidence: AoM and patching results for GPT‑2 and Qwen2.5, including context-swap patching for DISAMB and pseudo-ablation patching for COH; plus an SDH target-specificity stress test across eight models.
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
Primary CF metric (reported): shift-direction accuracy (computed on shift + invariant items; graded items are excluded unless stated otherwise). A shift item is correct iff (\operatorname{pref}(x_i)\neq \operatorname{pref}(x_i')). An invariant item is correct iff (\operatorname{pref}(x_i)= \operatorname{pref}(x_i')). CF shift-direction accuracy is the fraction correct over shift + invariant items.
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
Probing can show information is decodable, not that it is used (Belinkov & Glass 2019; Rogers et al. 2020). Causal mediation analysis applies counterfactual interventions on network-internal components to distinguish decodable from causally active information (Vig et al. 2020). Interchange interventions formalize this as counterfactual substitution on aligned internal variables across inputs, enabling causal abstraction tests (Geiger et al. 2021). The CPT patching protocol in §4.6 is an interchange intervention specialized to token-in-context hidden states and AoM preference margins; it does not perform a full mediation decomposition. Mechanistic interpretability work develops patching-style interventions and circuit frameworks (Elhage et al. 2021; Wang et al. 2023). The transformer-circuits view treats the residual stream as a shared communication channel, with attention heads iteratively writing context-conditioned information to token representations. Related work also shows that intervening on intermediate representations can redirect downstream generation in predictable ways (Li et al. 2021; Lindsey et al. 2025). Editing approaches intervene on internal associations (Meng et al. 2022). To connect causal effects to the “surface-statistics” objection, we report keyword baselines as a diagnostic. A desideratum for future work is to evaluate whether patching effects concentrate on items where such baselines fail, via stratified analysis.
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
Software + hardware provenance (as logged by the evaluation framework).
* Platform / accelerator: macOS-15.6-arm64-arm-64bit, using Apple mps (Metal Performance Shaders) where applicable (device=mps in logged runs).
* Python: 3.12.7
* PyTorch: 2.5.1
* Transformers: varies by run batch (logged per run). Examples:
    * GPT‑2 AoM + CPT layer sweep (Table 1/2 numbers): transformers==4.46.2 (logged in the GPT‑2 result CSV).
    * SDH specificity runs (Table 3/3b numbers): transformers==4.57.3 (logged in the specificity CSVs).
Dtypes and logging gaps.
* Log-probability computations use float32 (log-softmax and gathered token log-probabilities).
* Model parameter dtype varies by run batch:
    * GPT‑2 AoM runs: torch.float32 (logged).
    * Qwen/Llama specificity runs: commonly torch.bfloat16 or torch.float16 (logged per run in specificity CSVs).
* Tokenizer provenance: tokenizers are loaded from each checkpoint via transformers. The tokenizer library version was not recorded in the result CSVs in this submission.
* Checkpoint hashes / exact Hugging Face revisions: not recorded in the current result artifacts. Default model-repo “main” revisions were used at runtime unless otherwise configured.
* Code commit hash: not recorded here because the evaluation ran outside a git checkout (no .git directory; git_commit logged as empty).
* Legacy CSV caveat: some earlier result files predate full provenance columns and do not include the provenance fields above.
* Sweep launcher logs (for some runs): exact `aom_eval.py` argv (including dataset paths) is recorded in `results/*launcher.log`. [R5]
[R3, R4, R5]
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
The coherence suite contains 40 main items, each with matched ablate-relevant and ablate-irrelevant variants (total 120 contexts). The irrelevant ablation removes an unrelated sentence of comparable length. [E2a, E1]
4.3. Scoring and aggregation
Scoring uses log-softmax probabilities for numerical stability. Continuations are scored by mean per-token log-probability when length normalization is enabled (no_length_norm=False where recorded in the result artifacts). [R3, R4]
The evaluator uses:
* deterministic evaluation (model.eval()),
* log-prob computations in float32,
* strict finiteness checking.
Strict finiteness checks. In the logged runs reported here, strict_finite=True (fail on NaN/Inf). [R3]
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

CF intervention-span patching. For each CF item, the base prompt x and intervention prompt x' are tokenized, and the minimal contiguous token-level divergence (intervention span) is identified. In the reported runs, span alignment uses span_mode=left_aligned_truncated: when donor and receiver intervention spans differ in token count, we left-align and truncate to a shared span length. Hidden states at the aligned intervention span are cached from the x' (donor) run and patched into the x (receiver) run at each transformer block ℓ. The effect metric is the donor-directed preference margin shift (analogous to §4.6). Sham patching replaces receiver states with receiver states. Items with empty receiver (base) spans are skipped (empty_span); graded items are excluded by configuration. Skip counts are reported. [C10]

Both protocols use the same scoring, bootstrap, and finiteness-checking infrastructure as the DISAMB patching described in §4.3–4.6. [C9, C10]

5. Results
Results are reported for:
* AoM + CPT sweep: GPT‑2 and Qwen2.5-{0.5B, 1.5B, 3B} (Tables 1–2). [E2a, E1, E3, E4a]
* SDH specificity stress test (fixed layer): eight models total (Table 3, Table 3b). [E5a, E5b, E5c, E5d, E5e, E5f]
* CF intervention-span patching (coverage-limited; §5.8) for GPT‑2 and Qwen2.5-{0.5B, 1.5B, 3B} (Table 4). [E10a, E10b, C10]
* COH pseudo-ablation patching (§5.8) for GPT‑2 and Qwen2.5-{0.5B, 1.5B, 3B} (Table 4). [E10c, E10d, C9]
Unless stated otherwise, confidence intervals are 95% bootstrap CIs.
5.1. Behavioral AoM metrics
AoM composite scores:
* GPT‑2 (124M): 0.855
* Qwen2.5‑0.5B: 0.857
* Qwen2.5‑1.5B: 0.848
* Qwen2.5‑3B: 0.914
[E2a, E1]
[Table 1 about here] [E2a, E1]
5.2. AoM-DISAMB: high accuracy with a baseline caveat
DISAMB accuracies:
* GPT‑2 (124M): 0.865 [0.808, 0.923] (N = 52 pairs; 104 sides)
* Qwen2.5‑0.5B: 0.913 [0.865, 0.962]
* Qwen2.5‑1.5B: 0.962 [0.923, 0.990]
* Qwen2.5‑3B: 0.942 [0.894, 0.981]
[E2a, E1]
A keyword baseline attains 0.913 with pair-bootstrap CIs aligned to the DISAMB bootstrap unit (minimal pairs). [E2a, E1, C5]
DISAMB accuracy alone is therefore not the main philosophical load-bearer in this submission. CPT patching and CF/COH controls carry the weight.
Mean DISAMB margins increase with capacity within Qwen2.5 (reported in the CSV artifacts), consistent with increased separation between senses under these templates. [E1]
5.3. AoM-CF: controlled sensitivity with sham separation
CF shift-direction accuracy:
* GPT‑2: 0.850 [0.750, 0.933] (N = 120)
* Qwen2.5‑0.5B: 0.733 [0.617, 0.833]
* Qwen2.5‑1.5B: 0.783 [0.683, 0.883]
* Qwen2.5‑3B: 1.000 [1.000, 1.000]
[E2a, E1]
Sham separation is visible in mean absolute preference changes (|\Delta(x')-\Delta(x)|): invariant items shift less than shift items across models, including GPT‑2. [E8]
5.4. AoM-COH: coherence tracks informational role, not only length
Main-group coherence accuracies:
* GPT‑2: 0.850 [0.725, 0.950]
* Qwen2.5‑0.5B: 0.925 [0.825, 1.000]
* Qwen2.5‑1.5B: 0.800 [0.675, 0.925]
* Qwen2.5‑3B: 0.800 [0.675, 0.925]
[E2a, E1]
Ablations show an informational-role pattern: ablate-relevant degrades more than length-matched ablate-irrelevant across all four models, with varying magnitude. [E7]
5.5. CPT-style activation patching: causal leverage for token-in-context control
Activation patching is run on the disambiguation suite for GPT‑2 and Qwen2.5 models reported in Table 2. Each has 104 patching directions total, with 100 patched and 4 skipped due to span misalignment. [E3, E4a, C3]
Primary CPT aggregates:
* Mean max effect (defined in §4.6), increasing with capacity across the evaluated checkpoints:
    * GPT‑2: 0.667 [0.499, 0.856]
    * Qwen2.5‑0.5B: 1.861 [1.523, 2.263]
    * Qwen2.5‑1.5B: 2.448 [1.939, 3.026]
    * Qwen2.5‑3B: 3.114 [2.534, 3.800]
* Flip rate at best layer:
    * GPT‑2: 0.050 [0.010, 0.100]
    * Qwen2.5‑0.5B: 0.160 [0.090, 0.230]
    * Qwen2.5‑1.5B: 0.190 [0.110, 0.270]
    * Qwen2.5‑3B: 0.250 [0.180, 0.330]
* Mean argmax layer:
    * GPT‑2: 3.96 / (12-1) (≈36% depth)
    * Qwen2.5‑0.5B: 5.10 / (24-1) (≈22%)
    * Qwen2.5‑1.5B: 6.19 / (28-1) (≈23%)
    * Qwen2.5‑3B: 8.69 / (36-1) (≈25%)
* Sham patching baseline: near zero in all reported runs. [E3, E4a]
These results support CPT as a causal-control claim under this intervention regime. They do not establish exclusivity of the target span as the only causal locus. [E3, E4a]
[Table 2 about here] [E3, E4a]
5.6. Scaling summary (AoM + CPT sweep)
* CPT patching effects are present in GPT‑2 and Qwen2.5 and increase with capacity across the evaluated checkpoints. [E3, E4a]
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

Results (N = 80 constraint-span cases; 80 irrelevant-span cases; total 160).
* GPT‑2 (124M): constraint-span mean max degradation 1.094 [0.970, 1.219]; irrelevant-span 0.411 [0.305, 0.515]; constraint–irrelevant mean diff 0.683 (d=1.266); constraint-span sham baseline 4.53×10⁻⁷ [2.51×10⁻⁷, 6.83×10⁻⁷]. [E10c]
* Qwen2.5‑0.5B: constraint-span 1.687 [1.452, 1.945]; irrelevant-span 0.332 [0.255, 0.428]; mean diff 1.354 (d=1.526); constraint-span sham baseline 0.00887 [0.00509, 0.0130]. [E10d]
* Qwen2.5‑1.5B: constraint-span 1.838 [1.595, 2.133]; irrelevant-span 0.463 [0.354, 0.593]; mean diff 1.374 (d=1.368); constraint-span sham baseline 0.0206 [0.0130, 0.0289]. [E10d]
* Qwen2.5‑3B: constraint-span 1.403 [1.195, 1.627]; irrelevant-span 0.336 [0.276, 0.400]; mean diff 1.067 (d=1.377); constraint-span sham baseline 0.00983 [0.00601, 0.0140]. [E10d]

Mean max effects (averaged over constraint+irrelevant conditions) are 0.753 [0.651, 0.858] (GPT‑2), 1.010 [0.837, 1.193] (Qwen2.5‑0.5B), 1.150 [0.964, 1.355] (Qwen2.5‑1.5B), and 0.869 [0.725, 1.025] (Qwen2.5‑3B). Flip rates at the best layer are moderate (≈0.36–0.44), indicating robust margin shifts that do not always reverse the binary coherence decision. [E10c, E10d]

Interpretation. Across GPT‑2 and the Qwen2.5 family, patching in pseudo-ablated states at constraint-relevant positions yields substantially larger coherence degradation than patching at irrelevant positions, supporting a relevance-controlled causal effect on coherence. Sham baselines are small relative to the constraint-span effects. [E10c, E10d, C9]

Padding caveat. COH pseudo-ablation replaces sentences with padding tokens for positional alignment. This may introduce distributional shift relative to natural text. We interpret the result as evidence about causal sensitivity to removing information at specific spans under this alignment-preserving intervention; natural-language distractor replacements are an important robustness check for future work.

5.8.2. CF causal patching: partial evidence for CPT generalization (coverage-limited)
To test whether CPT-style causal leverage extends to counterfactual sensitivity, we apply intervention-span patching to the CF suite (§4.8). [C10]

Method (summary; see §4.8). For each CF item, base prompt x and intervention prompt x' are tokenized and the minimal contiguous token-level divergence (intervention span) is identified. In the reported runs, we use span_mode=left_aligned_truncated to align donor and receiver spans when they differ in token count by left-aligning and truncating to a shared span length. Hidden states at the aligned intervention span are cached from the x' (donor) run and patched into the x (receiver) run at each transformer block ℓ. The primary metric is the donor-directed preference margin shift; sham patching replaces receiver states with receiver states (no-op). Items with empty receiver (base) spans are skipped (empty_span); graded items are excluded by configuration. Stratified results by intervention type are reported in the result artifacts. [C10]

Coverage constraint. Of 140 CF items (60 shift, 60 invariant, 20 graded), 60 were evaluated (all shift items) and 80 were skipped: 60 invariant items are insertion-type shams that produce empty receiver (base) spans under this span definition (empty_span), and 20 graded items were excluded by configuration. The shift/invariant causal separation—the primary control for CF patching—could therefore not be evaluated in these runs. [C10]

Results (shift items only; N = 60).
* GPT‑2 (124M): mean max effect 0.701 [0.574, 0.850]; sham baseline 5.41×10⁻⁶
* Qwen2.5‑0.5B: mean max effect 1.479 [1.071, 1.895]; sham baseline 0.0161
* Qwen2.5‑1.5B: mean max effect 0.511 [0.390, 0.645]; sham baseline 0.0149
* Qwen2.5‑3B: mean max effect 1.501 [1.129, 1.911]; sham baseline 0.0131
[E10a, E10b]

Intervention-type stratification (shift items only; N = 20 per type) is heterogeneous across models (negation / quantifier / role-swap). [E10a, E10b]

| Model | Negation mean max effect (CI) | Quantifier mean max effect (CI) | Role-swap mean max effect (CI) |
|---|---|---|---|
| GPT‑2 (124M) | 0.508 [0.452, 0.569] | 1.132 [0.902, 1.368] | 0.463 [0.287, 0.692] |
| Qwen2.5‑0.5B | 0.593 [0.509, 0.681] | 3.491 [2.828, 4.094] | 0.353 [0.203, 0.516] |
| Qwen2.5‑1.5B | 0.834 [0.721, 0.954] | 0.175 [0.119, 0.250] | 0.525 [0.244, 0.894] |
| Qwen2.5‑3B | 0.761 [0.665, 0.852] | 0.294 [0.219, 0.375] | 3.447 [2.896, 4.066] |

Interpretation. On the shift subset under this alignment protocol, donor-directed effects are present and above sham baselines across all four models. We do not claim monotonic scaling of CF patching effects with capacity in these runs; magnitudes vary by model and intervention type. This provides converging evidence that CPT-style donor-directed causal control extends beyond disambiguation to counterfactual intervention sensitivity. [E10a, E10b]

However, the interpretive weight of this result is limited by two factors:
1. Coverage: 60 of 140 items (43%) were evaluable, but the evaluated subset is shift-only under this protocol; invariants were skipped (empty_span) and graded items were excluded by configuration.
2. Missing control: without invariant items, we cannot establish that the patching effect tracks meaning-relevance of the intervention. Behavioral sham separation in AoM-CF (§5.3) provides indirect evidence, but a direct CF shift/invariant separation under patching remains a concrete next step. [C10]

Resolution of the missing-control constraint (designing substitution-type invariant items with non-empty receiver (base) spans under this protocol, or extending the patch definition to handle insertion-type invariants) is a concrete next step that would enable a full shift/invariant causal comparison under patching. [C10]

5.8.3. Summary: causal evidence across AoM components
Table 4 summarizes causal patching evidence across AoM components. In this revision, DISAMB, COH, and CF patching provide causal evidence across all three AoM components, with the CF results limited to shift items under the current alignment protocol. [E3, E4a, E10a, E10b, E10c, E10d, C9, C10]

[Table 4 about here]


6. Discussion: from metrics to mechanism to philosophy
6.1. What the results establish (and what they do not)
Within the limits of templated datasets and a small number of seeds, the results establish an AoM competence profile under controlled variation:
* context-sensitive disambiguation and coherence behavior. [E2a, E1]
* stronger response to meaning-altering interventions than to shams. [E8, E7]
* causal leverage of token-in-context states via patching. [E3, E4a, C3]
* constraint-directed coherence degradation under COH pseudo-ablation patching (constraint vs irrelevant spans, relevance-controlled). [E10c, E10d, C9]
* donor-directed causal effects on counterfactual sensitivity under intervention-span patching on a shift subset under the reported alignment protocol (coverage-limited). [E10a, E10b, C10]
The results do not settle reference, grounding, normativity, or consciousness.
6.2. From behavior to mechanism: what patching adds
Behavioral AoM metrics alone underdetermine mechanism. CPT patching adds an intervention:
1. Define a meaning-relevant behavioral readout on minimal pairs: relative preference between sense-diagnostic labels.
2. Intervene on internal variables: (h_{M,\ell,p}(x)) at the ambiguous target span.
3. Measure donor-directed change via (\operatorname{effect}_\ell) (margin shift; §4.6).
4. Compare to controls (sham patching; alignment checks).
This yields causal evidence that token-in-context states at specific depths control disambiguation behavior. [E3, E4a, C3]
COH pseudo-ablation patching extends this intervention logic to discourse constraints: injecting pseudo-ablated states at constraint-relevant spans degrades coherence substantially more than patching length-matched irrelevant spans, supporting a relevance-controlled causal effect on coherence under this protocol. [E10c, E10d, C9]
Depth observation (protocol-conditional). In these runs, COH effects peak at mean argmax depths of ≈0.32–0.40 (mean argmax layer/(L-1)), often deeper than DISAMB patching peaks (≈0.22–0.36). This is an observation about the reported templates and protocols, not a global integration claim. [E10c, E10d, E3, E4a]
Spatial structure (SDH constraint). The SDH stress test adds a second constraint: at fixed depth (\ell^*) and within a local position window (position_window=8, buffers ±2), off-target control patches often produce donor-directed effects comparable to target patches. This rules out a general claim that causal control at (\ell^*) is uniquely anchored to the target span. The natural mechanistic reading is local relational distribution within the tested neighborhood, not isolated token storage. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
6.3. Anticipating hostile reviewers: strongest objections and responses
Objection 1: “AoM is only surface.”
AoM is an empirical target. It is not a metaphysical thesis. If meaning requires more than AoM, the burden is to specify competence-relevant necessity and detectable signatures (§6.6).
Objection 2: “Templates are too easy; keyword baseline is strong.”
Correct. DISAMB is cue-vulnerable. The paper reports that confound and does not treat DISAMB accuracy as decisive. [E2a, E1, E9, C5]
Objection 3: “Patching could be artifact.”
Sham patching is near zero and layerwise profiles are structured. Stronger controls (random-span patching, outside-window patching, activation norm diagnostics) remain desirable and are listed as future work. [E3, E4a, C3]
Objection 4: “Tokenization confounds.”
Span alignment checks skip misaligned directions. The paper reports skipped counts. This remains a validity condition. [E3, E4a, C3]
Objection 5: “COH ablations reflect length.”
Ablate-irrelevant controls for length reduction. Ablate-relevant degrades more. That is the intended diagnostic asymmetry. [E7]
6.4. Relation to Kaplan, Braun, Wittgenstein, and Kant (without overreach)
Kaplan (1989): characters and stability ideals
Kaplan’s framework is a theory of meaning rules, not a neural implementation claim. Its stability ideal—stable, context-independent character rules—contrasts with the mechanistic picture in which AoM behavior is controlled by context-conditioned internal states that can be causally intervened on. This paper does not argue that Kaplan entails token-local retrieval. The empirical pressure is mechanistic: transformers implement AoM behavior through context-conditioned internal states that are causally leveraged by patching, and SDH shows local non-exclusivity at fixed depth under the tested protocol. Convergent mechanistic work at different granularity treats token representations as iteratively composed, context-conditioned residual-stream states; this supports the plausibility of a context-primacy implementation picture without validating the present patching protocol (Elhage et al. 2021; Lindsey et al. 2025). [C3, E3, E4a, C4, E5a, E5b, E5c, E5d, E5e, E5f]
Braun (1995): structured/relational character
The patching and specificity results are consistent with Braun’s relational emphasis. They do not constitute a direct test of philosophical semantics. The best-supported claim is compatibility: local relational distribution of causal control fits more naturally with relationalist pictures than with stability-ideal mechanistic intuitions. Our finding that causal control is distributed across a local neighborhood (§5.7) aligns more naturally with Braun’s emphasis on relational structure than with stability-ideal mechanistic intuitions. The SDH stress test provides a mechanistic constraint: within the tested window, meaning-relevant causal control is often not uniquely localized at the ambiguous span. [C3, E3, E4a, C4, E5a, E5b, E5c, E5d, E5e, E5f]
Wittgenstein (1953) and Cavell (1979): use and normativity
AoM operationalizes a text-internal proxy for use-sensitivity in Wittgenstein’s sense (“the meaning of a word is its use in the language,” PI §43). AoM does not operationalize normativity or social correction. The SDH pattern supplies a mechanistic analogue: competence can be controlled by relational processing within context, not by isolated token carriers.
Kant: phenomena and methodological modesty
AoM/CPT/SDH is a phenomena-first program: measure competence signatures and causal controllers without claiming to settle meaning “in itself.”
6.5. Falsifiers and limits
CPT is falsifiable in this framework (F1–F2). SDH is falsifiable as a local distribution claim once outside-window controls are implemented (F4).
Current limits:
* templated, small datasets;
* DISAMB cue vulnerability; [E9, C5]
* CF causal patching is coverage-limited and shift-only under the reported alignment protocol (60/140 items); insertion-type invariant shams yield empty receiver (base) spans (empty_span) and were skipped, so the shift/invariant causal separation remains untested under patching in this revision. [C10, E10a, E10b]
* limited seeds in reported AoM+CPT sweeps; [E2a, E1, E3, E4a]
* incomplete provenance for some legacy artifacts (explicitly reported). [R4]
6.6. Burden shift and the context trilemma
If AoM is robust and scalable, meaning-beyond-AoM positions must specify:
1. the additional required property,
2. why it is competence-relevant,
3. detectable signatures of its absence.
The present results are compatible with multiple interpretations, including:(A) context-dependent semantics realized by relational states,(B) competence without traditional semantics,(C) parasitic competence compressed from human practice.
This paper does not choose among (A)–(C). It reports causal-control evidence and spatial-structure constraints.
Søgaard (2025) maps five explicit positions on whether language models possess semantics, distinguishing inferential, referential, and other varieties; the present results supply shared empirical structure—operational competence signatures and causal-control evidence—on top of which those positions can be further articulated.

7. Conclusion
This paper defines appearance of meaning (AoM) as an operational competence profile under controlled context variation and states CPT as a causal-control hypothesis for transformer architectures.
GPT‑2 and Qwen2.5 models achieve AoM composites from 0.855 to 0.914 on these suites. [E2a, E1]
CPT-style activation patching yields strong donor-directed effects with sham baselines near zero and early-to-mid depth peaks (≈22–36% depth across the evaluated checkpoints, using 0-indexed layers and depth_frac=layer/(L-1)). [E3, E4a]
A fixed-layer target-specificity stress test across eight models evaluates SDH. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
Specificity is heterogeneous: in most models tested, donor-directed control at (\ell^*) is not uniquely localized to the target span within the tested local window (position_window=8, buffers ±2). [E5a, E5b, E5c, E5d, E5e, E5f]
Only Qwen2.5‑3B shows a robust positive specificity delta under this protocol. [E5a, E5b, E5c, E5d, E5e, E5f]
Control selection strategy materially affects measured specificity and must be reported. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
Several concrete extensions follow from documented limitations. For CF patching, recovering the missing shift/invariant causal comparison requires either (i) handling insertion-type invariants under positional alignment (e.g., via padded/remapped span alignment) or (ii) redesigning invariants as substitution-type shams with non-empty base spans under the current span definition. For COH, replacing padding-based pseudo-ablations with fluent natural-language distractor replacements would test robustness to the distribution-shift concern. Hardening the DISAMB suite against cue leakage and adding outside-window controls for SDH would strengthen the most protocol-conditional components.

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
Table 1. Behavioral AoM metrics for GPT‑2 and Qwen2.5 models on the AoM prototype suites. DISAMB accuracy is computed over minimal pairs (52 pairs, pair-bootstrap). CF reports shift-direction accuracy over 120 items (60 shift, 60 invariant). COH reports constraint accuracy in the main condition and in matched context-ablation controls (40 items per group). Brackets denote 95% bootstrap confidence intervals (1000 replicates where logged). [E2a, E1, C6, C7]
Model	AoM composite	DISAMB acc (CI)	CF shift-dir acc (CI)	COH main acc (CI)	COH ablate-irrelevant (CI)	COH ablate-relevant (CI)
openai-community/gpt2 (124M)	0.855	0.865 [0.808, 0.923]	0.850 [0.750, 0.933]	0.850 [0.725, 0.950]	0.875 [0.750, 0.975]	0.675 [0.525, 0.800]
Qwen/Qwen2.5-0.5B	0.857	0.913 [0.865, 0.962]	0.733 [0.617, 0.833]	0.925 [0.825, 1.000]	0.900 [0.800, 0.975]	0.450 [0.300, 0.600]
Qwen/Qwen2.5-1.5B	0.848	0.962 [0.923, 0.990]	0.783 [0.683, 0.883]	0.800 [0.675, 0.925]	0.775 [0.650, 0.900]	0.425 [0.275, 0.575]
Qwen/Qwen2.5-3B	0.914	0.942 [0.894, 0.981]	1.000 [1.000, 1.000]	0.800 [0.675, 0.925]	0.750 [0.600, 0.875]	0.425 [0.275, 0.575]
Keyword baseline (DISAMB suite): 0.913 (pair-bootstrap CI aligned to DISAMB bootstrap unit). [E2a, E1, C5]

Table 2. CPT-style activation patching aggregates
Table 2. CPT-style context-swap activation patching aggregates on the disambiguation suite (GPT‑2 and Qwen2.5). “Mean max effect” averages (over directions) (\max_\ell \operatorname{effect}_\ell) as defined in §4.6. Flip rate reports donor-directed flips at the empirically best layer. Sham patching replaces with the receiver’s own states (no-op). Block counts (L) are inferred from the per-layer effect columns in the cited artifacts (L = 1 + max{i : cpt_effect_layer_i is finite}). [E3, E4a, C3]
Model	Patched / total (skipped)	Mean max effect (CI)	Flip rate @ best layer (CI)	Mean argmax layer	Sham mean max effect (CI)
GPT‑2 (124M; 12 blocks)	100/104 (4)	0.667 [0.499, 0.856]	0.050 [0.010, 0.100]	3.96	2.28×10⁻⁶ [1.46×10⁻⁶, 3.15×10⁻⁶]
Qwen2.5‑0.5B (24 blocks)	100/104 (4)	1.861 [1.523, 2.263]	0.160 [0.090, 0.230]	5.10	0.000000 [0.000000, 0.000000]
Qwen2.5‑1.5B (28 blocks)	100/104 (4)	2.448 [1.939, 3.026]	0.190 [0.110, 0.270]	6.19	0.000078 [0.000000, 0.000234]
Qwen2.5‑3B (36 blocks)	100/104 (4)	3.114 [2.534, 3.800]	0.250 [0.180, 0.330]	8.69	0.000565 [0.000000, 0.001694]
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
Table 4. DISAMB context-swap patching rows are reproduced from Table 2 for comparison. COH rows report pseudo-ablation patching on constraint vs irrelevant spans (key contrast: constraint–irrelevant degradation asymmetry). CF rows report intervention-span patching on shift items only under the reported alignment protocol (60/140 items; invariants skipped; graded excluded). Brackets denote 95% bootstrap confidence intervals. [E3, E4a, E10a, E10b, E10c, E10d]
| Experiment | Model | Mean Max Effect (CI) | Sham Baseline | Key Contrast | Cohen’s d | N Used / Total |
|---|---|---|---|---|---|---|
| DISAMB context-swap | GPT‑2 | 0.667 [0.499, 0.856] | 2.28×10⁻⁶ | effect vs sham | — | 100/104 |
| DISAMB context-swap | Qwen2.5‑0.5B | 1.861 [1.523, 2.263] | 0.0 | effect vs sham | — | 100/104 |
| DISAMB context-swap | Qwen2.5‑1.5B | 2.448 [1.939, 3.026] | 7.80×10⁻⁵ | effect vs sham | — | 100/104 |
| DISAMB context-swap | Qwen2.5‑3B | 3.114 [2.534, 3.800] | 5.65×10⁻⁴ | effect vs sham | — | 100/104 |
| COH pseudo-ablation | GPT‑2 | 1.094 [0.970, 1.219] | 4.53×10⁻⁷ | constraint - irrelevant = 0.683 | 1.266 | 160/160 |
| COH pseudo-ablation | Qwen2.5‑0.5B | 1.687 [1.452, 1.945] | 0.00887 | constraint - irrelevant = 1.354 | 1.526 | 160/160 |
| COH pseudo-ablation | Qwen2.5‑1.5B | 1.838 [1.595, 2.133] | 0.0206 | constraint - irrelevant = 1.374 | 1.368 | 160/160 |
| COH pseudo-ablation | Qwen2.5‑3B | 1.403 [1.195, 1.627] | 0.00983 | constraint - irrelevant = 1.067 | 1.377 | 160/160 |
| CF intervention-span | GPT‑2 | 0.701 [0.574, 0.850] | 5.41×10⁻⁶ | shift only (invariants skipped; graded excluded) | — | 60/140 |
| CF intervention-span | Qwen2.5‑0.5B | 1.479 [1.071, 1.895] | 0.0161 | shift only (invariants skipped; graded excluded) | — | 60/140 |
| CF intervention-span | Qwen2.5‑1.5B | 0.511 [0.390, 0.645] | 0.0149 | shift only (invariants skipped; graded excluded) | — | 60/140 |
| CF intervention-span | Qwen2.5‑3B | 1.501 [1.129, 1.911] | 0.0131 | shift only (invariants skipped; graded excluded) | — | 60/140 |
