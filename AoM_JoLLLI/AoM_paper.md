The Appearance of Meaning: Context-Dependence and Semantic Competence in Transformer Architectures
Author: Felix Borck (Goethe University; felix.borck@gmail.com)  
ESSLLI note: This manuscript develops and extends an ESSLLI 2025 Student Session presentation. The operational AoM evaluation framework, the CPT-style activation-patching protocol, and the cross-family empirical results reported here (GPT‑2 and Qwen2.5) are new to this submission (i.e., not previously published).
Keywords: transformer language models; context dependence; semantic competence; causal interpretability; activation patching; philosophy of language

Evidence tags (for auditability): bracketed IDs like [E3], [R0], [C2] refer to rows in `AoM_evidence_contract.md`.

Abstract (≈200 words)
Large language models routinely elicit interpretations of meaning and context-sensitivity from competent speakers. This paper does not argue that such models possess meaning proper. Instead, it presents a protocol-and-constraint framework that isolates an empirical explanandum—appearance of meaning (AoM)—understood as a competence profile exhibited under controlled contextual variation: (i) context-sensitive disambiguation, (ii) selective sensitivity to meaning-altering edits over meaning-preserving shams, and (iii) discourse-level constraint tracking beyond length-matched controls.

The central question is mechanistic and philosophically diagnostic: in transformer language models, what internal variables causally control these meaning-like preference patterns when weights are held fixed? We test a Context-Primacy Thesis (CPT): that donor-directed changes in AoM-relevant preference margins can be induced by intervening on contextualized token-in-context states at characteristic depths, beyond sham baselines. Across GPT‑2 and Qwen2.5 checkpoints, activation patching yields structured, depth-localized donor-directed effects with near-zero sham controls.

A fixed-depth target-specificity stress test further indicates that, under the canonical nearby matched-span control used in Table 3, point estimates are positive across the non-GPT‑2 models while GPT‑2 remains a negative case, with the clearest support in the Qwen models and more heterogeneous support among the Llama models. We therefore interpret SDH as a local matched-span concentration result rather than as full semantic specificity. The result is disciplined mechanistic constraint-setting for philosophical interpretations: an empirically anchored account of context-dependence in transformer computation that informs, without settling, disputes about semantic competence, reference, and normativity.

1. Introduction
Transformer LLMs produce language that competent speakers often judge as coherent and context-responsive. This behavioral profile motivates a question that does not require settling metaphysical debates: what mechanisms explain systematic meaning-like behavior under controlled contextual variation?
1.1. From “meaning” to an empirical target: appearance of meaning (AoM)
Appearance of meaning is a term of art. AoM is a competence profile that (i) tracks core aspects of semantic competence in human language use and (ii) is measurable under controlled context manipulations. We operationalize AoM as:
1. Context-sensitive disambiguation (AoM-DISAMB): systematic selection among ambiguous senses conditioned on surrounding cues.
2. Minimal-pair intervention sensitivity (AoM-CF): systematic output shifts under meaning-altering interventions, and relative invariance under meaning-preserving shams.
3. Discourse-level coherence (AoM-COH): constraint tracking across extended contexts, with stronger degradation under relevant ablations than under length-matched irrelevant ablations.

In the AoM composite, the CF component is scored on shift items (directional sensitivity), while sham/invariant separation is treated as an internal-validity control reported alongside the main CF metric (§2; §5.3).
All three are computed from log-probability comparisons over labeled continuations. AoM targets controlled sensitivity patterns, not free-form generation quality. This corresponds to what Mahowald et al. (2024) term formal linguistic competence—text-internal statistical mastery of language structure—while explicitly bracketing functional linguistic competence involving world knowledge, reasoning, and grounding.
By meaning proper, we mean reference-involving, normatively governed semantic content (truth-conditional or otherwise public-rule constrained); AoM deliberately does not target these properties.
1.2. Mechanistic hypothesis: Context-Primacy Thesis (CPT)
Transformers compute contextualized states at every position. That fact alone does not decide between semantic theories. This paper states a causal hypothesis about proximate controllers:
Context-Primacy Thesis (CPT): In transformer language models, causal control of AoM-relevant preference margins is exerted by contextualized token-in-context states at specific depths, in the sense that intervening on those states (with weights fixed) produces donor-directed changes beyond sham baselines.
CPT is an architectural claim about causal control under intervention. It is not a claim that LLMs have meaning proper.[^1]
Epistemically, this is primarily a protocol-and-constraint contribution. It isolates an operational explanandum (AoM), specifies a sham-controlled intervention framework (CPT-style patching) for testing which internal variables control AoM-relevant preference margins with weights fixed, and reports mechanistic constraints (including a target-specificity stress test) that broader accounts must accommodate. The suites are intentionally small and templated to maximize identifiability and internal validity of the causal inferences, not to deliver population-level estimates of LLM semantic competence under naturalistic variation.
1.3. Contributions
1. Explanandum (AoM): a disciplined operational target—appearance of meaning—defined as a competence profile under controlled contextual variation (DISAMB/CF/COH), explicitly bracketing meaning proper.
2. Causal mechanism claim (CPT): a falsifiable interventionist thesis that AoM-relevant preference margins are causally controlled by contextualized token-in-context states at characteristic depths, supported by sham-controlled activation patching and demonstrated across GPT‑2 and Qwen2.5 checkpoints.
3. Constraint on simplistic locality (SDH stress test): a fixed-depth target-specificity analysis that yields a local matched-span contrast at one tested layer, with Table 3 point estimates favoring the target span across the non-GPT‑2 models, most clearly in Qwen, while remaining silent on global token‑atomic exclusivity and full semantic specificity.
4. Methodological transparency: sham controls, relevance-controlled ablations, deterministic log-probability evaluation, and an audit trail (evidence tags and provenance) provided in appendices/supplementary artifacts.
1.4. Philosophical payoff and discipline
The paper is methodologically conservative. It treats AoM as an explanandum and asks which causal mechanisms in transformer computation control it. If AoM is granted as robust behavior, then dismissals that classify it as “mere surface” must identify an additional property, argue that it is competence-relevant, and state detectable signatures. This paper supplies an empirical anchor that makes those demands concrete.
The interpretive contrast is between token‑privileged implementation glosses (stable lexical carriers with context as a parameter applied at the carrier site) and relational glosses (context integrated across positions in computation), in an architecture whose internal variables are contextualized throughout. The question here is mechanistic: what variables causally control AoM behavior in these systems?
Here **token‑privileged** denotes a *comparative asymmetry* claim (target-span interventions systematically dominate nearby controls), whereas **token‑atomic** denotes a stronger *exclusivity* gloss (the target position as the uniquely dominant locus of control). **SDH directly stress-tests token‑privilege** at a fixed depth within a local window; it bears on token‑atomic exclusivity only insofar as exclusivity readings predict such comparative dominance.
[^1]: Following Kant’s phenomena/noumena discipline (KrV A51/B75), we treat AoM as an empirical explanandum and bracket claims about meaning proper. The target is conditions for the appearance of competence, not metaphysical semantics.
2. Formal targets
We evaluate three components of appearance of meaning (AoM) using deterministic log-probability comparisons over labeled continuations. DISAMB measures context-sensitive sense selection on minimal pairs. CF measures selective sensitivity to meaning-altering edits relative to meaning-preserving shams via preference flips and preference-margin changes. COH measures discourse-level constraint tracking, including relevance-controlled ablations that match context length while varying informational role.
For CF item i with base prompt x_i, intervention prompt x_i', and labeled continuations (a_i, b_i), define preference margins:
Δ_M(x_i) = S̄_M(x_i, a_i) − S̄_M(x_i, b_i), and
Δ_M(x_i') = S̄_M(x_i', a_i) − S̄_M(x_i', b_i).
For COH item i with context x_i, valid continuation v_i, and invalid continuation u_i, define the coherence decision rule:
COH-ACC_i(M) = 1[S̄_M(x_i, v_i) > S̄_M(x_i, u_i)].
Appendix A provides additional scoring details, including log-mean-exp label aggregation and bootstrap resampling specifications.
2.1. CPT stated, predicted, and falsifiable
Let h_{M,ℓ,p}(x) denote the hidden state vector at transformer block ℓ and token position p when processing prompt x. CPT predicts that interventions that replace token-in-context states at meaning-relevant locations—holding weights fixed—should change meaning-relevant output preferences.
With label score s(·), define donor-label margin:
margin(y) = s(y) − max_{y' ≠ y} s(y').
For donor-directed context-swap patching at layer ℓ, define:
effect_ℓ = margin_patched(y_donor) − margin_base(y_donor).
Sham patching uses the same patch site and procedure but replaces receiver states with receiver states (no-op), giving the baseline for CPT effect comparisons.
We test CPT using context-swap activation patching on disambiguation pairs by patching the receiver’s hidden state at the ambiguous target span from a donor run, sweeping ℓ across layers.
Target-specificity stress test (SDH)
In addition to CPT, we run a fixed-depth target-specificity stress test about the spatial structure of causal control:
SDH: target-specificity stress test (local non-exclusivity check). At a consolidation depth, disambiguation-relevant causal control may be distributed across a local positional neighborhood around the ambiguous span, such that patching nearby off-target positions yields non-trivial donor-directed effects.
SDH is used here as a stress test on token‑privileged implementation glosses of CPT, not as a third co‑equal thesis.
The target-specificity protocol in §4.5 and results in §5.7 test this stress condition directly.  
Testable predictions (within this operational program)

**P1 (Causal control via patching; CPT).** Patching a receiver’s target-span state with the donor’s corresponding state shifts the receiver’s preference toward the donor’s sense label, beyond sham/no-op baselines.

**P2 (Layer localization; CPT).** Patching-induced effects show a reproducible depth profile with early-to-mid peaks rather than random diffuse changes.

**P3 (Informational role over length; AoM-COH control).** Removing relevant constraints degrades coherence more than removing length-matched irrelevant context.

**P4 (Locality within window; SDH stress test).** At fixed depth (ℓ*), nearby control patches within the local window can retain non-zero donor-directed effects, and the measured target-minus-control concentration can depend on control design.
Falsifiers (in the current experimental regime)

**F1 (Causal null; CPT).** Patching effects indistinguishable from sham/no-op controls across layers.

**F2 (No localization; CPT).** No stable layerwise structure; effects are non-replicable.

**F3 (Length dominance; AoM-COH control failure).** Coherence degradation tracks truncation length more than relevance-controlled ablations.

**F4 (Global diffusion; SDH).** If patches far outside the tested position window yield effects comparable to within-window patches, SDH (as a local distribution claim) is falsified. Outside-window controls are not part of the present submission, so this falsifier remains prospective.
2.2. What AoM, CPT, and SDH are not
AoM is not defined as truth-tracking, grounding, normativity, or consciousness. CPT does not claim lexical information is absent; it is a claim about proximate causal control under intervention for these operational targets. SDH is not a claim of global positional symmetry but a local-window claim tied to an explicit protocol (§4.5). None of these settles reference, grounding, or normativity without additional detectable competence signatures.

3. Related work
3.1. Context sensitivity and formal frameworks
Formal semantics provides explicit context-sensitive models of interpretation (Kaplan 1989; Heim & Kratzer 1998), while transformer representations are strongly context-dependent in practice (Ethayarajh 2019). The present question is mechanistic: whether this contextual dependence is causally used in model behavior.
3.2. From probing to causal intervention
Probing establishes decodability, not causal use (Belinkov & Glass 2019; Rogers et al. 2020). Interchange-style interventions provide a causal framework by substituting aligned internal variables across inputs (Geiger et al. 2021). Our CPT protocol is a sham-controlled token-in-context specialization of this intervention family (formalized in §2.1; protocol details in Appendix C.1), positioned alongside patching/circuit work in mechanistic interpretability (Elhage et al. 2021; Wang et al. 2023).
3.3. Surface-statistics objections and philosophical framing
The “surface statistics” objection (Bender & Koller 2020) motivates explicit competence tests and causal controls rather than benchmark-only claims. Methodologically, we follow a Turing-style strategy of operationalizing testable competence profiles (Turing 1950), while treating Kaplan/Braun/Wittgenstein as interpretive frameworks for what implementation constraints would matter philosophically.

4. Methods
This section states only the load-bearing protocol choices for internal validity; full formal and implementation details are in Appendices A–D.
4.1. Models evaluated
For AoM+CPT sweeps we evaluate GPT‑2 (124M; Radford et al. 2019) and Qwen2.5-{0.5B, 1.5B, 3B} (Qwen Team 2024); for SDH we add Qwen3‑4B (Yang et al. 2025) and Llama‑3.{2‑1B, 2‑3B, 1‑8B} (Grattafiori et al. 2024; Meta 2024). Runs use deterministic `model.eval()` log-probability scoring (no sampling). [E2a, E1, E3, E4a, E5a, E5b, E5c, E5d, E5e, E5f]
4.2. Datasets
We use three controlled suites chosen for intervention identifiability: DISAMB (52 pairs), CF (shift/invariant/graded), and COH (80 items plus matched ablations). Schemas and validation are in Appendix B. [C1, E2a, E1]
4.3. Scoring, uncertainty, and controls
Scoring uses deterministic mean per-token log-probabilities; uncertainty is reported as 95% bootstrap CIs. Controls are central: sham patching (no-op), relevance-controlled contrasts (COH/CF), and explicit alignment/skip accounting. Formal scoring and CI details are in Appendix A; protocol details are in Appendix C. [R3, C8, C3, C9, C10]
4.4. CPT-style activation patching
To test CPT, donor hidden states are injected into aligned receiver positions on DISAMB minimal pairs with weights fixed, and donor-directed margin shifts are measured across layers. The `effect_ℓ` metric is defined in §2.1; aggregate reporting and protocol details are in Appendix C.1.
4.5. Target-specificity stress test (SDH)
SDH compares target-span and nearby control-span patching at fixed quarter-depth in a local window (`position_window=8`, exclusion buffers ±2), with deterministic control selection; strategy-specific diagnostics are reported outside the main narrative (Appendix D / supplement). [C4, E5a, E5b, E5c, E5d, E5e, E5f]
4.6. Extensions beyond disambiguation
We also run COH pseudo-ablation patching as a relevance-controlled extension in the main paper; CF intervention-span patching is reported in main results, with type-stratified diagnostics in supplement/appendix materials. [C9, C10, E10c, E10d, E10e]

5. Results
The main mechanistic results cover DISAMB+CPT+SDH, with COH and CF patching reported as causal extensions; behavioral AoM metrics are summarized in Table 1.
5.1. Behavioral AoM metrics
AoM composite scores are 0.869 (GPT‑2), 0.849 (Qwen2.5‑0.5B), 0.869 (Qwen2.5‑1.5B), and 0.903 (Qwen2.5‑3B). [E2a, E1]
5.2. AoM-DISAMB: high accuracy with a baseline caveat
DISAMB accuracy is high across models, and remains high on the keyword-failure subset where the cue heuristic fails. The keyword baseline is non-trivial (0.615 [0.558, 0.673]) but does not explain model performance on that subset; details are in Table 1 and Appendix artifacts. [E2a, E1, C5, E11b]
5.3. AoM-CF: controlled sensitivity with sham separation
Behavioral CF shift-direction accuracy is reported in Table 1. As an internal-validity control, invariant (sham) items exhibit smaller absolute preference-margin changes than shift items on this suite (Appendix / artifacts). [E2a, E1, E8]
5.4. AoM-COH: coherence tracks informational role, not only length
COH main accuracy and ablation contrasts (Table 1) show that ablate-relevant degrades more than length-matched ablate-irrelevant across models, supporting an informational-role effect beyond length alone. [E2a, E1, E7]
5.5. CPT-style activation patching: causal leverage for token-in-context control
CPT patching on DISAMB uses 104/104 patched donor→receiver directions in hardened runs and yields robust donor-directed margin shifts with near-zero sham baselines. The key aggregates are in Table 2, and Figure 1 shows structured early-to-mid depth profiles across GPT‑2 and Qwen2.5. [E3, E4a, C3]
5.6. Scaling summary (AoM + CPT sweep)
CPT effects are present in GPT‑2 and Qwen2.5 (non-monotonic within Qwen2.5); within Qwen2.5, CF behavioral shift-direction improves with scale on this suite; and COH ablation asymmetries are robust though main COH accuracy is not uniformly monotonic. [E3, E4a, E2a, E1, E7]
5.7. Target-specificity stress test: SDH results
Under the canonical matched-token nearby-span control reported in Table 3, the target-minus-control signed delta has positive point estimates across the non-GPT-2 models while GPT-2 is negative at fixed quarter-depth, with the clearest support in the Qwen models and more heterogeneous support among the Llama models. We therefore interpret SDH as a local target-span-concentration result relative to nearby matched spans under this protocol, not as full semantic specificity. [E5a, E5b, E5c, E5d, E5e, E5f, C4]
This Table 3 SDH analysis is protocol-conditional: it is evaluated at fixed (ℓ*) within a bounded local window and one matched-token control path, not at each model’s sweep-selected maximizing layer and not over global position space. Because the reported run does not vary control-selection strategy within the table, we do not draw strategy-sensitivity conclusions from it. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
5.8. COH causal patching extension (main)
COH pseudo-ablation patching shows a large and consistent constraint>irrelevant asymmetry with near-zero sham baselines across GPT‑2 and Qwen2.5 checkpoints, supporting relevance-sensitive causal effects on the coherence margin under this intervention regime. The main text reports the compact aggregate in Table 4; protocol details and caveats (including padding confounds) are in Appendix C.2 and §6.3. [E10c, E10d, C9]
5.8.3. Summary of causal evidence across components
Figure 2 shows panel-local contrasts across components: robust COH constraint>irrelevant separation across all four checkpoints, alongside heterogeneous CF shift>control separation with a Qwen2.5‑1.5B null discussed in §5.9. [E10e, E10c, E10d, C9, C10]
[Figure 2 about here] [E10e, E10c, E10d, C9, C10]
5.9. CF intervention-span patching (main)
On the relevance-controlled CF patching set (shift N=60 vs substitution controls N=20), GPT‑2, Qwen2.5‑0.5B, and Qwen2.5‑3B show clear shift>control separation (d = 1.179, 0.677, 0.798), while Qwen2.5‑1.5B is null (d ≈ 0). Full per-model breakdowns are in Table 4. [E10e, C10]
The expected relevance-controlled separation appears in 3/4 checkpoints, while Qwen2.5‑1.5B is a genuine checkpoint-level null under this intervention protocol. This qualifies any blanket generalization from CF patching and motivates model- and protocol-conditional interpretation. [E10e, C10]
The Qwen2.5‑1.5B behavioral CF shift-direction score remains high (0.800 in Table 1), yet CF patching shows no shift>control separation. Appendix F’s type-stratified outputs indicate a quantifier-specific weakness for this checkpoint (quantifier mean max effect 0.159), supporting a protocol-conditional heterogeneity reading rather than global incompetence on CF behavior. [E2a, E1, E10e]


6. Discussion: from metrics to mechanism to philosophy
6.1. What the results establish (and what they do not)
Within the limits of templated datasets and a small number of seeds, the results establish an AoM competence profile under controlled variation:
context-sensitive disambiguation and coherence behavior [E2a, E1], stronger response to meaning-altering interventions than to shams [E8, E7], causal leverage of token-in-context states via patching [E3, E4a, C3], constraint-directed coherence degradation under COH pseudo-ablation patching (constraint vs irrelevant spans, relevance-controlled) [E10c, E10d, C9], and relevance-controlled CF intervention-span patching with shift>control separation in 3/4 checkpoints (and a Qwen2.5‑1.5B null) [E10e, C10].
The results do not settle reference, grounding, normativity, or consciousness.
6.2. From behavior to mechanism: what patching adds
Behavioral AoM metrics alone underdetermine mechanism. CPT patching adds an intervention:
1. Define a meaning-relevant behavioral readout on minimal pairs: relative preference between sense-diagnostic labels.
2. Intervene on internal variables: h_{M,ℓ,p}(x) at the ambiguous target span.
3. Measure donor-directed change via effect_ℓ (margin shift; Appendix C.1).
4. Compare to controls (sham patching; alignment checks).
This yields causal evidence that token-in-context states at specific depths control disambiguation behavior. [E3, E4a, C3]
COH pseudo-ablation patching extends this intervention logic to discourse constraints: across the tested model families (GPT‑2 and Qwen2.5‑{0.5B, 1.5B, 3B}), injecting pseudo-ablated states at constraint-relevant spans degrades coherence substantially more than patching length-matched irrelevant spans (d = 1.27–1.53), supporting a relevance-controlled causal effect on coherence under this protocol. [E10c, E10d, C9]
The SDH stress test adds a narrower mechanistic constraint: under a fixed-layer nearby-matched-span comparison, nearby controls remain causally active, while Table 3 point estimates favor the target span across the non-GPT‑2 models, most clearly in Qwen. The appropriate mechanistic reading is target-span concentration under a local matched-span control, not full semantic specificity or token-atomic exclusivity. [C4, E5a, E5b, E5c, E5d, E5e, E5f]
6.3. Threats to validity and robustness checks
Templates and cue leakage. The keyword baseline (0.615) confirms that surface cues exist in these templates, but the keyword-failure subset analysis (40/52 pairs) shows that model DISAMB accuracy remains high when that heuristic fails. This is the core control against a pure-cue explanation in the current suite. [E2a, E1, C5, E11b]

Patching artifacts and alignment constraints. Sham patching remains near zero and layerwise profiles are structured, but stronger diagnostics (outside-window patching, random-span controls, activation-norm checks) are still desirable. Alignment checks and skip accounting remain hard validity conditions for interpreting any patching result. [E3, E4a, C3]

COH pseudo-ablation scope. Length-matched irrelevant ablations reduce a simple length-only explanation, and the observed constraint>irrelevant asymmetry is large across models. However, padding-based pseudo-ablations can conflate informational relevance with information presence; fluent distractor replacements are the next robustness check. [E10c, E10d, C9]

**Objection: “The observed control is still just complex syntactic-distributional matching, not semantic competence.”** Suppose, for the sake of argument, that the causal controllers isolated here-those that govern context-sensitive disambiguation, selective intervention sensitivity, and discourse-level constraint tracking-could be fully characterized in purely syntactic-distributional terms. This would not refute the present framework; it would place it on a deflationary horn of the interpretive space: substantial AoM-level competence can be causally realized without reference-involving or normatively governed **meaning proper**. Importantly, the present results do not purport to identify the patched variables as “semantic” rather than “syntactic.” What they establish is that AoM-relevant preference margins are controlled by **context-conditioned, relationally integrated internal states** at characteristic depths (with near-zero sham baselines and structured depth profiles), and that this control is often not token-atomic at the tested fixed depth (SDH). In this setting, labeling the mechanism “mere syntax” is not yet an explanation unless it is paired with an additional claim: which empirically detectable language behaviors, beyond the AoM competence profile operationalized here, remain exclusively diagnostic of meaning proper and why. As Piantadosi & Hill (2022) argue, sufficiently rich internal structural relations can blur the traditional syntax/semantics boundary; the residual dispute must therefore be cashed out in further measurable competence signatures rather than terminological stipulations. On this reading, “mere syntax” is either a promissory mechanistic research program (to specify the relevant structural computations) or a concession that syntactic-distributional structure is causally sufficient for the appearance of meaning.
6.4. Context, stability ideals, and implementation glosses: what the data can and cannot discriminate

A standing risk in philosophy-of-LLMs is a category mistake: moving too quickly from (i) a semantic framework as a theory of meaning to (ii) a claim about how a particular architecture must implement it. Kaplan’s character/content distinction, Braun’s critiques of overly simple factorization, and Wittgenstein’s emphasis on use are semantic or meta-semantic positions; none of them entails a specific neural implementation. The aim here is narrower. We use these frameworks to articulate contrasting implementation ideals—ways one might expect “context dependence” to be routed and localized in a computational system—and then ask what our intervention results constrain.

Levels of claim.
"Context" in the operational sections denotes linguistic co-text: the prefix and discourse material available to the model at inference time. We do not manipulate Kaplanian extra-linguistic context parameters (speaker, time, place), and we do not claim to test indexicals as such. The question is instead: given a system that exhibits robust context-sensitive competence on controlled items, is the causal control of that competence plausibly token‑privileged (anchored primarily at the target expression’s site) or relational (distributed across positions participating in an integration structure)?

A stability ideal and a token‑privileged gloss.
Kaplan’s framework is naturally associated with a stability ideal: an expression has a relatively stable contribution (character) and context supplies parameters that select a content. This is a semantic decomposition, not a neural story. But one tempting implementation gloss—common in informal talk about “word meaning inside the model”—is token‑privileged: the ambiguous token position is treated as the primary carrier of the relevant contribution, with surrounding context modulating that token’s state. On such a gloss, one expects a strong asymmetry: interventions at the target span should systematically dominate interventions at nearby off‑target spans, because the expression’s own site is privileged as the locus of control.

What CPT supports (and what it does not).
Our CPT-style patching results provide sham-controlled causal evidence that contextualized token‑in‑context states at specific depths exert donor-directed control over disambiguation preferences. This supports a context‑primacy picture in a precise interventionist sense: the causal variables that control meaning‑like preference margins are not static type‑level carriers, but context-conditioned states computed over the prefix. Importantly, CPT by itself is neutral on whether causal control is token‑atomic. A model could satisfy CPT even if control is distributed across multiple positions, provided patching at the target span intercepts some of that distributed signal.

The SDH stress test as a constraint on token privilege (not a semantic-theory verdict).
The fixed-depth target-specificity analysis adds a local constraint, but Table 3 is better read as target-span concentration than as local non-specificity. Nearby matched-span controls retain non-zero donor-directed effects; GPT-2 is negative; and among the non-GPT-2 models the positive deltas are clearest in Qwen and more heterogeneous in Llama. This does not show that the target span is always privileged, and it does not amount to full semantic specificity; the result is a local matched-span contrast at one fixed layer, and the pattern may vary by depth or control design. Nor does it discriminate Kaplan versus Braun as semantic theories. What it does show is that nearby matched spans can retain non-zero causal leverage even when point estimates often favor target-span patching under the reported local control.

What SDH rules out.
At the tested depth (ℓ*) and within the specified local window, Table 3 does not support a claim of target-span exclusivity or a claim of equal leverage across nearby spans. GPT‑2 is negative, several intervals are wide, and control effects remain non-zero.

What SDH is consistent with.
The observed pattern is consistent with an implementation in which target-span interventions often have larger point-estimate leverage than nearby matched-span controls while those nearby positions still provide partial causal access to the same downstream preference margin. This supports a mixed picture: local concentration without token-atomic exclusivity.

What SDH does not diagnose.
Positional non-exclusivity alone does not establish that the controlling information is organized by Braun-style relational semantic structure (for example, syntactic or discourse relations) rather than by generic information diffusion in a shared stream. SDH as implemented here varies position but does not control structural relevance of the patched positions. The present results therefore place pressure on strong token-privileged localization glosses under the tested protocol, but they do not yet discriminate relational-structure hypotheses from diffusion-based alternatives.

What a Braun-diagnostic stress test would require.
A diagnostic test for Braun-style relational structure would hold positional distance roughly fixed while varying whether patched positions stand in specific syntactic or discourse relations to the ambiguous expression (for example, argument head, modifier, or coreference antecedent) versus structurally irrelevant roles at comparable distance. Under a relational-structure hypothesis, causally effective off-target positions should concentrate in structurally related roles, not merely in a local neighborhood. This paper does not implement that control.

Wittgenstein and the limits of the operational target.
Wittgensteinian themes help locate what is—and is not—being claimed. On a use-oriented view, context is not merely an input that selects among pre-existing meanings; it is part of what constitutes the use that makes an utterance intelligible. AoM is explicitly a use-profile in this restricted sense: DISAMB/CF/COH test whether preferences track controlled changes in co-text, meaning-altering edits, and discourse constraints, and patching identifies internal variables that causally control those sensitivities. But this remains text-internal and non-normative. Nothing here establishes public criteria, social correction, or rule-following as a practice. The philosophical point is narrower: within transformer architectures, the proximate controllers of meaning-like behavior are context-conditioned and often not token‑atomic at the tested depth—so any position that treats context as merely auxiliary to stable lexical atoms must say how it expects such a picture to be realized in this architecture, and what empirical signatures would distinguish it.

Summary.
The experiments do not adjudicate semantic theories, but they do constrain plausible implementation glosses of stability-based pictures. In the tested intervention regimes, context dependence is not merely an external parameter applied to fixed lexical carriers; it is constitutively integrated into the causal variables that control meaning-like preference patterns.
6.5. Falsifiers and limits
CPT is falsifiable in this framework (F1–F2). SDH is falsifiable as a local distribution claim once outside-window controls are implemented (F4).
The current evidence is subject to several limitations. The suites remain small and templated (DISAMB 52 pairs plus the current CF/COH sets), which strengthens internal validity for a first-pass causal test but limits naturalistic generalization. Surface cues remain present in DISAMB (keyword baseline 0.615) despite high performance on the 40/52 keyword-failure pairs, so further adversarial hardening is needed [E2a, E1, C5, E11b]. CF causal patching remains heterogeneous across checkpoints (including a Qwen2.5‑1.5B null) under the current span definition and controls, motivating stronger invariant controls [C10, E10e]. Reported AoM+CPT sweeps also use limited seeds [E2a, E1, E3, E4a].
6.6. Burden shift and the context trilemma
The AoM program is intended to change what must be argued, not to settle metaphysics. The results in this paper provide (i) behavioral evidence for a consistent (suite-conditional) pattern of context-sensitivity across DISAMB/CF/COH under controlled manipulations, and (ii) causal evidence that internal token-in-context states at specific depths exert donor-directed control over AoM-relevant preference margins under activation patching, with sham baselines near zero and structured depth profiles. SDH adds a narrower mechanistic constraint: under a fixed-layer nearby-matched-span comparison, nearby controls remain causally active, while Table 3 point estimates favor the target span across the non-GPT‑2 models, most clearly in Qwen. These findings do not establish meaning proper, but they do make purely dismissive “mere surface” moves less informative: to maintain a meaning-beyond-AoM position while conceding AoM-like competence, one must identify an additional property that is (1) competence-relevant, (2) plausibly absent here, and (3) associated with detectable signatures that could in principle be measured.
If one insists that a system exhibiting AoM under relevance/sham controls and showing sham-controlled, depth-localized causal context-routing is merely a “stochastic parrot,” then “parrot” must be broadened to include systems with highly structured, relationally integrated, and causally measurable discourse-level constraint tracking-at which point “surface statistics” ceases to be a deflationary dismissal and becomes a substantive mechanistic explanans.
However, the force of this burden-shift is **conditional** on closing the remaining validation gaps in the present experimental regime. In this revision, DISAMB cue controls and COH relevance controls are strong, while CF causal diagnostics are mixed across checkpoints. As remaining gaps are closed—by broadening invariant controls, extending COH patching with fluent distractor replacements (in addition to padding), and adding outside-window SDH controls—the AoM evidence would increasingly constrain meaning-beyond-AoM proposals to be explicit about what more is required and how it would show up empirically.
The present results remain compatible with multiple interpretations: (A) context-dependent semantics realized by non-token-atomic control states; (B) robust competence without traditional semantics; (C) compressed, practice-derived competence without normativity. This paper does not adjudicate among (A)–(C). It aims to supply a shared empirical substrate—operational competence signatures plus causal-control and spatial-structure constraints—on which broader philosophical positions can make more discriminating commitments. Søgaard (2025) is cited as a map of the dialectical space—surveying five positions on whether and in what sense LMs have semantics (ranging from outright denial, through deflationary/instrumentalist views, to qualified attribution)—rather than as a premise in any argument here.

7. Conclusion
This paper supplies a disciplined empirical and mechanistic substrate for disputes about semantic competence in transformer architectures. The AoM framework isolates a controlled competence profile; CPT-style activation patching identifies contextualized token-in-context states at characteristic depths as causal controllers of that profile, with near-zero sham baselines and structured depth profiles across GPT-2 and Qwen2.5; and the SDH stress test adds a narrower fixed-layer matched-span contrast in which point estimates favor the target span across the non-GPT‑2 models while GPT‑2 remains a negative case, with the clearest support in Qwen and more heterogeneous support in Llama. These constraints do not establish full semantic specificity, but they do make it more costly to dismiss transformer context-sensitivity as “mere surface” without specifying what additional competence property is required and what its detectable signatures would be.

More broadly, the framework demonstrates that transformer architectures offer the philosophy of language a fully observable, manipulable system in which hypotheses about implementation, context-dependence, and semantic competence can be subjected to sham-controlled causal testing rather than resting solely on a priori reasoning.

Three extensions are highest priority. First, harden CF relevance controls and diagnose the Qwen2.5-1.5B null via per-type behavioral stratification. Second, replace padding-based COH pseudo-ablations with fluent, semantically irrelevant distractors to separate relevance from information-presence effects. Third, add outside-window SDH controls and structure-controlled off-target positions to discriminate relational-structure hypotheses from diffusion-based alternatives. Together these steps would convert the present protocol-and-constraint framework into a progressively sharper instrument for connecting transformer mechanisms to debates about semantic competence.

Appendix A. Additional scoring and uncertainty details
A.1. Scoring and label aggregation
Let M be a causal language model defining a conditional distribution over token sequences. For a prompt x and candidate continuation y = (y_1, ..., y_T), define the raw log-probability score:
S_M(x, y) = Σ_{t=1}^{T} log P_M(y_t | concat(x, y_{<t})).
Define the length-normalized score:
S̄_M(x, y) = (1/T) · S_M(x, y).
Reported results use length-normalized scoring (the evaluation flag `no_length_norm=False`), i.e., label scores are based on S̄_M unless stated otherwise.
When a label corresponds to multiple candidate continuations Y = {y^(1), ..., y^(k)}, label scores are aggregated via log-mean-exp:
s(Y) = log((1/k) · Σ_{j=1}^{k} exp(S̄_M(x, y^(j)))).

A.2. AoM composite
AoM(M) = (DISAMB(M) + CF(M) + COH(M)) / 3.
This composite is instrumental and not a definition of meaning.

A.3. Statistical uncertainty
Metrics report 95% bootstrap confidence intervals. Bootstrap replicates are 1000 in logged runs (`bootstrap_n=1000` where recorded). Resampling units are minimal pairs (DISAMB), items (CF/COH), and donor→receiver directions (CPT/SDH). Forward passes are run in evaluation mode with no sampling; reported seeds affect bootstrap resampling and deterministic selection procedures. [C8, R3]

Appendix B. Datasets and validation
Three JSONL suites are used: DISAMB (52 minimal pairs across 6 ambiguous word types), CF (140 items: 60 shift, 60 invariant, 20 graded), and COH (80 items with matched ablation variants; 240 contexts total). Schema validation and tokenization boundary checks are documented in the replication bundle supplementary materials (including full schemas, template examples, and validation diagnostics). [C1, E2a, E1]

Appendix C. Activation patching protocols
C.1. CPT-style activation patching
Aggregates and controls. Per donor→receiver direction i, compute layerwise effect_{i,ℓ} values (with effect_ℓ defined in §2.1). Define maxeff_i = max_ℓ effect_{i,ℓ} and ℓ_i* = argmax_ℓ effect_{i,ℓ}. Reported “mean max effect” is the bootstrap mean of maxeff_i over directions (95% CI); because it summarizes the largest observed effect within each direction’s sweep, we treat it as a descriptive post-selection summary rather than a confirmatory estimate. Reported `mean_argmax_layer` is the mean of the per-direction maximizing layers ℓ_i*, not a single shared model-level optimum. “Flip rate @ per-direction argmax layer” counts donor-directed flips at ℓ_i* when the unpatched receiver prediction is not y_donor and the patched prediction equals y_donor. Sham patching replaces receiver states with receiver states (no-op). Full protocol/implementation details are in supplementary materials.

C.2. Causal patching beyond disambiguation (COH and CF)
COH pseudo-ablation patching. For each COH item, construct alignment-preserving pseudo-ablated variants by replacing either the constraint sentence or an irrelevant sentence with padding tokens. Patch the corresponding span states from pseudo runs into the main run across layers ℓ. The effect metric is positive degradation in the coherence margin (valid vs invalid continuation), and the primary diagnostic is the constraint–irrelevant degradation asymmetry; sham patching is a no-op baseline. [C9]

CF intervention-span patching. For each CF item, identify the minimal contiguous token-level divergence between base prompt x and intervention prompt x'. With `span_mode=left_aligned_truncated`, donor and receiver spans are left-aligned and truncated to shared length when needed; donor states from x' are patched into x at each layer ℓ. The primary metric is donor-directed preference-margin shift with receiver→receiver sham no-op. Reported CF patching excludes graded items and uses 60 shift items plus 20 substitution-invariant controls with non-empty divergence spans. [C10]

Appendix D. SDH protocol details
The SDH target-specificity test compares donor-directed effects from target-span versus nearby matched-span control patching at fixed depth ℓ* = round(0.25 × (L−1)) under `position_window=8`, exclusion buffers (`buffer=2`, `donor_buffer=2`), and deterministic selection (`selection_seed=0`; implementation fallback order `matched_token`, `same_index`, `same_index_relaxed`). In the Table 3 SDH run, all reported controls come from the `matched_token` path rather than from a within-table strategy mixture. For each direction i, compute E_target,i = effect_{ℓ*} under target-span patching, E_ctrl,i = effect_{ℓ*} under matched-span control patching, Δ_i = E_target,i − E_ctrl,i, and win_i = 1[Δ_i>0]; report bootstrap means and 95% CIs for E_target, E_ctrl, Δ, and win rate. Strategy-comparison claims require separate artifacts and are not inferred from Table 3. [E5a, E5b, E5c, E5d, E5e, E5f, C4]

Appendix E. Provenance and evidence contract
Evidence tags (e.g., `[E3]`, `[R0]`, `[C2]`) map manuscript claims to rows in `AoM_evidence_contract.md`.
Supplementary materials in the project repository and archived replication bundle include expanded dataset schemas/templates/validation checks, CF span-alignment and interpretive-scope diagnostics, full deterministic provenance listings (hardware/software, commits, hashes, checkpoint revisions), and per-artifact manifests/run metadata.
This appendix intentionally omits full hash/version listings from the printed paper. [R0, R3, C2, C11]

Appendix F. Additional analyses
Intervention-type stratification (shift items; N = 20 per type) is heterogeneous across models. [E10e]

| Model | Negation mean max effect (CI) | Quantifier mean max effect (CI) | Role-swap mean max effect (CI) |
|---|---|---|---|
| GPT‑2 (124M) | 0.508 [0.452, 0.569] | 1.132 [0.902, 1.368] | 0.463 [0.287, 0.692] |
| Qwen2.5‑0.5B | 0.557 [0.483, 0.630] | 3.538 [2.859, 4.150] | 0.341 [0.197, 0.506] |
| Qwen2.5‑1.5B | 0.806 [0.700, 0.925] | 0.159 [0.106, 0.231] | 0.506 [0.219, 0.881] |
| Qwen2.5‑3B | 0.753 [0.663, 0.839] | 0.338 [0.278, 0.397] | 3.453 [2.903, 4.034] |

Sensitivity check (noun-substitution controls; criterion-based exclusion). Three noun-substitution controls produced unusually large invariant effects across Qwen models, consistent with weak lexical calibration for those specific "synonym" substitutions. Excluding those 3 controls yields updated shift-vs-control Cohen's d values: GPT-2 1.25 (+0.08), Qwen2.5-0.5B 0.81 (+0.13), Qwen2.5-1.5B 0.28 (+0.29), Qwen2.5-3B 0.99 (+0.19). The Qwen2.5-1.5B null therefore partially lifts but remains weak, so the checkpoint-level null is not solely an artifact of those control items. This exclusion is criterion-based (lexical calibration), not outcome-based. [E10e]

Statements and Declarations
Funding: This research received no external funding (self-funded).
Competing interests: The author declares no competing interests.
Data availability: Replication package, datasets, run manifests, and deterministic provenance (hardware, commits, hashes, checkpoint revisions) are available at https://github.com/Satori-1618/AoM_CPT and in the archived replication bundle linked from the repository.
Code availability: Evaluation code, table-generation scripts, and release-gate tooling are available in the same repository and archived replication bundle.
AI assistance: LLM tools were used for coding and language-editing support (including Anthropic Claude Code and OpenAI GPT-5/Codex variants). Hypotheses, analyses, result interpretation, and all final scientific claims were authored and verified by the human author.

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
Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783. https://doi.org/10.48550/arXiv.2407.21783
Heim, I., & Kratzer, A. (1998). Semantics in Generative Grammar. Blackwell.
Jannai, D., Meron, A., Lenz, B., Levine, Y., & Shoham, Y. (2023). Human or Not? A Gamified Approach to the Turing Test. arXiv:2305.20010.
Kaplan, D. (1989). Demonstratives. In Almog, Perry, & Wettstein (Eds.), Themes from Kaplan (pp. 481–563). Oxford University Press.
Kant, I. (1781/1787). Critique of Pure Reason (A/B pagination).
Katz, D. M., Bommarito, M. J., Gao, S., & Arredondo, P. (2024). GPT‑4 Passes the Bar Exam. Philosophical Transactions of the Royal Society A, 382, 20230254.
Lappin, S. (2024). Assessing the Strengths and Weaknesses of Large Language Models. Journal of Logic, Language and Information, 33, 9–20. https://doi.org/10.1007/s10849-023-09409-x
Li, B. Z., Nye, M., & Andreas, J. (2021). Implicit Representations of Meaning in Neural Language Models. ACL-IJCNLP 2021.
Lindsey, J., et al. (2025). On the Biology of a Large Language Model. Transformer Circuits. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
Mahowald, K., Ivanova, A. A., Blank, I. A., Kanwisher, N., Tenenbaum, J. B., & Fedorenko, E. (2024). Dissociating language and thought in large language models. Trends in Cognitive Sciences, 28(6), 517–540.
Meta. (2024). Llama-3.2-1B model card. Hugging Face. https://huggingface.co/meta-llama/Llama-3.2-1B
Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. (ROME; arXiv / workshop versions).
Milliere, R., & Buckner, C. (2024a). A Philosophical Introduction to Language Models—Part I: Continuity With Classic Debates. arXiv:2401.03910.
Milliere, R., & Buckner, C. (2024b). A Philosophical Introduction to Language Models—Part II: The Way Forward. arXiv:2405.03207.
Mollo, D. C., & Milliere, R. (2023). The Vector Grounding Problem. arXiv:2304.01481. (Forthcoming in Philosophy and the Mind Sciences.)
Piantadosi, S. T., & Hill, F. (2022). Meaning without reference in large language models. arXiv / preprint versions.
Qwen Team. (2024). Qwen2.5 Technical Report. arXiv:2412.15115. https://doi.org/10.48550/arXiv.2412.15115
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI (GPT‑2 technical report).
Rogers, A., Kovaleva, O., & Rumshisky, A. (2020). A Primer in BERTology: What We Know About How BERT Works. TACL.
Søgaard, A. (2025). Do Language Models Have Semantics? On the Five Positions. ACL 2025. https://aclanthology.org/2025.acl-long.1258.pdf
Turing, A. M. (1950). Computing Machinery and Intelligence. Mind, 59(236), 433–460.
Veltman, F. (1996). Defaults in Update Semantics. Journal of Philosophical Logic.
Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020). Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS 2020.
Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. ICLR 2023. arXiv:2211.00593.
Wittgenstein, L. (1953). Philosophical Investigations. Blackwell.
Yang, A., Li, A., Yang, B., et al. (2025). Qwen3 Technical Report. arXiv:2505.09388. https://doi.org/10.48550/arXiv.2505.09388

Tables and Figures
Table 1. Behavioral AoM metrics across models
Table 1. Behavioral AoM metrics for GPT‑2 and Qwen2.5 on hardened suites: DISAMB (52 pairs), CF shift-direction on shift items (N=60), and COH main/ablation accuracies (80 items; 240 contexts including controls). Brackets denote 95% bootstrap confidence intervals. [E2a, E1, C6, C7]
Model	AoM composite	DISAMB acc (CI)	CF shift-dir acc (CI)	COH main acc (CI)	COH ablate-irrelevant (CI)	COH ablate-relevant (CI)
openai-community/gpt2 (124M)	0.869	0.856 [0.788, 0.914]	0.850 [0.750, 0.933]	0.900 [0.838, 0.963]	0.888 [0.812, 0.950]	0.588 [0.487, 0.700]
Qwen/Qwen2.5-0.5B	0.849	0.913 [0.856, 0.962]	0.733 [0.617, 0.833]	0.900 [0.838, 0.963]	0.888 [0.825, 0.950]	0.362 [0.263, 0.475]
Qwen/Qwen2.5-1.5B	0.869	0.933 [0.885, 0.971]	0.800 [0.700, 0.883]	0.875 [0.800, 0.938]	0.850 [0.775, 0.925]	0.350 [0.263, 0.450]
Qwen/Qwen2.5-3B	0.903	0.933 [0.885, 0.981]	1.000 [1.000, 1.000]	0.775 [0.688, 0.863]	0.825 [0.750, 0.900]	0.375 [0.275, 0.487]
Keyword baseline (DISAMB suite): 0.615 [0.558, 0.673] (pair-bootstrap CI aligned to DISAMB bootstrap unit). [E2a, E1, C5]

Table 2. CPT-style activation patching aggregates
Table 2. CPT context-swap patching aggregates on DISAMB for GPT‑2 and Qwen2.5 (104 donor→receiver directions per model; zero skips). “Mean max effect” averages max_ℓ(effect_ℓ) within each direction’s layer sweep and is therefore reported as a descriptive post-selection summary; `mean_argmax_layer` is the mean of the per-direction maximizing layers, not a single shared optimum; sham is receiver→receiver no-op. [E3, E4a, C3]
Model	Patched / total (skipped)	Mean max effect (CI)	Flip rate @ per-direction argmax layer (CI)	Mean argmax layer (per-direction mean)	Sham mean max effect (CI)
GPT‑2 (124M; 12 blocks)	104/104 (0)	0.395 [0.301, 0.501]	0.000 [0.000, 0.000]	5.47	2.43×10⁻⁶ [1.23×10⁻⁶, 3.76×10⁻⁶]
Qwen2.5‑0.5B (24 blocks)	104/104 (0)	1.651 [1.241, 2.128]	0.058 [0.019, 0.106]	8.19	7.59×10⁻⁶ [6.13×10⁻⁶, 9.09×10⁻⁶]
Qwen2.5‑1.5B (28 blocks)	104/104 (0)	1.807 [1.461, 2.230]	0.115 [0.058, 0.183]	7.72	5.67×10⁻⁶ [3.69×10⁻⁶, 7.87×10⁻⁶]
Qwen2.5‑3B (36 blocks)	104/104 (0)	1.686 [1.304, 2.117]	0.048 [0.010, 0.087]	13.53	1.61×10⁻⁵ [1.42×10⁻⁵, 1.79×10⁻⁵]

Figure 1. CPT layerwise donor-directed effect profile summary
Figure 1. Mean donor-directed CPT effect by relative depth for the four AoM+CPT sweep models. Curves show early-to-mid peaks and near-zero sham baselines across checkpoints. [E3, E4a, C3]
Figure 2. Relevance-controlled contrasts across components
Figure 2. Relevance-controlled contrasts: CF shift vs substitution controls (left) and COH constraint vs irrelevant spans (right). CF shows shift>control in 3/4 checkpoints; COH separation is robust across all four (see Table 4). [E10e, E10c, E10d, C9, C10]

<!-- BEGIN GENERATED TABLE: sdh_specificity -->
Table 3. SDH target-minus-nearby-matched-span deltas (fixed layer)
Table 3. SDH fixed-depth target-minus-nearby-matched-span results at ℓ* = round(0.25 × (L−1)) under the canonical `matched_token` control, with `position_window=8` and exclusion buffers ±2. Reported effects are signed donor-directed margin shifts; Δ = E_target − E_ctrl and win rate = P(Δ_i > 0). N directions per model = 104. This table reports one control-selection path and should not be read as a strategy-comparison table. [E5a, E5b, E5c, E5d, E5e, E5f, C4]
| Model | Layers | ℓ* | E_target (CI) | E_ctrl, matched span (CI) | Δ = E_target − E_ctrl (CI) | Win rate (CI) |
|---|---:|---:|---|---|---|---|
| GPT‑2 (124M) | 12 | 3 | 0.09 [-0.05, 0.23] | 0.38 [0.13, 0.72] | -0.29 [-0.66, 0.00] | 0.52 [0.42, 0.62] |
| Qwen2.5‑0.5B | 24 | 6 | 0.73 [0.39, 1.11] | 0.07 [0.00, 0.17] | +0.66 [0.28, 1.06] | 0.64 [0.55, 0.73] |
| Qwen2.5‑1.5B | 28 | 7 | 0.62 [0.29, 0.93] | 0.08 [-0.05, 0.23] | +0.54 [0.20, 0.86] | 0.66 [0.58, 0.75] |
| Qwen2.5‑3B | 36 | 9 | 1.06 [0.71, 1.49] | 0.09 [-0.01, 0.20] | +0.97 [0.57, 1.41] | 0.73 [0.64, 0.82] |
| Qwen3‑4B | 36 | 9 | 0.87 [0.52, 1.20] | -0.05 [-0.15, 0.04] | +0.92 [0.58, 1.24] | 0.79 [0.71, 0.87] |
| Llama‑3.2‑1B | 16 | 4 | 0.38 [0.13, 0.65] | 0.12 [0.02, 0.24] | +0.26 [-0.03, 0.55] | 0.61 [0.52, 0.69] |
| Llama‑3.2‑3B | 28 | 7 | 0.21 [0.02, 0.41] | 0.00 [-0.06, 0.05] | +0.21 [0.00, 0.43] | 0.69 [0.61, 0.78] |
| Llama‑3.1‑8B | 32 | 8 | 0.31 [0.14, 0.47] | -0.02 [-0.09, 0.04] | +0.33 [0.18, 0.49] | 0.72 [0.63, 0.81] |
<!-- END GENERATED TABLE: sdh_specificity -->
Table 4. COH and CF causal patching contrasts (merged)
Table 4. Merged relevance-controlled contrasts for COH pseudo-ablations (constraint vs irrelevant) and CF intervention-span patching (shift vs substitution controls). COH uses N=160 total (80 per contrast); CF uses N=80 total (60 shift, 20 controls), with zero skips. All rows are sham-controlled with near-zero sham baselines; brackets denote 95% bootstrap confidence intervals. [E10c, E10d, E10e, C9, C10]

| Experiment | Model | Target effect (CI) | Control effect (CI) | Δ | Cohen’s d | Sham |
|---|---|---|---|---:|---:|---:|
| COH | GPT‑2 | 1.094 [0.970, 1.219] | 0.411 [0.305, 0.515] | 0.683 | 1.266 | 0.000 |
| COH | Qwen2.5‑0.5B | 1.687 [1.452, 1.945] | 0.332 [0.255, 0.428] | 1.355 | 1.526 | 0.009 |
| COH | Qwen2.5‑1.5B | 1.838 [1.595, 2.133] | 0.463 [0.354, 0.593] | 1.375 | 1.368 | 0.019 |
| COH | Qwen2.5‑3B | 1.403 [1.195, 1.627] | 0.336 [0.276, 0.400] | 1.067 | 1.377 | 0.010 |
| CF | GPT‑2 | 0.701 [0.574, 0.850] | 0.133 [0.031, 0.260] | 0.568 | 1.179 | 0.000 |
| CF | Qwen2.5‑0.5B | 1.478 [1.072, 1.911] | 0.402 [0.123, 0.873] | 1.076 | 0.677 | 0.000 |
| CF | Qwen2.5‑1.5B | 0.491 [0.370, 0.632] | 0.496 [0.224, 0.797] | -0.005 | -0.009 | 0.000 |
| CF | Qwen2.5‑3B | 1.515 [1.151, 1.929] | 0.356 [0.070, 0.777] | 1.159 | 0.798 | 0.000 |
