from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.data.schemas import DisambPair
from aom.interventions.activation_patching import PatchSpanSite, get_block_outputs
from aom.interventions.sae_adapter import SAEInputTransform, SAEProtocol, SAEPatchConfig
from aom.interventions.sae_patching import forward_with_sae_feature_patching_span
from aom.metrics.disamb import LabelScores, _encode_prompt, _logmeanexp, _margin, score_labels_next_continuations
from aom.token_spans import token_span_for_substring
from aom.utils import bootstrap_ci, get_logprob_computation_config


@dataclass(frozen=True)
class ReplaceFeaturesAtIndicesPolicy:
    token_indices: List[int]
    replacement_features: torch.Tensor  # (1, span_len, d_sae)

    def apply(self, features: torch.Tensor, *, site_mask: torch.Tensor, token_mask=None) -> torch.Tensor:  # noqa: ARG002
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, seq, d_sae)")
        if features.size(0) != self.replacement_features.size(0):
            raise ValueError("replacement batch size must match features batch size")
        if int(self.replacement_features.size(1)) != len(self.token_indices):
            raise ValueError("replacement span_len must match token_indices length")
        patched = features.clone()
        for i, tok in enumerate(self.token_indices):
            patched[:, int(tok), :] = self.replacement_features[:, i, :]
        return patched


def _infer_sae_device_dtype(sae: SAEProtocol) -> tuple[torch.device, torch.dtype]:
    if isinstance(sae, torch.nn.Module):
        p = next(sae.parameters(), None)
        if p is not None:
            return p.device, p.dtype
    W_dec = getattr(sae, "W_dec", None)
    if isinstance(W_dec, torch.Tensor):
        return W_dec.device, W_dec.dtype
    return torch.device("cpu"), torch.float32


@torch.no_grad()
def score_labels_next_continuations_sae_patched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    choices: Mapping[str, List[str]],
    device: torch.device,
    *,
    layer: int,
    token_indices: List[int],
    sae: SAEProtocol,
    transform: SAEInputTransform,
    policy: ReplaceFeaturesAtIndicesPolicy,
    config: Optional[SAEPatchConfig] = None,
    normalize_by_length: bool = True,
) -> LabelScores:
    prompt_ids = _encode_prompt(tokenizer, prompt, device=device)
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        vals: List[float] = []
        for cont in continuations:
            cont_ids = tokenizer(str(cont), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
            logits = forward_with_sae_feature_patching_span(
                model,
                input_ids=full_ids,
                site=PatchSpanSite(layer=int(layer), token_indices=tuple(int(x) for x in token_indices)),
                sae=sae,
                policy=policy,
                transform=transform,
                config=config,
            )
            P = prompt_ids.size(1)
            C = cont_ids.size(1)
            logits_slice = logits[:, P - 1 : P + C - 1, :].to(dtype=logprobs_dtype)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)  # (1, C)
            if not torch.isfinite(gathered).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite log-probability detected in SAE patched scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            lp = gathered.mean(dim=1) if normalize_by_length else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = _logmeanexp(vals)
    return LabelScores(by_label=scores)


@torch.no_grad()
def compute_sae_cpt_context_swap_patching(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    *,
    sae_by_layer: Mapping[int, SAEProtocol],
    transform_by_layer: Mapping[int, SAEInputTransform],
    layers: Optional[List[int]] = None,
    config: Optional[SAEPatchConfig] = None,
    normalize_by_length: bool = True,
    require_token_id_match: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    if layers is None:
        layers = sorted(int(l) for l in sae_by_layer.keys())
    layers = [int(l) for l in layers]
    for l in layers:
        if l not in sae_by_layer:
            raise ValueError(f"Missing SAE for layer {l}")
        if l not in transform_by_layer:
            raise ValueError(f"Missing transform for layer {l}")

    per_layer_sum = {int(l): 0.0 for l in layers}
    per_layer_n = {int(l): 0 for l in layers}
    sham_per_layer_sum = {int(l): 0.0 for l in layers}
    sham_per_layer_n = {int(l): 0 for l in layers}

    max_effects: List[float] = []
    max_layers: List[int] = []
    max_effects_norm: List[float] = []
    max_flips: List[float] = []
    sham_max_effects: List[float] = []

    n_total_directions = 0
    n_skipped_misaligned = 0

    for it in items:
        for donor, recv in ((it.a, it.b), (it.b, it.a)):
            n_total_directions += 1
            donor_expected = donor.expected_label
            donor_span, donor_token_ids = token_span_for_substring(
                tokenizer, donor.prompt, it.target, it.target_occurrence
            )
            recv_span, recv_token_ids = token_span_for_substring(tokenizer, recv.prompt, it.target, it.target_occurrence)

            if len(donor_span) != len(recv_span) or (require_token_id_match and donor_token_ids != recv_token_ids):
                n_skipped_misaligned += 1
                continue

            base_scores = score_labels_next_continuations(
                model, tokenizer, recv.prompt, it.choices, device, normalize_by_length=normalize_by_length
            )
            base_pred = base_scores.argmax_label()
            base_margin_for_expected = _margin(base_scores, expected=donor_expected)

            donor_ids = _encode_prompt(tokenizer, donor.prompt, device=device)
            recv_ids = _encode_prompt(tokenizer, recv.prompt, device=device)
            donor_out = get_block_outputs(model, donor_ids, layers=layers)
            recv_out = get_block_outputs(model, recv_ids, layers=layers)

            best_for_direction = None
            best_layer_for_direction = None
            best_sham_for_direction = None
            best_flip_for_direction = None
            best_norm_effect_for_direction = None

            for layer in layers:
                sae = sae_by_layer[int(layer)]
                transform = transform_by_layer[int(layer)]
                sae_device, sae_dtype = _infer_sae_device_dtype(sae)

                donor_slice = (
                    donor_out[int(layer)][0, donor_span, :]
                    .detach()
                    .unsqueeze(0)
                    .to(device=sae_device, dtype=sae_dtype)
                )
                recv_slice = (
                    recv_out[int(layer)][0, recv_span, :]
                    .detach()
                    .unsqueeze(0)
                    .to(device=sae_device, dtype=sae_dtype)
                )

                donor_features = sae.encode(transform.forward(donor_slice))
                recv_features = sae.encode(transform.forward(recv_slice))

                policy = ReplaceFeaturesAtIndicesPolicy(token_indices=list(recv_span), replacement_features=donor_features)
                sham_policy = ReplaceFeaturesAtIndicesPolicy(token_indices=list(recv_span), replacement_features=recv_features)

                patched_scores = score_labels_next_continuations_sae_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    layer=int(layer),
                    token_indices=list(recv_span),
                    sae=sae,
                    transform=transform,
                    policy=policy,
                    config=config,
                    normalize_by_length=normalize_by_length,
                )
                sham_scores = score_labels_next_continuations_sae_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    layer=int(layer),
                    token_indices=list(recv_span),
                    sae=sae,
                    transform=transform,
                    policy=sham_policy,
                    config=config,
                    normalize_by_length=normalize_by_length,
                )

                base_margin = base_margin_for_expected
                patched_margin = _margin(patched_scores, expected=donor_expected)
                effect = float(patched_margin - base_margin)
                norm_effect = float(effect / (abs(base_margin) + 1e-8))

                patched_pred = patched_scores.argmax_label()
                flipped = float((base_pred != donor_expected) and (patched_pred == donor_expected))

                sham_margin = _margin(sham_scores, expected=donor_expected)
                sham_effect = float(sham_margin - base_margin)

                per_layer_sum[int(layer)] += effect
                per_layer_n[int(layer)] += 1
                sham_per_layer_sum[int(layer)] += sham_effect
                sham_per_layer_n[int(layer)] += 1

                if best_for_direction is None or effect > best_for_direction:
                    best_for_direction = effect
                    best_layer_for_direction = int(layer)
                    best_flip_for_direction = flipped
                    best_norm_effect_for_direction = norm_effect
                if best_sham_for_direction is None or sham_effect > best_sham_for_direction:
                    best_sham_for_direction = sham_effect

            if best_for_direction is not None and best_layer_for_direction is not None:
                max_effects.append(float(best_for_direction))
                max_layers.append(int(best_layer_for_direction))
                if best_flip_for_direction is not None:
                    max_flips.append(float(best_flip_for_direction))
                if best_norm_effect_for_direction is not None:
                    max_effects_norm.append(float(best_norm_effect_for_direction))
            if best_sham_for_direction is not None:
                sham_max_effects.append(float(best_sham_for_direction))

    mean_max_effect, mean_max_effect_lo, mean_max_effect_hi = bootstrap_ci(
        max_effects, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
    )
    flip_rate, flip_lo, flip_hi = bootstrap_ci(max_flips, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
    mean_norm, mean_norm_lo, mean_norm_hi = bootstrap_ci(
        max_effects_norm, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
    )
    mean_sham, mean_sham_lo, mean_sham_hi = bootstrap_ci(
        sham_max_effects, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
    )

    return {
        "mean_max_effect": mean_max_effect,
        "mean_max_effect_ci_low": mean_max_effect_lo,
        "mean_max_effect_ci_high": mean_max_effect_hi,
        "mean_argmax_layer": float(sum(max_layers) / max(1, len(max_layers))) if max_layers else 0.0,
        "flip_rate_at_best_layer": flip_rate,
        "flip_rate_ci_low": flip_lo,
        "flip_rate_ci_high": flip_hi,
        "mean_norm_max_effect": mean_norm,
        "mean_norm_max_effect_ci_low": mean_norm_lo,
        "mean_norm_max_effect_ci_high": mean_norm_hi,
        "mean_sham_max_effect": mean_sham,
        "mean_sham_max_effect_ci_low": mean_sham_lo,
        "mean_sham_max_effect_ci_high": mean_sham_hi,
        "n_directions_total": int(n_total_directions),
        "n_directions_patched": int(len(max_effects)),
        "n_directions_skipped_misaligned": int(n_skipped_misaligned),
        **{f"effect_layer_{l}": per_layer_sum[l] / max(1, per_layer_n[l]) for l in layers},
        **{f"sham_effect_layer_{l}": sham_per_layer_sum[l] / max(1, sham_per_layer_n[l]) for l in layers},
    }
