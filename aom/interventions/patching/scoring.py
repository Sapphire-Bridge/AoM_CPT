from __future__ import annotations

import math
from typing import Dict, List, Literal, Mapping

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.interventions.activation_patching import PatchSpanSite, forward_with_patched_block_output_span
from aom.utils import get_logprob_computation_config, logprob_of_continuation_ids


LabelAggregation = Literal["logmeanexp", "mean"]


def _encode(tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(str(text), return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def _logmeanexp(xs: List[float]) -> float:
    if len(xs) < 1:
        raise ValueError("logmeanexp requires at least one value")
    t = torch.tensor(list(xs), dtype=torch.float64)
    return float(torch.logsumexp(t, dim=0) - math.log(len(xs)))


def _aggregate(vals: List[float], *, agg: LabelAggregation) -> float:
    if len(vals) < 1:
        raise ValueError("aggregation requires at least one value")
    if agg == "logmeanexp":
        return float(_logmeanexp(vals))
    if agg == "mean":
        return float(sum(float(x) for x in vals) / max(1, len(vals)))
    raise ValueError(f"Unknown label aggregation: {agg!r}")


@torch.no_grad()
def score_labels_next_continuations_ids(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    choices: Mapping[str, List[str]],
    device: torch.device,
    normalize_by_length: bool,
    label_aggregation: LabelAggregation,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        vals: List[float] = []
        for cont in continuations:
            cont_ids = _encode(tokenizer, str(cont), device=device)
            lp = logprob_of_continuation_ids(
                model,
                prompt_ids=prompt_ids,
                continuation_ids=cont_ids,
                normalize_by_length=bool(normalize_by_length),
            )
            vals.append(float(lp.item()))
        scores[str(label)] = float(_aggregate(vals, agg=label_aggregation))
    return scores


@torch.no_grad()
def score_labels_next_continuations_patched_ids(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    choices: Mapping[str, List[str]],
    device: torch.device,
    patch_site: PatchSpanSite,
    replacement: torch.Tensor,
    normalize_by_length: bool,
    label_aggregation: LabelAggregation,
) -> Dict[str, float]:
    """
    Score choices under a patched forward pass (block output patched at `patch_site`).

    This re-runs the patched model per continuation for clarity and reproducibility.
    """
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        vals: List[float] = []
        for cont in continuations:
            cont_ids = _encode(tokenizer, str(cont), device=device)
            full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
            logits = forward_with_patched_block_output_span(
                model,
                input_ids=full_ids,
                site=patch_site,
                replacement=replacement,
            )

            P = prompt_ids.size(1)
            C = cont_ids.size(1)
            logits_slice = logits[:, P - 1 : P + C - 1, :].to(dtype=logprobs_dtype)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)  # (1, C)
            if not torch.isfinite(gathered).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite log-probability detected during patched scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            lp = gathered.mean(dim=1) if bool(normalize_by_length) else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = float(_aggregate(vals, agg=label_aggregation))
    return scores

