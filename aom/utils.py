from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .stats import MetricValue, bootstrap_ci, bootstrap_ci_metric

_LOGPROBS_DTYPE: torch.dtype = torch.float32
_STRICT_FINITE: bool = True


def configure_logprob_computation(*, logprobs_dtype: torch.dtype, strict_finite: bool) -> None:
    """
    Configure how log-probabilities are computed across the evaluator.

    Default behavior is set at import time (float32 log-softmax + strict finite checks).
    The CLI wires this up via `aom_eval.py`.
    """
    global _LOGPROBS_DTYPE, _STRICT_FINITE
    _LOGPROBS_DTYPE = logprobs_dtype
    _STRICT_FINITE = bool(strict_finite)


def get_logprob_computation_config() -> Tuple[torch.dtype, bool]:
    return _LOGPROBS_DTYPE, _STRICT_FINITE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(rows: Iterable[Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            if is_dataclass(row):
                row = asdict(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def logprob_of_continuation_ids(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    normalize_by_length: bool = False,
) -> torch.Tensor:
    """
    Compute log P(continuation | prompt) for a batch, using already-tokenized IDs.

    Shapes:
      prompt_ids: (B, P)
      continuation_ids: (B, C)
    Returns:
      logp: (B,) (sum or mean per token)
    """
    if prompt_ids.ndim != 2 or continuation_ids.ndim != 2:
        raise ValueError("prompt_ids and continuation_ids must be rank-2 tensors")
    if prompt_ids.size(0) != continuation_ids.size(0):
        raise ValueError("prompt_ids and continuation_ids must have same batch size")
    if continuation_ids.size(1) < 1:
        raise ValueError("continuation_ids must have length >= 1")

    full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    if attention_mask is None:
        attention_mask = torch.ones_like(full_ids)
    else:
        if attention_mask.shape != full_ids.shape:
            raise ValueError("attention_mask must match full_ids shape")

    out = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits  # type: ignore[attr-defined]

    # continuation tokens live at positions [P, P+C-1]; each is predicted by logits at [P-1, P+C-2].
    P = prompt_ids.size(1)
    C = continuation_ids.size(1)
    logits_slice = logits[:, P - 1 : P + C - 1, :].to(dtype=_LOGPROBS_DTYPE)
    log_probs = F.log_softmax(logits_slice, dim=-1)
    gathered = log_probs.gather(2, continuation_ids.unsqueeze(-1)).squeeze(-1)  # (B, C)
    if not torch.isfinite(gathered).all():
        if _STRICT_FINITE:
            raise FloatingPointError(
                "Non-finite log-probability detected in continuation scoring "
                f"(device={gathered.device}, logits_dtype={logits.dtype}, logprobs_dtype={gathered.dtype})."
            )
        gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
    if normalize_by_length:
        return gathered.mean(dim=1)
    return gathered.sum(dim=1)


def batched(iterable: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]
