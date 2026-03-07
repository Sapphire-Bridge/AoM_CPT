from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def _safe_get_commit_hash(obj) -> str | None:
    try:
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            v = getattr(cfg, "_commit_hash", None)
            if v:
                return str(v)
        v2 = getattr(obj, "_commit_hash", None)
        return str(v2) if v2 else None
    except Exception:
        return None


@dataclass(frozen=True)
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    architecture: str
    model_commit_hash: Optional[str] = None
    tokenizer_revision_effective: Optional[str] = None


def load_causal_lm(
    model_name_or_path: str,
    device: torch.device,
    *,
    torch_dtype: Optional[str] = None,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager",
    device_map: Optional[str] = None,
) -> LoadedModel:
    dtype = None
    if torch_dtype is not None:
        dtype = getattr(torch, torch_dtype)
    elif "qwen" in model_name_or_path.lower():
        dtype = torch.bfloat16

    tokenizer_kwargs: dict = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "use_fast": True,
    }
    tok_rev = tokenizer_revision if tokenizer_revision is not None else revision
    if tok_rev is not None:
        tokenizer_kwargs["revision"] = str(tok_rev)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if revision is not None:
        model_kwargs["revision"] = str(revision)
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    def _load_model(kwargs: dict) -> PreTrainedModel:
        try:
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        except ImportError as e:
            if device_map is not None:
                raise ImportError(
                    "Loading with `device_map` requires the `accelerate` package. "
                    "Install it (e.g. `pip install accelerate`) or rerun without `device_map`."
                ) from e
            raise

    try:
        model = _load_model(model_kwargs)
    except TypeError:
        # Some architectures/configs may not accept attn_implementation; retry without it.
        model_kwargs.pop("attn_implementation", None)
        model = _load_model(model_kwargs)

    if device_map is None:
        model.to(device)
    model.eval()

    from aom.interventions.activation_patching import detect_architecture

    arch = detect_architecture(model)

    model_commit_hash = _safe_get_commit_hash(model)

    tokenizer_revision_effective = None
    try:
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            tokenizer_revision_effective = init_kwargs.get("revision", None)
    except Exception:
        tokenizer_revision_effective = None

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        architecture=arch,
        model_commit_hash=model_commit_hash,
        tokenizer_revision_effective=tokenizer_revision_effective,
    )
