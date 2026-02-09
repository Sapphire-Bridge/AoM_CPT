from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import torch
import transformers

from aom.interventions.activation_patching import get_num_layers
from aom.mechanistic.attention_recorder import AttentionPatternRecorder
from aom.mechanistic.synthetic import build_allowed_ids, generate_induction_batch, permute_first_half
from aom.models.loader import load_causal_lm

from .base import BackendLoadResult, BackendRunResult, InductionCollector, InductionBackend


def _max_position_embeddings(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return None


@dataclass(frozen=True)
class HFEagerBackend(InductionBackend):
    name: str = "hf"

    def load(
        self,
        *,
        model_name_or_path: str,
        device: torch.device,
        torch_dtype: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
        attn_implementation: str,
    ) -> BackendLoadResult:
        loaded = load_causal_lm(
            model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
            attn_implementation=attn_implementation,  # recorder forces eager if needed
            device_map=None,
        )
        model = loaded.model
        tokenizer = loaded.tokenizer

        exclude_ids: Set[int] = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if isinstance(pad_id, int):
            exclude_ids.add(int(pad_id))

        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None or int(vocab_size) < 1:
            raise ValueError("could not determine vocab_size for synthetic generation")

        n_layers = int(get_num_layers(model))
        max_seq_len = _max_position_embeddings(model)

        param0 = next(model.parameters(), None)
        model_param_dtype = str(param0.dtype) if param0 is not None else ""
        model_param_device = str(param0.device) if param0 is not None else ""

        num_attention_heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
        num_key_value_heads = getattr(getattr(model, "config", None), "num_key_value_heads", None)

        return BackendLoadResult(
            model=model,
            tokenizer=tokenizer,
            architecture=str(loaded.architecture),
            device=device,
            n_layers=n_layers,
            vocab_size=int(vocab_size),
            exclude_token_ids=frozenset(exclude_ids),
            max_seq_len=max_seq_len,
            model_param_dtype=model_param_dtype,
            model_param_device=model_param_device,
            num_attention_heads=int(num_attention_heads) if isinstance(num_attention_heads, int) else None,
            num_key_value_heads=int(num_key_value_heads) if isinstance(num_key_value_heads, int) else None,
            backend_version=str(transformers.__version__),
        )

    def run_batches(
        self,
        *,
        loaded: BackendLoadResult,
        layers: List[int],
        base_len: int,
        repeats: int,
        batch_size: int,
        n_batches: int,
        seed: int,
        baseline: str,
        debug_store_attn: bool,
        validate_attn: str,
        tl_crosscheck: bool,
        bootstrap_n: int,
        bootstrap_seed: int,
        ci: float,
    ) -> BackendRunResult:
        model = loaded.model
        device = loaded.device
        _ = bool(tl_crosscheck)
        _ = int(bootstrap_n)
        _ = int(bootstrap_seed)
        _ = float(ci)

        allowed_ids = build_allowed_ids(int(loaded.vocab_size), set(loaded.exclude_token_ids))
        recorder = AttentionPatternRecorder(
            model,
            layers=layers,
            base_len=int(base_len),
            repeats=int(repeats),
            baseline_mode=str(baseline),
            debug_store_attn=bool(debug_store_attn),
            validate_attn=str(validate_attn),
        )
        model.eval()

        with torch.no_grad(), recorder:
            for batch_idx in range(int(n_batches)):
                batch_seed = int(seed) + int(batch_idx)
                base_ids = generate_induction_batch(
                    vocab_size=int(loaded.vocab_size),
                    base_len=int(base_len),
                    repeats=int(repeats),
                    batch_size=int(batch_size),
                    seed=int(batch_seed),
                    allowed_ids=allowed_ids,
                )
                input_ids = base_ids.to(device)
                attention_mask = torch.ones_like(input_ids)

                recorder.set_mode("repeat")
                _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                if str(baseline) == "shuffle":
                    permuted = permute_first_half(
                        base_ids, base_len=int(base_len), seed=int(batch_seed), resample_identity=True
                    )
                    permuted = permuted.to(device)
                    recorder.set_mode("control")
                    _ = model(input_ids=permuted, attention_mask=attention_mask, use_cache=False)

        return BackendRunResult(collector=recorder, extras={})
