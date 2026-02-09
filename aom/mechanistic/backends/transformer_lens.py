from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from aom.mechanistic.induction import (
    bootstrap_ci_matrix,
    calculate_first_half_mass,
    calculate_induction_diag_fraction,
    calculate_induction_score,
    validate_attention_weights,
)
from aom.mechanistic.synthetic import build_allowed_ids, generate_induction_batch, permute_first_half

from .base import BackendLoadResult, BackendRunResult, InductionBackend


def _as_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    return None


def _infer_architecture(model_name_or_path: str, cfg: Any) -> str:
    candidates: List[str] = []
    candidates.append(str(model_name_or_path))
    for attr in ("model_family", "model_name", "architecture", "original_architecture"):
        v = getattr(cfg, attr, None)
        if isinstance(v, str) and v:
            candidates.append(v)
    joined = " ".join(candidates).lower()
    if "qwen3" in joined:
        return "qwen3"
    if "qwen" in joined:
        return "qwen2"
    if "llama" in joined:
        return "llama"
    if "mistral" in joined:
        return "mistral"
    if "neox" in joined or "gpt_neox" in joined:
        return "gpt_neox"
    if "gpt2" in joined or "gpt-2" in joined:
        return "gpt2"
    return "transformer_lens"


class _TLCollector:
    def __init__(
        self,
        *,
        layers: List[int],
        base_len: int,
        repeats: int,
        baseline_mode: str,
        debug_store_attn: bool,
        validate_attn: str,
        tl_crosscheck: bool,
        tl_head_detector: Any,
    ) -> None:
        self.layers = [int(l) for l in layers]
        self.base_len = int(base_len)
        self.repeats = int(repeats)
        self.seq_len = int(self.base_len * self.repeats)
        if self.repeats != 2:
            raise ValueError("repeats must be 2 for V1 induction metric")
        if baseline_mode not in {"shuffle", "offset0"}:
            raise ValueError(f"invalid baseline_mode={baseline_mode!r}")
        self.baseline_mode = str(baseline_mode)
        self.debug_store_attn = bool(debug_store_attn)
        if validate_attn not in {"none", "first", "always"}:
            raise ValueError(f"invalid validate_attn={validate_attn!r}")
        self.validate_attn = str(validate_attn)
        self._validated_layers: set[int] = set()
        self.mode = "repeat"

        self._samples: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "repeat": {int(l): [] for l in self.layers},
            "control": {int(l): [] for l in self.layers},
        }
        self._mass_samples: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "repeat": {int(l): [] for l in self.layers},
            "control": {int(l): [] for l in self.layers},
        }
        self._fraction_samples: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "repeat": {int(l): [] for l in self.layers},
            "control": {int(l): [] for l in self.layers},
        }
        self._debug_attn: Dict[int, Optional[torch.Tensor]] = {int(l): None for l in self.layers}

        self.tl_crosscheck = bool(tl_crosscheck)
        self._tl_head_detector = tl_head_detector
        self._tl_detection_pattern: Optional[torch.Tensor] = None
        self._tl_abs_samples: Dict[int, List[torch.Tensor]] = {int(l): [] for l in self.layers}
        self._tl_mul_samples: Dict[int, List[torch.Tensor]] = {int(l): [] for l in self.layers}

    def set_mode(self, mode: str) -> None:
        if mode not in {"repeat", "control"}:
            raise ValueError(f"invalid mode={mode!r}")
        self.mode = mode

    def set_tl_detection_pattern(self, pattern: Optional[torch.Tensor]) -> None:
        self._tl_detection_pattern = pattern

    def _maybe_validate(self, layer: int, attn_weights: torch.Tensor) -> None:
        if self.validate_attn == "none":
            return
        if self.validate_attn == "always" or int(layer) not in self._validated_layers:
            validate_attention_weights(attn_weights, self.seq_len)
            if self.validate_attn == "first":
                self._validated_layers.add(int(layer))

    def _capture(self, layer: int, attn_weights: torch.Tensor) -> None:
        if not isinstance(attn_weights, torch.Tensor):
            raise ValueError("attention weights must be a torch.Tensor")

        self._maybe_validate(int(layer), attn_weights)
        if self.debug_store_attn:
            self._debug_attn[int(layer)] = attn_weights.detach().float().cpu()

        scores = calculate_induction_score(attn_weights, self.base_len).detach().float().cpu()
        mass = calculate_first_half_mass(attn_weights, self.base_len).detach().float().cpu()
        frac = calculate_induction_diag_fraction(attn_weights, self.base_len).detach().float().cpu()
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        if mass.ndim == 1:
            mass = mass.unsqueeze(0)
        if frac.ndim == 1:
            frac = frac.unsqueeze(0)

        if self.baseline_mode == "offset0":
            from aom.mechanistic.induction import calculate_induction_control

            baseline = calculate_induction_control(attn_weights, self.base_len, mode="offset0").detach().float().cpu()
            if baseline.ndim == 1:
                baseline = baseline.unsqueeze(0)
            self._samples["repeat"][int(layer)].append(scores)
            self._samples["control"][int(layer)].append(baseline)
            self._mass_samples["repeat"][int(layer)].append(mass)
            self._mass_samples["control"][int(layer)].append(mass)
            self._fraction_samples["repeat"][int(layer)].append(frac)
            self._fraction_samples["control"][int(layer)].append(frac)
        else:
            self._samples[self.mode][int(layer)].append(scores)
            self._mass_samples[self.mode][int(layer)].append(mass)
            self._fraction_samples[self.mode][int(layer)].append(frac)

        if self.tl_crosscheck and self.mode == "repeat":
            self._capture_tl_crosscheck(int(layer), attn_weights)

    def _capture_tl_crosscheck(self, layer: int, attn_weights: torch.Tensor) -> None:
        if self._tl_head_detector is None:
            return
        detection = self._tl_detection_pattern
        if detection is None:
            return
        try:
            detection = detection.to(device=attn_weights.device)
            if detection.dtype != attn_weights.dtype:
                detection = detection.to(dtype=attn_weights.dtype)
            abs_score = self._tl_head_detector.compute_head_attention_similarity_score(
                attn_weights, detection, error_measure="abs"
            )
            mul_score = self._tl_head_detector.compute_head_attention_similarity_score(
                attn_weights, detection, error_measure="mul"
            )
            abs_tensor = self._as_sample_matrix(abs_score).detach().float().cpu()
            mul_tensor = self._as_sample_matrix(mul_score).detach().float().cpu()
            self._tl_abs_samples[int(layer)].append(abs_tensor)
            self._tl_mul_samples[int(layer)].append(mul_tensor)
        except Exception:
            return

    @staticmethod
    def _as_sample_matrix(score: Any) -> torch.Tensor:
        if isinstance(score, torch.Tensor):
            if score.ndim == 1:
                return score.unsqueeze(0)
            if score.ndim == 2:
                return score
            if score.ndim > 2:
                return score.reshape(-1, score.shape[-1])
        raise ValueError("unexpected TransformerLens similarity score shape")

    def get_samples(self, layer: int, mode: str) -> List[torch.Tensor]:
        return list(self._samples[str(mode)][int(layer)])

    def get_mass_samples(self, layer: int, mode: str) -> List[torch.Tensor]:
        return list(self._mass_samples[str(mode)][int(layer)])

    def get_fraction_samples(self, layer: int, mode: str) -> List[torch.Tensor]:
        return list(self._fraction_samples[str(mode)][int(layer)])

    def finalize_extras(
        self, *, layers: List[int], bootstrap_n: int, bootstrap_seed: int, ci: float
    ) -> Dict[int, Dict[int, Dict[str, float]]]:
        if not self.tl_crosscheck:
            return {}
        extras: Dict[int, Dict[int, Dict[str, float]]] = {}
        for layer in layers:
            abs_mat = _concat_samples(self._tl_abs_samples.get(int(layer), []))
            mul_mat = _concat_samples(self._tl_mul_samples.get(int(layer), []))
            if abs_mat.numel() == 0 or mul_mat.numel() == 0:
                continue
            abs_mean, abs_lo, abs_hi = bootstrap_ci_matrix(
                abs_mat.numpy(), n_bootstrap=int(bootstrap_n), ci=float(ci), seed=int(bootstrap_seed)
            )
            mul_mean, mul_lo, mul_hi = bootstrap_ci_matrix(
                mul_mat.numpy(), n_bootstrap=int(bootstrap_n), ci=float(ci), seed=int(bootstrap_seed)
            )
            n_heads = int(abs_mat.shape[1])
            for h in range(n_heads):
                extras.setdefault(int(layer), {}).setdefault(int(h), {}).update(
                    {
                        "tl_induction_score_abs": float(abs_mean[h]),
                        "tl_induction_score_abs_ci_low": float(abs_lo[h]),
                        "tl_induction_score_abs_ci_high": float(abs_hi[h]),
                        "tl_induction_score_mul": float(mul_mean[h]),
                        "tl_induction_score_mul_ci_low": float(mul_lo[h]),
                        "tl_induction_score_mul_ci_high": float(mul_hi[h]),
                    }
                )
        return extras


def _concat_samples(samples: List[torch.Tensor]) -> torch.Tensor:
    if not samples:
        return torch.empty((0, 0), dtype=torch.float32)
    normed = [s.unsqueeze(0) if s.ndim == 1 else s for s in samples]
    return torch.cat(normed, dim=0)


def _resolve_pattern_hook_name(model: Any, layer: int) -> str:
    preferred = f"blocks.{int(layer)}.attn.hook_pattern"
    hook_dict = getattr(model, "hook_dict", None)
    if isinstance(hook_dict, dict) and preferred in hook_dict:
        return preferred
    if isinstance(hook_dict, dict):
        candidates = [
            k for k in hook_dict.keys() if k.endswith(".attn.hook_pattern") and f"blocks.{int(layer)}." in k
        ]
        if len(candidates) == 1:
            return str(candidates[0])
    return preferred


@dataclass(frozen=True)
class TransformerLensBackend(InductionBackend):
    name: str = "transformer_lens"

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
        _ = str(attn_implementation)
        try:
            import transformer_lens  # type: ignore[import-not-found]
            from transformer_lens import HookedTransformer  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError(
                "TransformerLens backend requires `transformer_lens` installed. "
                "Install it with `pip install transformer-lens`."
            ) from e

        dtype = getattr(torch, torch_dtype) if torch_dtype is not None else None
        hf_model_kwargs: Dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "trust_remote_code": bool(trust_remote_code),
        }
        hf_tokenizer_kwargs: Dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "trust_remote_code": bool(trust_remote_code),
            "use_fast": True,
        }
        if dtype is not None:
            hf_model_kwargs["torch_dtype"] = dtype

        model = _load_hooked_transformer(
            HookedTransformer,
            model_name_or_path=model_name_or_path,
            device=device,
            hf_model_kwargs=hf_model_kwargs,
            hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        )
        model.eval()

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("TransformerLens model did not expose a tokenizer (model.tokenizer is None)")

        exclude_ids: Set[int] = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if isinstance(pad_id, int):
            exclude_ids.add(int(pad_id))

        cfg = getattr(model, "cfg", None)
        n_layers = _as_int(getattr(cfg, "n_layers", None))
        if n_layers is None or n_layers < 1:
            raise RuntimeError("could not determine n_layers from TransformerLens model.cfg")
        vocab_size = _as_int(getattr(cfg, "d_vocab", None))
        if vocab_size is None:
            vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None or int(vocab_size) < 1:
            raise ValueError("could not determine vocab_size for synthetic generation")

        max_seq_len = _as_int(getattr(cfg, "n_ctx", None))
        architecture = _infer_architecture(model_name_or_path, cfg)

        param0 = next(model.parameters(), None)
        model_param_dtype = str(param0.dtype) if param0 is not None else ""
        model_param_device = str(param0.device) if param0 is not None else ""

        num_attention_heads = _as_int(getattr(cfg, "n_heads", None))
        num_key_value_heads = _as_int(getattr(cfg, "n_key_value_heads", None))

        return BackendLoadResult(
            model=model,
            tokenizer=tokenizer,
            architecture=str(architecture),
            device=device,
            n_layers=int(n_layers),
            vocab_size=int(vocab_size),
            exclude_token_ids=frozenset(exclude_ids),
            max_seq_len=int(max_seq_len) if isinstance(max_seq_len, int) else None,
            model_param_dtype=model_param_dtype,
            model_param_device=model_param_device,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            backend_version=str(getattr(transformer_lens, "__version__", "")),
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

        tl_head_detector = None
        if bool(tl_crosscheck):
            tl_head_detector = _try_import_tl_head_detector()

        collector = _TLCollector(
            layers=layers,
            base_len=int(base_len),
            repeats=int(repeats),
            baseline_mode=str(baseline),
            debug_store_attn=bool(debug_store_attn),
            validate_attn=str(validate_attn),
            tl_crosscheck=bool(tl_crosscheck) and tl_head_detector is not None,
            tl_head_detector=tl_head_detector,
        )

        allowed_ids = build_allowed_ids(int(loaded.vocab_size), set(loaded.exclude_token_ids))
        hooks = _build_tl_hooks(model, collector, layers)

        with torch.no_grad():
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
                tokens = base_ids.to(device)

                collector.set_mode("repeat")
                if collector.tl_crosscheck:
                    collector.set_tl_detection_pattern(_make_tl_detection_pattern(tl_head_detector, tokens))
                _run_tl_forward(model, tokens, hooks)

                if str(baseline) == "shuffle":
                    permuted = permute_first_half(base_ids, base_len=int(base_len), seed=int(batch_seed), resample_identity=True)
                    permuted = permuted.to(device)
                    collector.set_mode("control")
                    collector.set_tl_detection_pattern(None)
                    _run_tl_forward(model, permuted, hooks)

        extras = collector.finalize_extras(
            layers=layers, bootstrap_n=int(bootstrap_n), bootstrap_seed=int(bootstrap_seed), ci=float(ci)
        )
        return BackendRunResult(collector=collector, extras=extras)


def _load_hooked_transformer(
    hooked_transformer_cls: Any,
    *,
    model_name_or_path: str,
    device: torch.device,
    hf_model_kwargs: Dict[str, Any],
    hf_tokenizer_kwargs: Dict[str, Any],
) -> Any:
    import inspect

    device_str = str(device)
    sig = inspect.signature(hooked_transformer_cls.from_pretrained)
    kwargs: Dict[str, Any] = {}
    if "device" in sig.parameters:
        kwargs["device"] = device_str

    model_kwargs = dict(hf_model_kwargs)
    tokenizer_kwargs = dict(hf_tokenizer_kwargs)

    if "hf_model_kwargs" in sig.parameters:
        kwargs["hf_model_kwargs"] = model_kwargs
    elif "model_kwargs" in sig.parameters:
        kwargs["model_kwargs"] = model_kwargs

    if "hf_tokenizer_kwargs" in sig.parameters:
        kwargs["hf_tokenizer_kwargs"] = tokenizer_kwargs
    elif "tokenizer_kwargs" in sig.parameters:
        kwargs["tokenizer_kwargs"] = tokenizer_kwargs

    try:
        return hooked_transformer_cls.from_pretrained(model_name_or_path, **kwargs)
    except TypeError as e:
        raise RuntimeError(
            "Could not call `HookedTransformer.from_pretrained` with the current "
            "TransformerLens version. Try upgrading `transformer-lens`."
        ) from e


def _build_tl_hooks(model: Any, collector: _TLCollector, layers: List[int]) -> List[Tuple[str, Callable]]:
    hooks: List[Tuple[str, Callable]] = []
    hook_dict = getattr(model, "hook_dict", None)
    for layer in layers:
        hook_name = _resolve_pattern_hook_name(model, int(layer))
        if isinstance(hook_dict, dict) and hook_name not in hook_dict:
            similar = sorted([k for k in hook_dict.keys() if f"blocks.{int(layer)}." in k and ".attn." in k])[:25]
            hint = f" Similar hooks for this layer: {similar!r}" if similar else ""
            raise RuntimeError(
                f"Could not find TransformerLens hook {hook_name!r} for layer={int(layer)}.{hint} "
                "Try `--backend hf` or upgrade `transformer-lens`."
            )

        def _make_hook(layer_idx: int):
            def hook_fn(pattern: torch.Tensor, hook) -> torch.Tensor:
                collector._capture(int(layer_idx), pattern)
                return pattern

            return hook_fn

        hooks.append((hook_name, _make_hook(int(layer))))
    return hooks


def _run_tl_forward(model: Any, tokens: torch.Tensor, hooks: List[Tuple[str, Callable]]) -> None:
    run_with_hooks = getattr(model, "run_with_hooks", None)
    if callable(run_with_hooks):
        _ = run_with_hooks(tokens, return_type="logits", fwd_hooks=list(hooks))
        return
    raise RuntimeError(
        "TransformerLens backend expected a HookedTransformer with `run_with_hooks`. "
        "Got an object without it; try upgrading `transformer-lens`."
    )


def _try_import_tl_head_detector() -> Any:
    try:
        from transformer_lens import head_detector  # type: ignore[import-not-found]

        return head_detector
    except Exception:
        return None


def _make_tl_detection_pattern(head_detector: Any, tokens: torch.Tensor) -> Optional[torch.Tensor]:
    if head_detector is None:
        return None
    get_pattern = getattr(head_detector, "get_induction_head_detection_pattern", None)
    if not callable(get_pattern):
        return None
    try:
        return get_pattern(tokens.detach().to("cpu"))
    except Exception:
        return None
