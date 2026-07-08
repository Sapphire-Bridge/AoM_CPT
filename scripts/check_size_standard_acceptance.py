#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_counterfactual_pairs_with_manifest
from aom.interventions.patching.scoring import score_labels_next_continuations_ids
from aom.interventions.patching.size_standard_protocol import SizeStandardNamedSpanProtocol
from aom.models.loader import LoadedModel, load_causal_lm
from aom.utils import configure_logprob_computation, get_best_device


DEFAULT_TOKENIZERS = ["gpt2", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
DEFAULT_BEHAVIOR_MODELS = ["Qwen/Qwen2.5-3B"]
DETERMINACY_CHOICES = {
    "def_small": [" definitely small", " clearly small"],
    "borderline": [" borderline", " neither clearly small nor large"],
    "def_large": [" definitely large", " clearly large"],
}
CONTEXT_BIAS_PROMPTS: dict[str, list[str]] = {
    "standard_swap:t1": [
        "The reference for this object is 100 cm. The measured object is 100 cm. Therefore it is",
    ],
    "standard_swap:t2": [
        "An object has a trial standard of 100 cm. Its measured length is 100 cm, so it is",
    ],
    "standard_swap:t3": [
        "This object is measured at 100 cm. Relative to a 100 cm standard, it is",
    ],
    "class_swap:t1": [
        "For class object, 100 cm is",
    ],
    "class_swap:t2": [
        "As object sizes go, 100 cm is",
    ],
    "class_swap:t3": [
        "Compared with a typical object, 100 cm is",
    ],
    "class_penumbra:t1": [
        "For class object, 100 cm is",
    ],
    "class_penumbra:t2": [
        "As object sizes go, 100 cm is",
    ],
    "class_penumbra:t3": [
        "Compared with a typical object, 100 cm is",
    ],
    "conflict_swap:t1": [
        "For this object, the class is object. The stated reference is 100 cm. Its measured length is 100 cm, so it is",
    ],
    "conflict_swap:t2": [
        "This item's class is object. The comparison standard is 100 cm. The item measures 100 cm and is",
    ],
    "conflict_swap:t3": [
        "Category: object. Reference length: 100 cm. Observed length: 100 cm. Verdict:",
    ],
    "penumbra_standard:t1": [
        "The reference for this object is 100 cm. The measured object is 100 cm. Therefore it is",
    ],
    "penumbra_standard:t2": [
        "An object has a trial standard of 100 cm. Its measured length is 100 cm, so it is",
    ],
    "penumbra_standard:t3": [
        "This object is measured at 100 cm. Relative to a 100 cm standard, it is",
    ],
}
DEFAULT_DETERMINACY_BIAS_CONTEXTS: tuple[dict[str, str], ...] = (
    {
        "expected_label": "def_small",
        "prompt": "Compared with a building, a grain of rice is",
    },
    {
        "expected_label": "def_small",
        "prompt": "Compared with a suitcase, a paperclip is",
    },
    {
        "expected_label": "def_large",
        "prompt": "Compared with a coin, an airplane is",
    },
    {
        "expected_label": "def_large",
        "prompt": "Compared with a teacup, a refrigerator is",
    },
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(s: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return out.strip("_") or "x"


def _margin(scores: Mapping[str, float], expected: str) -> float:
    if expected not in scores:
        raise KeyError(f"Missing expected label {expected!r} in scores {sorted(scores)}")
    return float(scores[expected] - max(v for k, v in scores.items() if k != expected))


def _argmax_label(scores: Mapping[str, float]) -> str:
    return max(scores.items(), key=lambda kv: float(kv[1]))[0]


def _corrected_scores(scores: Mapping[str, float], bias: Mapping[str, float]) -> dict[str, float]:
    return {str(label): float(score) - float(bias.get(str(label), 0.0)) for label, score in scores.items()}


def _binary_label_metrics(
    scores: Mapping[str, float],
    *,
    small_label: str = "def_small",
    large_label: str = "def_large",
) -> dict[str, float | str]:
    if small_label not in scores:
        raise KeyError(f"Missing small label {small_label!r} in scores {sorted(scores)}")
    if large_label not in scores:
        raise KeyError(f"Missing large label {large_label!r} in scores {sorted(scores)}")
    small = float(scores[small_label])
    large = float(scores[large_label])
    margin = float(large - small)
    mx = max(small, large)
    exp_small = math.exp(small - mx)
    exp_large = math.exp(large - mx)
    denom = exp_small + exp_large
    p_small = float(exp_small / denom)
    p_large = float(exp_large / denom)
    entropy = 0.0
    for p in (p_small, p_large):
        if p > 0.0:
            entropy -= p * math.log(p)
    return {
        "pred_label": large_label if margin >= 0.0 else small_label,
        "margin": margin,
        "abs_margin": abs(margin),
        "prob_def_small": p_small,
        "prob_def_large": p_large,
        "entropy": float(entropy),
        "entropy_norm": float(entropy / math.log(2.0)),
    }


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be an object in {path}")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _slugged_path(base: Path, slug: str, *, default_name: str, suffix: str) -> Path:
    if str(base).strip():
        if base.suffix:
            return base.with_name(f"{base.stem}.{slug}{base.suffix}")
        return base.with_name(f"{base.name}.{slug}{suffix}")
    return base.parent / f"{default_name}.{slug}{suffix}"


def filtered_jsonl_path(*, out_dir: Path, base_path: str, model_slug: str) -> Path:
    if str(base_path).strip():
        return _slugged_path(Path(str(base_path)), model_slug, default_name="size_standard.filtered", suffix=".jsonl")
    return out_dir / f"size_standard.filtered.{model_slug}.jsonl"


def filtered_summary_path(*, filtered_path: Path, base_path: str, model_slug: str) -> Path:
    if str(base_path).strip():
        return _slugged_path(Path(str(base_path)), model_slug, default_name="size_standard.filtered", suffix=".summary.json")
    return filtered_path.with_suffix(".summary.json")


def _resolve_behavior_models(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "behavior_models", None)
    if raw is not None and len(raw) > 0:
        return [str(x) for x in raw]
    legacy = str(getattr(args, "behavior_model", "") or "").strip()
    if legacy:
        return [legacy]
    return list(DEFAULT_BEHAVIOR_MODELS)


def _bias_prompts(raw: str) -> list[str]:
    prompts = [p.strip() for p in str(raw).split("|") if p.strip()]
    if not prompts:
        raise ValueError("--bias_prompts must contain at least one non-empty prompt")
    return prompts


def _bias_context_key(md: Mapping[str, Any]) -> str:
    family = str(md.get("family", "")).strip()
    template_id = str(md.get("template_id", "")).strip()
    if not family or not template_id:
        return ""
    return f"{family}:t{template_id}"


def _mean_label_bias(contexts: Mapping[str, Mapping[str, float]], choices: Mapping[str, list[str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in choices:
        vals = [float(bias[str(label)]) for bias in contexts.values() if str(label) in bias]
        out[str(label)] = float(sum(vals) / len(vals)) if vals else 0.0
    return out


def _determinacy_bias_from_scored_contexts(
    scored_contexts: Sequence[Mapping[str, Any]],
    choices: Mapping[str, list[str]],
) -> dict[str, float]:
    """Estimate lexical determinacy bias only from non-content observations.

    In particular, borderline bias must be estimated from clear-small and
    clear-large contexts; a neutral context is semantically borderline and would
    subtract the signal this readout is meant to measure.
    """
    accum: dict[str, list[float]] = {str(label): [] for label in choices}
    for ctx in scored_contexts:
        expected = str(ctx.get("expected_label", "")).strip()
        scores = ctx.get("scores", {})
        if expected and expected not in accum:
            raise ValueError(f"Unknown determinacy expected_label={expected!r}")
        if not isinstance(scores, Mapping):
            raise ValueError("Each determinacy bias context needs a scores mapping")
        for label in accum:
            if label == expected:
                continue
            if label not in scores:
                raise ValueError(f"Missing determinacy score for label={label!r}")
            accum[label].append(float(scores[label]))
    missing = [label for label, vals in accum.items() if not vals]
    if missing:
        raise ValueError(
            "Determinacy bias contexts must include non-content observations "
            f"for every label; missing {missing!r}"
        )
    return {label: float(sum(vals) / len(vals)) for label, vals in sorted(accum.items())}


def _load_determinacy_bias_contexts(raw: str) -> list[dict[str, str]]:
    spec = str(raw or "").strip()
    if not spec:
        return [dict(ctx) for ctx in DEFAULT_DETERMINACY_BIAS_CONTEXTS]
    path = Path(spec)
    if not path.exists():
        raise ValueError(
            "--determinacy_bias_prompts must point to a JSONL file with prompt/expected_label "
            "objects or a text file with expected_label<TAB>prompt lines"
        )
    rows: list[dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        for idx, obj in enumerate(_read_jsonl_rows(path), start=1):
            prompt = str(obj.get("prompt", "")).strip()
            expected = str(obj.get("expected_label", obj.get("label", ""))).strip()
            if not prompt or not expected:
                raise ValueError(f"Determinacy context {path}:{idx} needs prompt and expected_label")
            rows.append({"prompt": prompt, "expected_label": expected})
    else:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if "\t" not in text:
                raise ValueError(
                    f"Determinacy context {path}:{line_no} must use expected_label<TAB>prompt"
                )
            expected, prompt = text.split("\t", 1)
            expected = expected.strip()
            prompt = prompt.strip()
            if not prompt or not expected:
                raise ValueError(f"Determinacy context {path}:{line_no} needs prompt and expected_label")
            rows.append({"prompt": prompt, "expected_label": expected})
    if not rows:
        raise ValueError(f"No determinacy bias contexts found in {path}")
    return rows


def _parse_min_kept_by_family(raw: list[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in raw or []:
        if ":" not in str(spec):
            raise ValueError(f"--min_kept_by_family entries must be FAMILY:N, got {spec!r}")
        family, n_raw = str(spec).split(":", 1)
        family = family.strip()
        if not family:
            raise ValueError(f"Empty family in --min_kept_by_family entry {spec!r}")
        n = int(n_raw)
        if n < 0:
            raise ValueError(f"Negative minimum in --min_kept_by_family entry {spec!r}")
        out[family] = int(n)
    return out


def _side_abs_log_ratio(md: Mapping[str, Any], side_name: str) -> float | None:
    keys = (
        ("cf_abs_log_ratio_to_prior", "cf_log_ratio_to_prior", "cf_abs_log_ratio", "cf_log_ratio")
        if str(side_name) == "cf"
        else ("abs_log_ratio_to_prior", "log_ratio_to_prior", "abs_log_ratio", "log_ratio")
    )
    raw = None
    for key in keys:
        raw = md.get(key, None)
        if raw not in (None, ""):
            break
    if raw is None or raw == "":
        return None
    try:
        return abs(float(raw))
    except Exception:
        return None


def _is_soft_penumbra_side(md: Mapping[str, Any], side_name: str, threshold: float) -> bool:
    family = str(md.get("family", ""))
    if family not in {"penumbra_standard", "class_penumbra"}:
        return False
    if family == "class_penumbra" and str(side_name) != "base":
        return False
    target_abs = md.get("target_abs_log_ratio", None)
    if target_abs not in (None, ""):
        try:
            return abs(float(target_abs)) <= float(threshold)
        except Exception:
            pass
    abs_log = _side_abs_log_ratio(md, side_name)
    return abs_log is not None and abs(float(abs_log)) <= float(threshold)


def _score_prompt(
    *,
    loaded: LoadedModel,
    prompt: str,
    choices: Mapping[str, list[str]],
    device: torch.device,
) -> dict[str, float]:
    ids = loaded.tokenizer(str(prompt), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if ids.size(1) < 1:
        raise ValueError("Prompt for label scoring must tokenize to at least one token")
    return score_labels_next_continuations_ids(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompt_ids=ids,
        choices=choices,
        device=device,
        normalize_by_length=True,
        label_aggregation="logmeanexp",
    )


def _compute_label_bias(
    *,
    loaded: LoadedModel,
    prompts: list[str],
    choices: Mapping[str, list[str]],
    device: torch.device,
) -> dict[str, Any]:
    prompt_scores: list[dict[str, Any]] = []
    accum: dict[str, list[float]] = {str(label): [] for label in choices}
    for prompt in prompts:
        scores = _score_prompt(loaded=loaded, prompt=str(prompt), choices=choices, device=device)
        prompt_scores.append({"prompt": str(prompt), "scores": dict(scores)})
        for label, score in scores.items():
            accum.setdefault(str(label), []).append(float(score))
    label_bias = {
        str(label): float(sum(vals) / len(vals))
        for label, vals in sorted(accum.items())
        if vals
    }
    return {
        "prompts": list(prompts),
        "prompt_scores": prompt_scores,
        "label_bias": label_bias,
    }


def _compute_context_label_biases(
    *,
    loaded: LoadedModel,
    choices: Mapping[str, list[str]],
    device: torch.device,
) -> dict[str, Any]:
    contexts: dict[str, Any] = {}
    label_biases: dict[str, dict[str, float]] = {}
    for context_key, prompts in sorted(CONTEXT_BIAS_PROMPTS.items()):
        bias_info = _compute_label_bias(
            loaded=loaded,
            prompts=list(prompts),
            choices=choices,
            device=device,
        )
        label_bias = {str(k): float(v) for k, v in dict(bias_info["label_bias"]).items()}
        label_biases[str(context_key)] = label_bias
        contexts[str(context_key)] = bias_info
    return {
        "source": "context_neutral_defaults",
        "contexts": contexts,
        "label_bias_by_context": label_biases,
        "label_bias": _mean_label_bias(label_biases, choices),
    }


def _compute_determinacy_label_bias(
    *,
    loaded: LoadedModel,
    contexts: Sequence[Mapping[str, str]],
    choices: Mapping[str, list[str]],
    device: torch.device,
) -> dict[str, Any]:
    scored_contexts: list[dict[str, Any]] = []
    for ctx in contexts:
        prompt = str(ctx.get("prompt", "")).strip()
        expected = str(ctx.get("expected_label", "")).strip()
        if not prompt or not expected:
            raise ValueError("Determinacy bias contexts require prompt and expected_label")
        scores = _score_prompt(loaded=loaded, prompt=prompt, choices=choices, device=device)
        scored_contexts.append(
            {
                "prompt": prompt,
                "expected_label": expected,
                "scores": {str(label): float(score) for label, score in scores.items()},
            }
        )
    label_bias = _determinacy_bias_from_scored_contexts(scored_contexts, choices)
    return {
        "source": "determinate_context_non_content",
        "contexts": [dict(ctx) for ctx in contexts],
        "context_scores": scored_contexts,
        "label_bias": label_bias,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Acceptance checks for size_standard.jsonl.")
    p.add_argument("--size_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--tokenizer_models", nargs="+", default=list(DEFAULT_TOKENIZERS))
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--skip_behavioral", action="store_true")
    p.add_argument("--behavior_model", type=str, default="", help="Deprecated single-model alias.")
    p.add_argument("--behavior_models", nargs="+", default=None)
    p.add_argument("--behavior_revision", type=str, default=None)
    p.add_argument("--behavior_tokenizer_revision", type=str, default=None)
    p.add_argument("--behavior_torch_dtype", type=str, default="float32")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--gate_mode", type=str, default="bias_corrected", choices=["raw", "bias_corrected"])
    p.add_argument(
        "--bias_prompts",
        type=str,
        default="",
        help=(
            "Optional global bias prompts separated by '|'. "
            "Default: use neutralized task-context prompts per family/template."
        ),
    )
    p.add_argument(
        "--assert_sides",
        type=str,
        default="base,cf",
        help="Comma-separated sides to assert. Default: base,cf.",
    )
    p.add_argument(
        "--min_gate_margin",
        type=float,
        default=0.0,
        help="Minimum absolute corrected margin for asserted gate decisions. Default: 0.0.",
    )
    p.add_argument(
        "--penumbra_soft_abs_log_ratio",
        type=float,
        default=0.15,
        help=(
            "For penumbra_standard rows, and for the base side of class_penumbra rows, "
            "record margins but do not reject by argmax when the target abs log ratio is at or below this threshold. "
            "Default: 0.15."
        ),
    )
    p.add_argument(
        "--score_determinacy",
        action="store_true",
        help="Also write definitely-small/borderline/definitely-large continuation scores to the margins CSV.",
    )
    p.add_argument(
        "--determinacy_bias_mode",
        type=str,
        default="bias_corrected",
        choices=["raw", "bias_corrected"],
        help="Bias mode for determinacy argmax readout. Default: bias_corrected.",
    )
    p.add_argument(
        "--determinacy_bias_prompts",
        type=str,
        default="",
        help=(
            "Optional determinacy bias context file. JSONL rows need prompt and expected_label; "
            "plain text uses expected_label<TAB>prompt. Default: clear-small and clear-large contexts."
        ),
    )
    p.add_argument("--emit_filtered_jsonl", type=str, default="")
    p.add_argument("--emit_filtered_summary", type=str, default="")
    p.add_argument("--filter_manifest_path", type=str, default="")
    p.add_argument("--min_filtered_rows", type=int, default=1)
    p.add_argument(
        "--min_kept_by_family",
        nargs="*",
        default=None,
        help="Optional family-specific minimums, e.g. standard_swap:25 class_swap:25 conflict_swap:20.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    size_path = Path(str(args.size_path))
    out_dir = Path(str(args.out_dir)) if str(args.out_dir).strip() else size_path.parent
    summary_path = out_dir / "size_standard.acceptance.json"
    filter_manifest_path = (
        Path(str(args.filter_manifest_path))
        if str(args.filter_manifest_path).strip()
        else out_dir / "size_standard.filter_manifest.json"
    )
    if bool(args.skip_behavioral) and (str(args.emit_filtered_jsonl).strip() or str(args.filter_manifest_path).strip()):
        raise ValueError("Filtered artifacts require behavioral scoring; remove --skip_behavioral")

    items, manifest = load_counterfactual_pairs_with_manifest(
        size_path,
        role="size_standard",
        error_policy="raise",
    )
    if not items:
        raise ValueError("size_standard dataset has zero valid rows")
    has_class_penumbra = any(str((it.metadata or {}).get("family", "")) == "class_penumbra" for it in items)
    if bool(has_class_penumbra) and not bool(args.skip_behavioral) and not bool(args.score_determinacy):
        raise ValueError("class_penumbra rows require --score_determinacy; determinacy is the registered readout for this arm.")
    protocol = SizeStandardNamedSpanProtocol()

    tokenizer_summaries: dict[str, Any] = {}
    for model_name in [str(x) for x in args.tokenizer_models]:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
        )
        if not bool(getattr(tok, "is_fast", False)):
            raise ValueError(f"Tokenizer for {model_name!r} is not fast.")
        cases, skips = protocol.build_cases(tokenizer=tok, items=items, device=torch.device("cpu"))
        n_span_mismatch = sum(1 for c in cases if len(c.receiver_span) != len(c.donor_span))
        skip_counts = Counter(str(s.reason) for s in skips)
        tokenizer_summaries[str(model_name)] = {
            "n_cases": int(len(cases)),
            "n_skips": int(len(skips)),
            "n_cases_skipped_span_mismatch": int(n_span_mismatch),
            "skip_reason_counts": dict(sorted(skip_counts.items())),
        }
        if skips:
            raise ValueError(f"Protocol skips for tokenizer={model_name!r}: {dict(skip_counts)}")
        if n_span_mismatch:
            raise ValueError(f"Span mismatches for tokenizer={model_name!r}: {n_span_mismatch}")

    behavioral_by_model: dict[str, Any] = {}
    manifest_entries: list[dict[str, Any]] = []
    post_write_errors: list[str] = []
    if not bool(args.skip_behavioral):
        configure_logprob_computation(logprobs_dtype=torch.float32, strict_finite=True)
        if args.device == "auto":
            device = get_best_device()
        else:
            device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

        sides_to_assert = {s.strip() for s in str(args.assert_sides).split(",") if s.strip()}
        invalid_sides = sorted(s for s in sides_to_assert if s not in {"base", "cf"})
        if invalid_sides:
            raise ValueError(f"Invalid --assert_sides entries: {invalid_sides}")
        min_gate_margin = float(getattr(args, "min_gate_margin", 0.0))
        if min_gate_margin < 0:
            raise ValueError("--min_gate_margin must be >= 0")
        if float(args.penumbra_soft_abs_log_ratio) < 0:
            raise ValueError("--penumbra_soft_abs_log_ratio must be >= 0")

        raw_rows = _read_jsonl_rows(size_path)
        raw_by_id: dict[str, dict[str, Any]] = {}
        for row in raw_rows:
            item_id = str(row.get("item_id", ""))
            if item_id in raw_by_id:
                raise ValueError(f"Duplicate item_id in raw JSONL: {item_id}")
            raw_by_id[item_id] = row

        choices = items[0].choices
        behavior_models = _resolve_behavior_models(args)
        explicit_bias_prompts = str(args.bias_prompts).strip()
        bias_prompts = _bias_prompts(explicit_bias_prompts) if explicit_bias_prompts else []
        min_kept_by_family = _parse_min_kept_by_family(getattr(args, "min_kept_by_family", None))

        for model_name in behavior_models:
            model_slug = _slug(model_name)
            margins_path = out_dir / f"size_standard.margins.{model_slug}.csv"
            filtered_path = filtered_jsonl_path(
                out_dir=out_dir,
                base_path=str(args.emit_filtered_jsonl),
                model_slug=model_slug,
            )
            filtered_summary_p = filtered_summary_path(
                filtered_path=filtered_path,
                base_path=str(args.emit_filtered_summary),
                model_slug=model_slug,
            )

            loaded = load_causal_lm(
                str(model_name),
                device=device,
                torch_dtype=str(args.behavior_torch_dtype),
                revision=args.behavior_revision,
                tokenizer_revision=args.behavior_tokenizer_revision,
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                attn_implementation="eager",
            )

            if str(args.gate_mode) == "bias_corrected":
                if bias_prompts:
                    bias_info = _compute_label_bias(
                        loaded=loaded,
                        prompts=bias_prompts,
                        choices=choices,
                        device=device,
                    )
                    bias_info["source"] = "explicit_global_prompts"
                    label_bias_by_context: dict[str, dict[str, float]] = {}
                    label_bias = {str(k): float(v) for k, v in dict(bias_info["label_bias"]).items()}
                else:
                    bias_info = _compute_context_label_biases(
                        loaded=loaded,
                        choices=choices,
                        device=device,
                    )
                    label_bias_by_context = {
                        str(k): {str(label): float(score) for label, score in dict(v).items()}
                        for k, v in dict(bias_info["label_bias_by_context"]).items()
                    }
                    label_bias = {str(k): float(v) for k, v in dict(bias_info["label_bias"]).items()}
            else:
                label_bias_by_context = {}
                label_bias = {str(label): 0.0 for label in choices}
                bias_info = {"source": "raw_no_bias", "prompts": [], "prompt_scores": [], "label_bias": dict(label_bias)}

            if bool(args.score_determinacy):
                if str(args.determinacy_bias_mode) == "bias_corrected":
                    determinacy_contexts = _load_determinacy_bias_contexts(str(args.determinacy_bias_prompts))
                    determinacy_bias_info = _compute_determinacy_label_bias(
                        loaded=loaded,
                        contexts=determinacy_contexts,
                        choices=DETERMINACY_CHOICES,
                        device=device,
                    )
                    determinacy_label_bias = {
                        str(k): float(v) for k, v in dict(determinacy_bias_info["label_bias"]).items()
                    }
                else:
                    determinacy_label_bias = {str(label): 0.0 for label in DETERMINACY_CHOICES}
                    determinacy_bias_info = {
                        "source": "raw_no_bias",
                        "contexts": [],
                        "context_scores": [],
                        "label_bias": dict(determinacy_label_bias),
                    }
            else:
                determinacy_label_bias = {str(label): 0.0 for label in DETERMINACY_CHOICES}
                determinacy_bias_info = {}

            def score_gate_row(
                *,
                item_id: str,
                md: Mapping[str, Any],
                prompt: str,
                expected: str,
                side_name: str,
                gate_asserted: bool,
            ) -> tuple[dict[str, Any], str | None]:
                raw_scores = _score_prompt(
                    loaded=loaded,
                    prompt=str(prompt),
                    choices=choices,
                    device=device,
                )
                bias_context = _bias_context_key(md)
                side_label_bias = label_bias_by_context.get(str(bias_context), label_bias)
                gate_scores = _corrected_scores(raw_scores, side_label_bias)
                raw_pred = _argmax_label(raw_scores)
                gate_pred = _argmax_label(gate_scores)
                raw_margin = _margin(raw_scores, str(expected))
                gate_margin = _margin(gate_scores, str(expected))
                side_abs_log_ratio = _side_abs_log_ratio(md, side_name)
                row: dict[str, Any] = {
                    "model": str(model_name),
                    "model_slug": str(model_slug),
                    "gate_mode": str(args.gate_mode),
                    "bias_context": str(bias_context),
                    "item_id": str(item_id),
                    "side": str(side_name),
                    "gate_asserted": bool(gate_asserted),
                    "family": str(md.get("family", "")),
                    "patch_span": str(md.get("patch_span", "")),
                    "receiver_group_id": str(md.get("receiver_group_id", "")),
                    "conflict_receiver_id": str(md.get("conflict_receiver_id", "")),
                    "conflict_direction": str(md.get("conflict_direction", "")),
                    "donor_kind": str(md.get("donor_kind", "")),
                    "class_name": str(md.get("class_name", "")),
                    "cf_class_name": str(md.get("cf_class_name", "")),
                    "standard_cm": md.get("standard_cm", ""),
                    "cf_standard_cm": md.get("cf_standard_cm", ""),
                    "value_cm": md.get("value_cm", ""),
                    "value_text": str(md.get("value_text", "")),
                    "abs_log_ratio": "" if side_abs_log_ratio is None else float(side_abs_log_ratio),
                    "log_ratio_to_prior": md.get("log_ratio_to_prior", ""),
                    "cf_log_ratio_to_prior": md.get("cf_log_ratio_to_prior", ""),
                    "abs_log_ratio_to_prior": md.get("abs_log_ratio_to_prior", ""),
                    "cf_abs_log_ratio_to_prior": md.get("cf_abs_log_ratio_to_prior", ""),
                    "penumbra_axis": str(md.get("penumbra_axis", "")),
                    "penumbral_relation": str(md.get("penumbral_relation", "")),
                    "value_pair_id": str(md.get("value_pair_id", "")),
                    "rank_in_chain": str(md.get("rank_in_chain", "")),
                    "monotonicity_direction": str(md.get("monotonicity_direction", "")),
                    "margin_bin": str(md.get("margin_bin", "")),
                    "expected_label": str(expected),
                    "pred_label": raw_pred,
                    "gate_pred_label": gate_pred,
                    "size_margin": float(raw_margin),
                    "gate_margin": float(gate_margin),
                }
                for label in sorted(raw_scores):
                    row[f"score_{label}"] = float(raw_scores[label])
                    row[f"bias_{label}"] = float(side_label_bias.get(str(label), 0.0))
                    row[f"gate_score_{label}"] = float(gate_scores[label])
                if bool(args.score_determinacy):
                    det_scores = _score_prompt(
                        loaded=loaded,
                        prompt=str(prompt),
                        choices=DETERMINACY_CHOICES,
                        device=device,
                    )
                    det_gate_scores = _corrected_scores(det_scores, determinacy_label_bias)
                    det_pred = _argmax_label(det_scores)
                    det_gate_pred = _argmax_label(det_gate_scores)
                    det_binary = _binary_label_metrics(det_scores)
                    det_binary_corrected = _binary_label_metrics(det_gate_scores)
                    row["determinacy_pred_label"] = str(det_pred)
                    row["determinacy_borderline_margin"] = float(
                        det_scores["borderline"] - max(det_scores["def_small"], det_scores["def_large"])
                    )
                    row["determinacy_pred_label_corrected"] = str(det_gate_pred)
                    row["determinacy_borderline_margin_corrected"] = float(
                        det_gate_scores["borderline"]
                        - max(det_gate_scores["def_small"], det_gate_scores["def_large"])
                    )
                    row["determinacy_bias_mode"] = str(args.determinacy_bias_mode)
                    row["determinacy_bias_context"] = json.dumps(
                        determinacy_bias_info.get("contexts", []),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    for label in sorted(det_scores):
                        row[f"det_score_{label}"] = float(det_scores[label])
                        row[f"det_bias_{label}"] = float(determinacy_label_bias.get(str(label), 0.0))
                        row[f"det_score_{label}_corrected"] = float(det_gate_scores[label])
                    for key, value in det_binary.items():
                        row[f"determinacy_binary_{key}"] = value
                    for key, value in det_binary_corrected.items():
                        row[f"determinacy_binary_{key}_corrected"] = value
                reason: str | None = None
                if bool(gate_asserted):
                    if gate_pred != str(expected):
                        reason = f"{side_name}_gate_pred_{gate_pred}_expected_{expected}"
                    elif abs(float(gate_margin)) < float(min_gate_margin):
                        reason = f"{side_name}_gate_margin_below_{float(min_gate_margin):.6g}"
                return row, reason

            margin_rows: list[dict[str, Any]] = []
            filtered_rows: list[dict[str, Any]] = []
            rejected_rows: list[dict[str, Any]] = []
            rejection_counts: Counter[str] = Counter()
            side_counts: Counter[tuple[str, str, str, str]] = Counter()
            gate_side_counts: Counter[tuple[str, str, str, str]] = Counter()

            for it in items:
                md = it.metadata or {}
                item_side_rows: list[dict[str, Any]] = []
                item_reasons: list[str] = []
                for side_name, side in (("base", it.base), ("cf", it.cf)):
                    penumbra_soft = _is_soft_penumbra_side(
                        md,
                        side_name,
                        threshold=float(args.penumbra_soft_abs_log_ratio),
                    )
                    gate_asserted = (
                        (side_name in sides_to_assert)
                        and str(md.get("family", "")) != "conflict_swap"
                        and not bool(penumbra_soft)
                    )
                    row, reason = score_gate_row(
                        item_id=str(it.item_id),
                        md=md,
                        prompt=str(side.prompt),
                        expected=str(side.expected_label),
                        side_name=str(side_name),
                        gate_asserted=bool(gate_asserted),
                    )
                    row["penumbra_soft_gate"] = bool(penumbra_soft)
                    margin_rows.append(row)
                    item_side_rows.append(row)
                    side_counts[
                        (
                            str(side_name),
                            str(md.get("family", "")),
                            str(side.expected_label),
                            str(row["pred_label"]),
                        )
                    ] += 1
                    gate_side_counts[
                        (
                            str(side_name),
                            str(md.get("family", "")),
                            str(side.expected_label),
                            str(row["gate_pred_label"]),
                        )
                    ] += 1
                    if reason is not None:
                        item_reasons.append(str(reason))

                if str(md.get("family", "")) == "conflict_swap":
                    siblings = md.get("conflict_gate_siblings", None)
                    if not isinstance(siblings, Mapping):
                        item_reasons.append("missing_conflict_gate_siblings")
                    else:
                        for role in ("class", "standard"):
                            sib = siblings.get(role)
                            if not isinstance(sib, Mapping):
                                item_reasons.append(f"missing_conflict_sibling_{role}")
                                continue
                            row, reason = score_gate_row(
                                item_id=str(it.item_id),
                                md=md,
                                prompt=str(sib.get("prompt", "")),
                                expected=str(sib.get("expected_label", "")),
                                side_name=f"sibling_{role}",
                                gate_asserted=True,
                            )
                            margin_rows.append(row)
                            item_side_rows.append(row)
                            side_counts[
                                (
                                    f"sibling_{role}",
                                    str(md.get("family", "")),
                                    str(row["expected_label"]),
                                    str(row["pred_label"]),
                                )
                            ] += 1
                            gate_side_counts[
                                (
                                    f"sibling_{role}",
                                    str(md.get("family", "")),
                                    str(row["expected_label"]),
                                    str(row["gate_pred_label"]),
                                )
                            ] += 1
                            if reason is not None:
                                item_reasons.append(str(reason))

                kept = len(item_reasons) == 0
                reject_reason = ";".join(item_reasons)
                for row in item_side_rows:
                    row["kept"] = bool(kept)
                    row["reject_reason"] = str(reject_reason)
                if kept:
                    raw_row = raw_by_id.get(str(it.item_id))
                    if raw_row is None:
                        raise ValueError(f"Missing raw row for item_id={it.item_id}")
                    filtered_rows.append(raw_row)
                else:
                    raw_row = raw_by_id.get(str(it.item_id), {"item_id": str(it.item_id)})
                    rejected_rows.append(raw_row)
                    for reason in item_reasons:
                        rejection_counts[str(reason)] += 1

            _write_csv(margins_path, margin_rows)
            _write_jsonl(filtered_path, filtered_rows)
            kept_by_family = Counter(str((row.get("metadata") or {}).get("family", "")) for row in filtered_rows)
            rejected_by_family = Counter(str((row.get("metadata") or {}).get("family", "")) for row in rejected_rows)
            model_summary = {
                "source_size_path": str(size_path),
                "filtered_jsonl": str(filtered_path),
                "filtered_summary": str(filtered_summary_p),
                "margins_csv": str(margins_path),
                "n_input_rows": int(len(items)),
                "n_kept_rows": int(len(filtered_rows)),
                "n_rejected_rows": int(len(rejected_rows)),
                "kept_by_family": dict(sorted(kept_by_family.items())),
                "rejected_by_family": dict(sorted(rejected_by_family.items())),
                "rejection_counts": dict(sorted(rejection_counts.items())),
                "behavior_model": str(model_name),
                "model_slug": str(model_slug),
                "behavior_revision": "" if args.behavior_revision is None else str(args.behavior_revision),
                "behavior_tokenizer_revision": ""
                if args.behavior_tokenizer_revision is None
                else str(args.behavior_tokenizer_revision),
                "hf_model_commit_hash": str(loaded.model_commit_hash or ""),
                "hf_tokenizer_revision_effective": str(loaded.tokenizer_revision_effective or ""),
                "local_files_only": bool(args.local_files_only),
                "torch_dtype": str(args.behavior_torch_dtype),
                "tokenizer_models_checked": [str(x) for x in args.tokenizer_models],
                "assert_sides": sorted(sides_to_assert),
                "min_gate_margin": float(min_gate_margin),
                "min_kept_by_family": {str(k): int(v) for k, v in sorted(min_kept_by_family.items())},
                "penumbra_soft_abs_log_ratio": float(args.penumbra_soft_abs_log_ratio),
                "score_determinacy": bool(args.score_determinacy),
                "determinacy_bias_mode": str(args.determinacy_bias_mode),
                "determinacy_continuous_readout": (
                    "two_way_definite_margin_entropy" if bool(args.score_determinacy) else ""
                ),
                "determinacy_bias_info": determinacy_bias_info,
                "conflict_gate_policy": "siblings",
                "gate_mode": str(args.gate_mode),
                "bias_info": bias_info,
                "raw_side_prediction_counts": {
                    "|".join(k): int(v) for k, v in sorted(side_counts.items())
                },
                "gate_side_prediction_counts": {
                    "|".join(k): int(v) for k, v in sorted(gate_side_counts.items())
                },
                "first_rejections": [str(r.get("item_id", "")) for r in rejected_rows[:10]],
            }
            _write_json(filtered_summary_p, model_summary)
            behavioral_by_model[str(model_name)] = dict(model_summary)
            entry = {
                "model": str(model_name),
                "model_slug": str(model_slug),
                "filtered_jsonl": str(filtered_path),
                "filtered_summary": str(filtered_summary_p),
                "margins_csv": str(margins_path),
                "n_kept_rows": int(len(filtered_rows)),
                "n_rejected_rows": int(len(rejected_rows)),
                "gate_mode": str(args.gate_mode),
                "min_gate_margin": float(min_gate_margin),
                "min_kept_by_family": {str(k): int(v) for k, v in sorted(min_kept_by_family.items())},
                "penumbra_soft_abs_log_ratio": float(args.penumbra_soft_abs_log_ratio),
                "score_determinacy": bool(args.score_determinacy),
                "determinacy_bias_mode": str(args.determinacy_bias_mode),
                "determinacy_continuous_readout": (
                    "two_way_definite_margin_entropy" if bool(args.score_determinacy) else ""
                ),
                "determinacy_bias_info": determinacy_bias_info,
                "conflict_gate_policy": "siblings",
                "bias_info": bias_info,
            }
            manifest_entries.append(entry)
            if len(filtered_rows) < int(args.min_filtered_rows):
                post_write_errors.append(
                    f"{model_name}: filtered artifact has {len(filtered_rows)} rows, "
                    f"below --min_filtered_rows {int(args.min_filtered_rows)}"
                )
            for family, min_n in sorted(min_kept_by_family.items()):
                actual_n = int(kept_by_family.get(str(family), 0))
                if actual_n < int(min_n):
                    post_write_errors.append(
                        f"{model_name}: kept_by_family[{family}]={actual_n}, below required {int(min_n)}"
                    )

            del loaded
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        filter_manifest = {
            "manifest_type": "size_standard_filter_manifest",
            "created_at_utc": _utc_now_iso(),
            "source_size_path": str(size_path),
            "gate_mode": str(args.gate_mode),
            "assert_sides": sorted(sides_to_assert),
            "min_gate_margin": float(min_gate_margin),
            "penumbra_soft_abs_log_ratio": float(args.penumbra_soft_abs_log_ratio),
            "score_determinacy": bool(args.score_determinacy),
            "determinacy_bias_mode": str(args.determinacy_bias_mode),
            "determinacy_continuous_readout": (
                "two_way_definite_margin_entropy" if bool(args.score_determinacy) else ""
            ),
            "conflict_gate_policy": "siblings",
            "min_kept_by_family": {str(k): int(v) for k, v in sorted(min_kept_by_family.items())},
            "tokenizer_models_checked": [str(x) for x in args.tokenizer_models],
            "entries": manifest_entries,
            "models_by_name": {str(e["model"]): dict(e) for e in manifest_entries},
        }
        _write_json(filter_manifest_path, filter_manifest)

    summary = {
        "size_path": str(size_path),
        "n_rows_loaded": int(len(items)),
        "manifest": manifest.as_dict(),
        "tokenizers": tokenizer_summaries,
        "filter_manifest": "" if bool(args.skip_behavioral) else str(filter_manifest_path),
        "behavioral_by_model": behavioral_by_model,
    }
    _write_json(summary_path, summary)
    print(f"Wrote acceptance summary to {summary_path}")
    if not bool(args.skip_behavioral):
        print(f"Wrote filter manifest to {filter_manifest_path}")
    if post_write_errors:
        raise ValueError("; ".join(post_write_errors))


if __name__ == "__main__":
    main()
