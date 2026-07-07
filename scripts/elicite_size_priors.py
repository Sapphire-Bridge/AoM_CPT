#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.interventions.patching.scoring import score_labels_next_continuations_ids
from aom.models.loader import LoadedModel, load_causal_lm
from aom.utils import configure_logprob_computation, get_best_device


DEFAULT_CLASSES = ["ant", "human", "elephant"]
DEFAULT_GRID_CM = "0.1,0.2,0.5,1,2,5,10,20,50,100,170,300,500,1000"
TEMPLATES = (
    "A typical {class_name} is about",
    "The normal length of a {class_name} is about",
    "For a {class_name}, a typical length is",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(s: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return out.strip("_") or "x"


def _fmt_cm(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{float(x):g}"


def _parse_grid(raw: str) -> list[float]:
    vals: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"Grid values must be finite and positive: {part!r}")
        vals.append(float(val))
    if not vals:
        raise ValueError("Empty candidate grid")
    return sorted(set(vals))


def _score_prompt(
    *,
    loaded: LoadedModel,
    prompt: str,
    choices: Mapping[str, list[str]],
    device: torch.device,
) -> dict[str, float]:
    ids = loaded.tokenizer(str(prompt), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if ids.size(1) < 1:
        raise ValueError("Prompt must tokenize to at least one token")
    return score_labels_next_continuations_ids(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompt_ids=ids,
        choices=choices,
        device=device,
        normalize_by_length=True,
        label_aggregation="logmeanexp",
    )


def _softmax_weighted_log_cm(scores: Mapping[str, float]) -> float:
    labels = list(scores)
    vals = torch.tensor([float(scores[k]) for k in labels], dtype=torch.float64)
    weights = torch.softmax(vals, dim=0)
    logs = torch.tensor([math.log(float(k)) for k in labels], dtype=torch.float64)
    return float(torch.exp((weights * logs).sum()).item())


def _second_gap(scores: Mapping[str, float]) -> float:
    vals = sorted((float(v) for v in scores.values()), reverse=True)
    if len(vals) < 2:
        return float("nan")
    return float(vals[0] - vals[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Elicit model-internal typical size priors over a cm grid.")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    p.add_argument("--candidate_grid_cm", type=str, default=DEFAULT_GRID_CM)
    p.add_argument("--out_path", type=str, default=str(ROOT / "data_size_standard" / "size_priors.qwen_qwen2_5_3b.json"))
    p.add_argument("--torch_dtype", type=str, default="float32")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logprob_computation(logprobs_dtype=torch.float32, strict_finite=True)
    device = get_best_device() if str(args.device) == "auto" else torch.device(str(args.device))
    loaded = load_causal_lm(
        str(args.model),
        device=device,
        torch_dtype=str(args.torch_dtype),
        revision=args.revision,
        tokenizer_revision=args.tokenizer_revision,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation="eager",
    )
    grid = _parse_grid(str(args.candidate_grid_cm))
    choices = {str(_fmt_cm(x)): [f" {_fmt_cm(x)} cm"] for x in grid}
    model_priors: dict[str, Any] = {}
    for class_name in [str(c) for c in args.classes]:
        prompt_scores: list[dict[str, Any]] = []
        accum: dict[str, list[float]] = {str(_fmt_cm(x)): [] for x in grid}
        for tmpl in TEMPLATES:
            prompt = tmpl.format(class_name=class_name)
            scores = _score_prompt(loaded=loaded, prompt=prompt, choices=choices, device=device)
            prompt_scores.append({"prompt": prompt, "scores": dict(scores)})
            for label, score in scores.items():
                accum[str(label)].append(float(score))
        mean_scores = {
            label: float(sum(vals) / len(vals))
            for label, vals in sorted(accum.items(), key=lambda kv: float(kv[0]))
            if vals
        }
        best_label = max(mean_scores.items(), key=lambda kv: float(kv[1]))[0]
        model_priors[str(class_name)] = {
            "model_prior_cm_argmax": float(best_label),
            "model_prior_cm_expected_log": float(_softmax_weighted_log_cm(mean_scores)),
            "prior_confidence": float(_second_gap(mean_scores)),
            "score_method": "next_continuation_logmeanexp_length_normalized",
            "unit": "cm",
            "candidate_scores": mean_scores,
            "prompt_scores": prompt_scores,
        }

    out = {
        "created_at_utc": _utc_now_iso(),
        "model": str(args.model),
        "model_slug": _slug(str(args.model)),
        "revision": "" if args.revision is None else str(args.revision),
        "tokenizer_revision": "" if args.tokenizer_revision is None else str(args.tokenizer_revision),
        "hf_model_commit_hash": str(loaded.model_commit_hash or ""),
        "hf_tokenizer_revision_effective": str(loaded.tokenizer_revision_effective or ""),
        "torch_dtype": str(args.torch_dtype),
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
        "candidate_grid_cm": [float(x) for x in grid],
        "templates": list(TEMPLATES),
        "model_priors": model_priors,
    }
    out_path = Path(str(args.out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote model size priors to {out_path}")


if __name__ == "__main__":
    main()
