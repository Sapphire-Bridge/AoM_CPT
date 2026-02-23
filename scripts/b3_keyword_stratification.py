#!/usr/bin/env python3
"""B3: Stratify model DISAMB accuracy by keyword-baseline correctness.

For each of the 52 pairs, flags whether the keyword baseline gets it correct.
Splits model DISAMB accuracy into keyword-correct vs keyword-incorrect groups.
Bootstraps CIs on each group separately.

Usage:
    python scripts/b3_keyword_stratification.py \
        --disamb_path data_paper_hardened_v2/disamb_pairs.jsonl \
        --models gpt2 Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aom.baselines import predict_disamb_keyword
from aom.data.loaders import load_disamb_pairs_with_manifest
from aom.metrics.disamb import score_labels_next_continuations
from aom.models.loader import load_causal_lm
from aom.stats import bootstrap_ci
from aom.utils import get_best_device, set_seed


def compute_keyword_per_pair(items):
    """Return dict pair_id -> keyword_pair_accuracy (0.0, 0.5, or 1.0)."""
    out = {}
    for it in items:
        a_pred = predict_disamb_keyword(it, it.a.prompt)
        b_pred = predict_disamb_keyword(it, it.b.prompt)
        a_correct = float(a_pred == it.a.expected_label)
        b_correct = float(b_pred == it.b.expected_label)
        out[it.pair_id] = 0.5 * (a_correct + b_correct)
    return out


@torch.no_grad()
def compute_model_per_pair(model, tokenizer, items, device):
    """Return dict pair_id -> model_pair_accuracy (0.0, 0.5, or 1.0)."""
    out = {}
    for it in items:
        side_acc = []
        for side in (it.a, it.b):
            scores = score_labels_next_continuations(
                model, tokenizer, side.prompt, it.choices, device,
                normalize_by_length=True,
            )
            pred = scores.argmax_label()
            side_acc.append(float(pred == side.expected_label))
        out[it.pair_id] = float(sum(side_acc) / len(side_acc))
    return out


def stratify_and_bootstrap(
    model_per_pair: Dict[str, float],
    keyword_per_pair: Dict[str, float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, object]:
    """Split model accuracy by keyword correctness; bootstrap each group."""
    # Keyword "correct" = pair accuracy >= 0.5 (at least one side right)
    # More strict: keyword "fully correct" = pair accuracy == 1.0
    kw_correct_ids = [pid for pid, acc in keyword_per_pair.items() if acc >= 1.0]
    kw_incorrect_ids = [pid for pid, acc in keyword_per_pair.items() if acc < 1.0]
    kw_partial_correct_ids = [pid for pid, acc in keyword_per_pair.items() if 0 < acc < 1.0]
    kw_zero_ids = [pid for pid, acc in keyword_per_pair.items() if acc == 0.0]

    model_on_kw_correct = [model_per_pair[pid] for pid in kw_correct_ids if pid in model_per_pair]
    model_on_kw_incorrect = [model_per_pair[pid] for pid in kw_incorrect_ids if pid in model_per_pair]
    model_on_kw_zero = [model_per_pair[pid] for pid in kw_zero_ids if pid in model_per_pair]
    model_all = [model_per_pair[pid] for pid in model_per_pair]

    def _boot(vals):
        if len(vals) < 2:
            mu = sum(vals) / max(1, len(vals)) if vals else float("nan")
            return mu, float("nan"), float("nan")
        return bootstrap_ci(vals, n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    mu_all, lo_all, hi_all = _boot(model_all)
    mu_kc, lo_kc, hi_kc = _boot(model_on_kw_correct)
    mu_ki, lo_ki, hi_ki = _boot(model_on_kw_incorrect)
    mu_kz, lo_kz, hi_kz = _boot(model_on_kw_zero)

    return {
        "n_total": len(model_all),
        "n_kw_correct": len(model_on_kw_correct),
        "n_kw_incorrect": len(model_on_kw_incorrect),
        "n_kw_zero": len(model_on_kw_zero),
        "overall": f"{mu_all:.3f} [{lo_all:.3f}, {hi_all:.3f}]",
        "kw_correct": f"{mu_kc:.3f} [{lo_kc:.3f}, {hi_kc:.3f}]",
        "kw_incorrect": f"{mu_ki:.3f} [{lo_ki:.3f}, {hi_ki:.3f}]",
        "kw_zero": f"{mu_kz:.3f} [{lo_kz:.3f}, {hi_kz:.3f}]",
        "raw_overall": (mu_all, lo_all, hi_all),
        "raw_kw_correct": (mu_kc, lo_kc, hi_kc),
        "raw_kw_incorrect": (mu_ki, lo_ki, hi_ki),
        "raw_kw_zero": (mu_kz, lo_kz, hi_kz),
    }


def main():
    parser = argparse.ArgumentParser(description="B3: keyword stratification analysis")
    parser.add_argument("--disamb_path", default="data_paper_hardened_v2/disamb_pairs.jsonl")
    parser.add_argument("--models", nargs="+",
                        default=["gpt2", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_n", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_best_device()
    print(f"Device: {device}")

    # Load disamb pairs
    items, manifest = load_disamb_pairs_with_manifest(
        args.disamb_path, error_policy="raise",
    )
    print(f"Loaded {len(items)} disamb pairs")

    # Keyword baseline per pair
    keyword_per_pair = compute_keyword_per_pair(items)
    n_kw_correct = sum(1 for v in keyword_per_pair.values() if v >= 1.0)
    n_kw_incorrect = sum(1 for v in keyword_per_pair.values() if v < 1.0)
    n_kw_zero = sum(1 for v in keyword_per_pair.values() if v == 0.0)
    print(f"\nKeyword baseline per-pair breakdown:")
    print(f"  Fully correct (both sides): {n_kw_correct}/{len(keyword_per_pair)}")
    print(f"  Not fully correct: {n_kw_incorrect}/{len(keyword_per_pair)}")
    print(f"  Fully wrong (both sides): {n_kw_zero}/{len(keyword_per_pair)}")

    # Per-pair detail
    print(f"\nPer-pair keyword predictions:")
    for pid, acc in sorted(keyword_per_pair.items()):
        item = next(it for it in items if it.pair_id == pid)
        word = (item.metadata or {}).get("word", item.target)
        print(f"  {pid:30s} word={word:10s} kw_acc={acc:.1f}")

    # Model scoring
    results = {}
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        loaded = load_causal_lm(model_name, device, attn_implementation="eager")
        model, tokenizer = loaded.model, loaded.tokenizer
        model.eval()

        model_per_pair = compute_model_per_pair(model, tokenizer, items, device)

        strat = stratify_and_bootstrap(
            model_per_pair, keyword_per_pair,
            n_bootstrap=args.bootstrap_n, seed=args.seed,
        )
        results[model_name] = strat

        print(f"\n  Model: {model_name}")
        print(f"  Overall DISAMB:         {strat['overall']}")
        print(f"  Keyword-correct pairs:  {strat['kw_correct']}  (n={strat['n_kw_correct']})")
        print(f"  Keyword-incorrect pairs: {strat['kw_incorrect']}  (n={strat['n_kw_incorrect']})")
        print(f"  Keyword-zero pairs:     {strat['kw_zero']}  (n={strat['n_kw_zero']})")

        # Per-pair detail for this model
        print(f"\n  Per-pair model vs keyword:")
        for pid in sorted(model_per_pair.keys()):
            item = next(it for it in items if it.pair_id == pid)
            word = (item.metadata or {}).get("word", item.target)
            m_acc = model_per_pair[pid]
            k_acc = keyword_per_pair[pid]
            marker = "  " if k_acc >= 1.0 else "**"
            print(f"    {marker} {pid:30s} word={word:10s} model={m_acc:.1f} kw={k_acc:.1f}")

        # Free memory
        del model, tokenizer, loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE: Model DISAMB accuracy stratified by keyword baseline")
    print(f"{'='*60}")
    header = f"{'Model':30s} {'Overall':20s} {'KW-correct':20s} {'KW-incorrect':20s} {'KW-zero':20s}"
    print(header)
    print("-" * len(header))
    for model_name, strat in results.items():
        short = model_name.split("/")[-1]
        print(f"{short:30s} {strat['overall']:20s} {strat['kw_correct']:20s} {strat['kw_incorrect']:20s} {strat['kw_zero']:20s}")

    # Key diagnostic
    print(f"\n{'='*60}")
    print("KEY DIAGNOSTIC: Model accuracy on keyword-INCORRECT pairs")
    print("(If still well above chance [>0.75], strong rebuttal to cue-leakage)")
    print(f"{'='*60}")
    for model_name, strat in results.items():
        short = model_name.split("/")[-1]
        mu, lo, hi = strat["raw_kw_incorrect"]
        above_75 = "YES" if lo > 0.75 else ("MARGINAL" if mu > 0.75 else "NO")
        print(f"  {short:30s} {mu:.3f} [{lo:.3f}, {hi:.3f}]  above 0.75? {above_75}")


if __name__ == "__main__":
    main()
