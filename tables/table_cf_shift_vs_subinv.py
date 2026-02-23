from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


MODEL_ORDER: List[str] = [
    "gpt2",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]

MODEL_DISPLAY: Dict[str, str] = {
    "gpt2": "GPT-2",
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "model",
    "seed",
    "cf_patch_stratum_expected_effect__shift_mean_max_effect",
    "cf_patch_stratum_expected_effect__shift_mean_max_effect_ci_low",
    "cf_patch_stratum_expected_effect__shift_mean_max_effect_ci_high",
    "cf_patch_stratum_expected_effect__invariant_mean_max_effect",
    "cf_patch_stratum_expected_effect__invariant_mean_max_effect_ci_low",
    "cf_patch_stratum_expected_effect__invariant_mean_max_effect_ci_high",
    "cf_patch_comparison_expected_effect_shift_vs_invariant_mean_diff",
    "cf_patch_comparison_expected_effect_shift_vs_invariant_cohens_d",
    "cf_patch_comparison_expected_effect_shift_vs_invariant_n_a",
    "cf_patch_comparison_expected_effect_shift_vs_invariant_n_b",
    "cf_patch_mean_sham_max_effect",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render a publication table: CF patching effects on shift items vs substitution-type invariant items "
            "(with CIs, difference, Cohen's d, sham baseline)."
        )
    )
    p.add_argument("--in_csv", type=str, required=True)
    p.add_argument("--out_md", type=str, default="")
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--out_tex", type=str, default="")
    p.add_argument("--digits", type=int, default=3)
    return p.parse_args()


def _as_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _model_key(model_name: str) -> tuple[int, str]:
    if model_name in MODEL_ORDER:
        return (MODEL_ORDER.index(model_name), model_name)
    return (len(MODEL_ORDER), model_name.lower())


def _fmt(x: float, digits: int) -> str:
    return f"{float(x):.{int(digits)}f}"


def _fmt_ci(mean: float, lo: float, hi: float, digits: int) -> str:
    return f"{_fmt(mean, digits)} [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _require_single_row(rows: List[Dict[str, str]], *, model: str) -> Dict[str, str]:
    if len(rows) != 1:
        raise ValueError(
            f"Found {len(rows)} rows for model={model!r}. "
            "This table requires one row per model (single seed), because multi-seed CI pooling is not implemented."
        )
    return rows[0]


def _extract_metric(row: Dict[str, str], key: str, *, model: str) -> float:
    v = _as_float(row.get(key))
    if v is None:
        raise ValueError(f"Missing/non-finite value for {key!r} in model={model!r}, seed={row.get('seed')!r}")
    return float(v)


def _extract_required_count(row: Dict[str, str], key: str, *, model: str) -> int:
    v = _as_float(row.get(key))
    if v is None:
        raise ValueError(f"Missing/non-finite value for {key!r} in model={model!r}")
    return int(v)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing!r}")
        for row in r:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})
    if not rows:
        raise ValueError(f"Input CSV has no rows: {path}")
    return rows


def main() -> None:
    args = parse_args()
    in_csv = Path(str(args.in_csv))
    rows = _read_rows(in_csv)

    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model", ""))].append(row)

    records: List[Dict[str, object]] = []
    for model in sorted(by_model.keys(), key=_model_key):
        model_rows = by_model[model]
        row = _require_single_row(model_rows, model=model)
        n_a = _extract_required_count(
            row,
            "cf_patch_comparison_expected_effect_shift_vs_invariant_n_a",
            model=model,
        )
        n_b = _extract_required_count(
            row,
            "cf_patch_comparison_expected_effect_shift_vs_invariant_n_b",
            model=model,
        )
        if n_a <= 0 or n_b <= 0:
            raise ValueError(
                f"Comparison counts must be > 0 for model={model!r}; got n_a={n_a}, n_b={n_b}. "
                "This usually means invariant controls were skipped."
            )

        shift_mean = _extract_metric(
            row,
            "cf_patch_stratum_expected_effect__shift_mean_max_effect",
            model=model,
        )
        shift_lo = _extract_metric(
            row,
            "cf_patch_stratum_expected_effect__shift_mean_max_effect_ci_low",
            model=model,
        )
        shift_hi = _extract_metric(
            row,
            "cf_patch_stratum_expected_effect__shift_mean_max_effect_ci_high",
            model=model,
        )

        inv_mean = _extract_metric(
            row,
            "cf_patch_stratum_expected_effect__invariant_mean_max_effect",
            model=model,
        )
        inv_lo = _extract_metric(
            row,
            "cf_patch_stratum_expected_effect__invariant_mean_max_effect_ci_low",
            model=model,
        )
        inv_hi = _extract_metric(
            row,
            "cf_patch_stratum_expected_effect__invariant_mean_max_effect_ci_high",
            model=model,
        )

        diff = _extract_metric(
            row,
            "cf_patch_comparison_expected_effect_shift_vs_invariant_mean_diff",
            model=model,
        )
        cohens_d = _extract_metric(
            row,
            "cf_patch_comparison_expected_effect_shift_vs_invariant_cohens_d",
            model=model,
        )
        sham = _extract_metric(
            row,
            "cf_patch_mean_sham_max_effect",
            model=model,
        )

        records.append(
            {
                "model_raw": model,
                "model": MODEL_DISPLAY.get(model, model),
                "shift_mean": float(shift_mean),
                "shift_ci_low": float(shift_lo),
                "shift_ci_high": float(shift_hi),
                "invariant_mean": float(inv_mean),
                "invariant_ci_low": float(inv_lo),
                "invariant_ci_high": float(inv_hi),
                "shift_minus_invariant_diff": float(diff),
                "cohens_d": float(cohens_d),
                "sham_baseline": float(sham),
                "n_shift": int(n_a),
                "n_invariant": int(n_b),
            }
        )

    digits = int(args.digits)

    md_lines: List[str] = []
    md_lines.append("| Model | Shift mean max effect (CI) | Invariant mean max effect (CI) | Shift-Invariant diff | Cohen's d | Sham baseline |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for rec in records:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(rec["model"]),
                    _fmt_ci(float(rec["shift_mean"]), float(rec["shift_ci_low"]), float(rec["shift_ci_high"]), digits),
                    _fmt_ci(
                        float(rec["invariant_mean"]),
                        float(rec["invariant_ci_low"]),
                        float(rec["invariant_ci_high"]),
                        digits,
                    ),
                    _fmt(float(rec["shift_minus_invariant_diff"]), digits),
                    _fmt(float(rec["cohens_d"]), digits),
                    _fmt(float(rec["sham_baseline"]), digits),
                ]
            )
            + " |"
        )
    md_text = "\n".join(md_lines) + "\n"

    csv_rows: List[Dict[str, str]] = []
    for rec in records:
        csv_rows.append(
            {
                "model": str(rec["model"]),
                "model_raw": str(rec["model_raw"]),
                "shift_mean": _fmt(float(rec["shift_mean"]), digits),
                "shift_ci_low": _fmt(float(rec["shift_ci_low"]), digits),
                "shift_ci_high": _fmt(float(rec["shift_ci_high"]), digits),
                "invariant_mean": _fmt(float(rec["invariant_mean"]), digits),
                "invariant_ci_low": _fmt(float(rec["invariant_ci_low"]), digits),
                "invariant_ci_high": _fmt(float(rec["invariant_ci_high"]), digits),
                "shift_minus_invariant_diff": _fmt(float(rec["shift_minus_invariant_diff"]), digits),
                "cohens_d": _fmt(float(rec["cohens_d"]), digits),
                "sham_baseline": _fmt(float(rec["sham_baseline"]), digits),
                "n_shift": str(int(rec["n_shift"])),
                "n_invariant": str(int(rec["n_invariant"])),
            }
        )

    tex_lines: List[str] = []
    tex_lines.append("\\begin{tabular}{lrrrrr}")
    tex_lines.append("\\toprule")
    tex_lines.append("Model & Shift mean max effect (CI) & Invariant mean max effect (CI) & Shift-Invariant diff & Cohen's d & Sham baseline\\\\")
    tex_lines.append("\\midrule")
    for rec in records:
        tex_lines.append(
            " & ".join(
                [
                    str(rec["model"]).replace("_", "\\_"),
                    _fmt_ci(float(rec["shift_mean"]), float(rec["shift_ci_low"]), float(rec["shift_ci_high"]), digits),
                    _fmt_ci(
                        float(rec["invariant_mean"]),
                        float(rec["invariant_ci_low"]),
                        float(rec["invariant_ci_high"]),
                        digits,
                    ),
                    _fmt(float(rec["shift_minus_invariant_diff"]), digits),
                    _fmt(float(rec["cohens_d"]), digits),
                    _fmt(float(rec["sham_baseline"]), digits),
                ]
            )
            + "\\\\"
        )
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_text = "\n".join(tex_lines) + "\n"

    if str(args.out_md).strip():
        out_md = Path(str(args.out_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md_text, encoding="utf-8")
    else:
        print(md_text, end="")

    if str(args.out_csv).strip():
        out_csv = Path(str(args.out_csv))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "model_raw",
                    "shift_mean",
                    "shift_ci_low",
                    "shift_ci_high",
                    "invariant_mean",
                    "invariant_ci_low",
                    "invariant_ci_high",
                    "shift_minus_invariant_diff",
                    "cohens_d",
                    "sham_baseline",
                    "n_shift",
                    "n_invariant",
                ],
            )
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)

    if str(args.out_tex).strip():
        out_tex = Path(str(args.out_tex))
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(tex_text, encoding="utf-8")


if __name__ == "__main__":
    main()
