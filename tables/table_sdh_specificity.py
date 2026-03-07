from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MODEL_ORDER: tuple[str, ...] = (
    "gpt2",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Meta-Llama-3.1-8B",
)

MODEL_LABELS: dict[str, str] = {
    "gpt2": "GPT‑2 (124M)",
    "Qwen/Qwen2.5-0.5B": "Qwen2.5‑0.5B",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5‑1.5B",
    "Qwen/Qwen2.5-3B": "Qwen2.5‑3B",
    "Qwen/Qwen3-4B": "Qwen3‑4B",
    "meta-llama/Llama-3.2-1B": "Llama‑3.2‑1B",
    "meta-llama/Llama-3.2-3B": "Llama‑3.2‑3B",
    "meta-llama/Meta-Llama-3.1-8B": "Llama‑3.1‑8B",
}

MODEL_N_LAYERS: dict[str, int] = {
    "gpt2": 12,
    "Qwen/Qwen2.5-0.5B": 24,
    "Qwen/Qwen2.5-1.5B": 28,
    "Qwen/Qwen2.5-3B": 36,
    "Qwen/Qwen3-4B": 36,
    "meta-llama/Llama-3.2-1B": 16,
    "meta-llama/Llama-3.2-3B": 28,
    "meta-llama/Meta-Llama-3.1-8B": 32,
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "model",
    "cpt_spec_layer",
    "cpt_spec_mean_signed_target_effect",
    "cpt_spec_mean_signed_target_effect_ci_low",
    "cpt_spec_mean_signed_target_effect_ci_high",
    "cpt_spec_mean_signed_ctrl_effect",
    "cpt_spec_mean_signed_ctrl_effect_ci_low",
    "cpt_spec_mean_signed_ctrl_effect_ci_high",
    "cpt_spec_mean_signed_delta",
    "cpt_spec_mean_signed_delta_ci_low",
    "cpt_spec_mean_signed_delta_ci_high",
    "cpt_spec_win_rate_signed",
    "cpt_spec_win_rate_signed_ci_low",
    "cpt_spec_win_rate_signed_ci_high",
    "cpt_spec_n_directions_target_patched",
    "cpt_spec_n_directions_ctrl_patched_matched_token",
    "cpt_spec_n_directions_ctrl_patched_same_index",
    "cpt_spec_n_directions_ctrl_patched_same_index_relaxed",
)

TITLE = "Table 3. SDH target-minus-nearby-matched-span deltas (fixed layer)"
CAPTION_TEMPLATE = (
    "Table 3. SDH fixed-depth target-minus-nearby-matched-span results at ℓ* = round(0.25 × (L−1)) "
    "under the canonical `matched_token` control, with `position_window=8` and exclusion buffers ±2. "
    "Reported effects are signed donor-directed margin shifts; Δ = E_target − E_ctrl and win rate = P(Δ_i > 0). "
    "N directions per model = {n_directions}. This table reports one control-selection path and should not be read "
    "as a strategy-comparison table. [E5a, E5b, E5c, E5d, E5e, E5f, C4]"
)


@dataclass(frozen=True)
class SdhRow:
    model_id: str
    model_label: str
    n_layers: int
    fixed_layer: int
    target_mean: float
    target_ci_low: float
    target_ci_high: float
    ctrl_mean: float
    ctrl_ci_low: float
    ctrl_ci_high: float
    delta_mean: float
    delta_ci_low: float
    delta_ci_high: float
    win_rate_mean: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    n_target_patched: int
    n_ctrl_matched_token: int
    n_ctrl_same_index: int
    n_ctrl_same_index_relaxed: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Table 3 (SDH specificity) from a specificity CSV.")
    p.add_argument("--in_csv", "--input_csv", dest="in_csv", type=str, required=True)
    p.add_argument("--out_tex", "--output_tex", dest="out_tex", type=str, default="")
    p.add_argument("--out_md", "--output_md", dest="out_md", type=str, default="")
    return p.parse_args()


def _as_float(x: object) -> float:
    if x is None:
        raise ValueError("Expected a numeric value, got None")
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        raise ValueError(f"Expected a finite float, got {x!r}")
    value = float(s)
    if not math.isfinite(value):
        raise ValueError(f"Expected a finite float, got {x!r}")
    return float(value)


def _as_int(x: object) -> int:
    value = _as_float(x)
    if not float(value).is_integer():
        raise ValueError(f"Expected an integer value, got {x!r}")
    return int(value)


def _fmt(value: float, *, signed: bool = False) -> str:
    rounded = round(value, 2)
    if rounded == 0:
        rounded = 0.0
    if signed:
        return f"{rounded:+.2f}"
    return f"{rounded:.2f}"


def _fmt_ci(mean: float, low: float, high: float, *, signed_mean: bool = False) -> str:
    return f"{_fmt(mean, signed=signed_mean)} [{_fmt(low)}, {_fmt(high)}]"


def _escape_tex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [col for col in REQUIRED_COLUMNS if col not in header]
        if missing:
            raise ValueError(f"CSV {path} is missing required columns: {missing!r}")
        return [{str(k): ("" if v is None else str(v).strip()) for k, v in row.items()} for row in reader]


def normalize_rows(rows: Sequence[dict[str, str]]) -> list[SdhRow]:
    if not rows:
        raise ValueError("No SDH rows found")

    seen_models: set[str] = set()
    normalized: list[SdhRow] = []
    for row in rows:
        model_id = row["model"]
        if model_id not in MODEL_LABELS:
            raise ValueError(f"Unsupported model for SDH table rendering: {model_id!r}")
        if model_id in seen_models:
            raise ValueError(f"Duplicate SDH row for model {model_id!r}")
        seen_models.add(model_id)
        normalized.append(
            SdhRow(
                model_id=model_id,
                model_label=MODEL_LABELS[model_id],
                n_layers=MODEL_N_LAYERS[model_id],
                fixed_layer=_as_int(row["cpt_spec_layer"]),
                target_mean=_as_float(row["cpt_spec_mean_signed_target_effect"]),
                target_ci_low=_as_float(row["cpt_spec_mean_signed_target_effect_ci_low"]),
                target_ci_high=_as_float(row["cpt_spec_mean_signed_target_effect_ci_high"]),
                ctrl_mean=_as_float(row["cpt_spec_mean_signed_ctrl_effect"]),
                ctrl_ci_low=_as_float(row["cpt_spec_mean_signed_ctrl_effect_ci_low"]),
                ctrl_ci_high=_as_float(row["cpt_spec_mean_signed_ctrl_effect_ci_high"]),
                delta_mean=_as_float(row["cpt_spec_mean_signed_delta"]),
                delta_ci_low=_as_float(row["cpt_spec_mean_signed_delta_ci_low"]),
                delta_ci_high=_as_float(row["cpt_spec_mean_signed_delta_ci_high"]),
                win_rate_mean=_as_float(row["cpt_spec_win_rate_signed"]),
                win_rate_ci_low=_as_float(row["cpt_spec_win_rate_signed_ci_low"]),
                win_rate_ci_high=_as_float(row["cpt_spec_win_rate_signed_ci_high"]),
                n_target_patched=_as_int(row["cpt_spec_n_directions_target_patched"]),
                n_ctrl_matched_token=_as_int(row["cpt_spec_n_directions_ctrl_patched_matched_token"]),
                n_ctrl_same_index=_as_int(row["cpt_spec_n_directions_ctrl_patched_same_index"]),
                n_ctrl_same_index_relaxed=_as_int(row["cpt_spec_n_directions_ctrl_patched_same_index_relaxed"]),
            )
        )

    target_counts = {row.n_target_patched for row in normalized}
    if len(target_counts) != 1:
        raise ValueError(f"Target-patched counts are not constant across rows: {sorted(target_counts)!r}")

    if any(row.n_ctrl_same_index != 0 or row.n_ctrl_same_index_relaxed != 0 for row in normalized):
        raise ValueError(
            "SDH strategy counts contradict the matched_token-only caption assumption: "
            "same_index or same_index_relaxed counts are nonzero."
        )

    order_index = {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    return sorted(normalized, key=lambda row: order_index.get(row.model_id, len(MODEL_ORDER)))


def build_caption(rows: Sequence[SdhRow]) -> str:
    if not rows:
        raise ValueError("Cannot build an SDH caption without rows")
    n_directions = rows[0].n_target_patched
    return CAPTION_TEMPLATE.format(n_directions=n_directions)


def render_markdown(rows: Sequence[SdhRow]) -> str:
    caption = build_caption(rows)
    lines = [
        TITLE,
        caption,
        "| Model | Layers | ℓ* | E_target (CI) | E_ctrl, matched span (CI) | Δ = E_target − E_ctrl (CI) | Win rate (CI) |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model_label,
                    str(row.n_layers),
                    str(row.fixed_layer),
                    _fmt_ci(row.target_mean, row.target_ci_low, row.target_ci_high),
                    _fmt_ci(row.ctrl_mean, row.ctrl_ci_low, row.ctrl_ci_high),
                    _fmt_ci(row.delta_mean, row.delta_ci_low, row.delta_ci_high, signed_mean=True),
                    _fmt_ci(row.win_rate_mean, row.win_rate_ci_low, row.win_rate_ci_high),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_tex(rows: Sequence[SdhRow]) -> str:
    lines = [
        "\\begin{tabular}{lrrllll}",
        "\\toprule",
        "Model & Layers & $\\ell^{*}$ & $E_{\\mathrm{target}}$ (CI) & $E_{\\mathrm{ctrl}}$, matched span (CI) & $\\Delta = E_{\\mathrm{target}} - E_{\\mathrm{ctrl}}$ (CI) & Win rate (CI)\\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _escape_tex(row.model_label),
                    str(row.n_layers),
                    str(row.fixed_layer),
                    _escape_tex(_fmt_ci(row.target_mean, row.target_ci_low, row.target_ci_high)),
                    _escape_tex(_fmt_ci(row.ctrl_mean, row.ctrl_ci_low, row.ctrl_ci_high)),
                    _escape_tex(_fmt_ci(row.delta_mean, row.delta_ci_low, row.delta_ci_high, signed_mean=True)),
                    _escape_tex(_fmt_ci(row.win_rate_mean, row.win_rate_ci_low, row.win_rate_ci_high)),
                ]
            )
            + "\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def generate_outputs(csv_path: Path) -> tuple[str, str]:
    rows = normalize_rows(load_rows(csv_path))
    return render_tex(rows), render_markdown(rows)


def main() -> None:
    args = parse_args()
    csv_path = Path(str(args.in_csv))
    tex_text, md_text = generate_outputs(csv_path)

    if str(args.out_tex).strip():
        out_tex = Path(str(args.out_tex))
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(tex_text, encoding="utf-8")
    if str(args.out_md).strip():
        out_md = Path(str(args.out_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md_text, encoding="utf-8")
    if not str(args.out_tex).strip() and not str(args.out_md).strip():
        print(md_text, end="")


if __name__ == "__main__":
    main()
