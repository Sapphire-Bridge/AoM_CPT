from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

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

COH_FILES: Tuple[str, ...] = (
    "coh_patching.csv",
    "coh_patching_qwen.csv",
    "coh_patching_qwen15.csv",
    "coh_patching_qwen3b.csv",
)


@dataclass
class LayerwiseProfile:
    model: str
    layers: List[int]
    effect: List[float]
    sham: List[float]


@dataclass
class EffectWithCI:
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class SpecificityRecord:
    model: str
    target: EffectWithCI
    control: EffectWithCI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create representative figures from canonical AoM table-source CSVs.")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results_submission_full"))
    p.add_argument("--tables_dir", type=str, default=str(ROOT / "tables_submission_full"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "figures_submission_full"))
    p.add_argument("--dpi", type=int, default=300)
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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _extract_layer_series(row: Dict[str, str], prefix: str) -> List[Tuple[int, float]]:
    pairs: List[Tuple[int, float]] = []
    for key, value in row.items():
        if not key.startswith(prefix):
            continue
        idx_str = key.rsplit("_", 1)[-1]
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        v = _as_float(value)
        if v is None:
            continue
        pairs.append((idx, v))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _model_sort_key(model: str) -> Tuple[int, str]:
    if model in MODEL_ORDER:
        return (MODEL_ORDER.index(model), model)
    return (len(MODEL_ORDER), model.lower())


def load_layerwise_profiles(aom_eval_csv: Path) -> List[LayerwiseProfile]:
    rows = _read_csv(aom_eval_csv)
    profiles: List[LayerwiseProfile] = []
    for row in sorted(rows, key=lambda r: _model_sort_key(str(r.get("model", "")))):
        model = str(row.get("model", "")).strip()
        if not model:
            continue
        effect_pairs = _extract_layer_series(row, "cpt_effect_layer_")
        sham_pairs = _extract_layer_series(row, "cpt_sham_effect_layer_")
        if not effect_pairs or not sham_pairs:
            continue
        effect_dict = {k: v for k, v in effect_pairs}
        sham_dict = {k: v for k, v in sham_pairs}
        common_layers = sorted(set(effect_dict.keys()).intersection(sham_dict.keys()))
        if not common_layers:
            continue
        profiles.append(
            LayerwiseProfile(
                model=model,
                layers=common_layers,
                effect=[effect_dict[i] for i in common_layers],
                sham=[sham_dict[i] for i in common_layers],
            )
        )
    if not profiles:
        raise ValueError(f"No layerwise CPT profile columns found in {aom_eval_csv}")
    return profiles


def load_cf_specificity(cf_shift_csv: Path) -> Dict[str, SpecificityRecord]:
    rows = _read_csv(cf_shift_csv)
    out: Dict[str, SpecificityRecord] = {}
    for row in rows:
        model_raw = str(row.get("model_raw", "")).strip()
        model = model_raw or str(row.get("model", "")).strip()
        if not model:
            continue
        shift_mean = _as_float(
            row.get("cf_patch_stratum_expected_effect__shift_mean_max_effect") or row.get("shift_mean")
        )
        shift_lo = _as_float(
            row.get("cf_patch_stratum_expected_effect__shift_mean_max_effect_ci_low") or row.get("shift_ci_low")
        )
        shift_hi = _as_float(
            row.get("cf_patch_stratum_expected_effect__shift_mean_max_effect_ci_high") or row.get("shift_ci_high")
        )
        inv_mean = _as_float(
            row.get("cf_patch_stratum_expected_effect__invariant_mean_max_effect") or row.get("invariant_mean")
        )
        inv_lo = _as_float(
            row.get("cf_patch_stratum_expected_effect__invariant_mean_max_effect_ci_low") or row.get("invariant_ci_low")
        )
        inv_hi = _as_float(
            row.get("cf_patch_stratum_expected_effect__invariant_mean_max_effect_ci_high") or row.get("invariant_ci_high")
        )
        if None in (shift_mean, shift_lo, shift_hi, inv_mean, inv_lo, inv_hi):
            continue
        out[model] = SpecificityRecord(
            model=model,
            target=EffectWithCI(mean=float(shift_mean), ci_low=float(shift_lo), ci_high=float(shift_hi)),
            control=EffectWithCI(mean=float(inv_mean), ci_low=float(inv_lo), ci_high=float(inv_hi)),
        )
    if not out:
        raise ValueError(f"No usable CF specificity rows in {cf_shift_csv}")
    return out


def load_coh_specificity(results_dir: Path) -> Dict[str, SpecificityRecord]:
    out: Dict[str, SpecificityRecord] = {}
    for fn in COH_FILES:
        path = results_dir / fn
        if not path.exists():
            continue
        rows = _read_csv(path)
        for row in rows:
            model = str(row.get("model", "")).strip()
            if not model:
                continue
            rel_mean = _as_float(row.get("coh_patch_stratum_condition__constraint_span_mean_max_effect"))
            rel_lo = _as_float(row.get("coh_patch_stratum_condition__constraint_span_mean_max_effect_ci_low"))
            rel_hi = _as_float(row.get("coh_patch_stratum_condition__constraint_span_mean_max_effect_ci_high"))
            irr_mean = _as_float(row.get("coh_patch_stratum_condition__irrelevant_span_mean_max_effect"))
            irr_lo = _as_float(row.get("coh_patch_stratum_condition__irrelevant_span_mean_max_effect_ci_low"))
            irr_hi = _as_float(row.get("coh_patch_stratum_condition__irrelevant_span_mean_max_effect_ci_high"))
            if None in (rel_mean, rel_lo, rel_hi, irr_mean, irr_lo, irr_hi):
                continue
            out[model] = SpecificityRecord(
                model=model,
                target=EffectWithCI(mean=float(rel_mean), ci_low=float(rel_lo), ci_high=float(rel_hi)),
                control=EffectWithCI(mean=float(irr_mean), ci_low=float(irr_lo), ci_high=float(irr_hi)),
            )
    if not out:
        raise ValueError(f"No usable COH specificity rows found in {results_dir}")
    return out


def _set_clean_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_layerwise_profiles(profiles: Iterable[LayerwiseProfile], out_dir: Path, dpi: int) -> None:
    profiles_list = list(profiles)
    n_panels = len(profiles_list)
    n_cols = 2
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]  # type: ignore[arg-type]

    y_vals: List[float] = []
    for prof in profiles_list:
        y_vals.extend(prof.effect)
        y_vals.extend(prof.sham)
    y_min = min(y_vals) if y_vals else -0.1
    y_max = max(y_vals) if y_vals else 1.0
    pad = max(0.08, 0.08 * (y_max - y_min))

    for ax, prof in zip(axes_list, profiles_list):
        L = len(prof.layers)
        if L <= 1:
            continue
        depth = [i / float(L - 1) for i in range(L)]
        peak_idx = max(range(L), key=lambda i: prof.effect[i])
        peak_layer = prof.layers[peak_idx]
        peak_depth = depth[peak_idx]

        ax.plot(depth, prof.effect, color="#1f77b4", linewidth=2.2, label="Donor-directed effect")
        ax.plot(depth, prof.sham, color="#ff7f0e", linewidth=1.6, linestyle="--", label="Sham control")
        ax.scatter([peak_depth], [prof.effect[peak_idx]], color="#1f77b4", s=36, zorder=4)
        ax.annotate(
            f"peak L={peak_layer}",
            xy=(peak_depth, prof.effect[peak_idx]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color="#1f77b4",
        )

        for frac in (0.25, 0.50, 0.75):
            ax.axvline(frac, color="0.75", linestyle=":", linewidth=0.8, alpha=0.75)
        ax.axhline(0.0, color="0.45", linewidth=0.9, alpha=0.75)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(MODEL_DISPLAY.get(prof.model, prof.model), fontsize=11, pad=6)
        ax.set_xlabel("Relative layer depth")
        _set_clean_axes(ax)

    for ax in axes_list[n_panels:]:
        ax.set_visible(False)

    if axes_list:
        axes_list[0].set_ylabel("Mean donor-directed effect")
        if len(axes_list) > 1:
            axes_list[1].legend(loc="upper right", fontsize=9, frameon=False)

    fig.suptitle("Figure 1: CPT Layerwise Donor-Directed Effect Profiles", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure1_cpt_layerwise_profiles.png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_dir / "figure1_cpt_layerwise_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def _error_bars(rec: EffectWithCI) -> List[List[float]]:
    return [[max(0.0, rec.mean - rec.ci_low)], [max(0.0, rec.ci_high - rec.mean)]]


def plot_specificity_panels(
    cf_by_model: Dict[str, SpecificityRecord],
    coh_by_model: Dict[str, SpecificityRecord],
    out_dir: Path,
    dpi: int,
) -> None:
    models = [m for m in MODEL_ORDER if m in cf_by_model and m in coh_by_model]
    if not models:
        raise ValueError("No overlapping models across CF and COH specificity summaries.")

    x = list(range(len(models)))
    off = 0.14
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    cf_ax, coh_ax = axes
    for i, model in enumerate(models):
        rec = cf_by_model[model]
        cf_ax.errorbar(
            i - off,
            rec.target.mean,
            yerr=_error_bars(rec.target),
            fmt="o",
            color="#1f77b4",
            markersize=6,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )
        cf_ax.errorbar(
            i + off,
            rec.control.mean,
            yerr=_error_bars(rec.control),
            fmt="s",
            color="#ff7f0e",
            markersize=5.5,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )

    for i, model in enumerate(models):
        rec = coh_by_model[model]
        coh_ax.errorbar(
            i - off,
            rec.target.mean,
            yerr=_error_bars(rec.target),
            fmt="o",
            color="#1f77b4",
            markersize=6,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )
        coh_ax.errorbar(
            i + off,
            rec.control.mean,
            yerr=_error_bars(rec.control),
            fmt="s",
            color="#ff7f0e",
            markersize=5.5,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )

    for ax in axes:
        ax.axhline(0.0, color="0.45", linewidth=0.9, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=15, ha="right")
        _set_clean_axes(ax)

    cf_ax.set_title("CF: Shift vs Invariant (Table 4)", fontsize=11)
    coh_ax.set_title("COH: Constraint vs Irrelevant (Table 4)", fontsize=11)
    cf_ax.set_ylabel("Mean max effect (95% CI)")

    cf_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="#1f77b4",
            markersize=6,
            label="Shift (meaning-altering)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="#ff7f0e",
            markersize=6,
            label="Invariant (label-preserving)",
        ),
    ]
    coh_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="#1f77b4",
            markersize=6,
            label="Constraint span",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="#ff7f0e",
            markersize=6,
            label="Irrelevant span",
        ),
    ]
    cf_ax.legend(handles=cf_handles, loc="upper left", frameon=False, fontsize=9)
    coh_ax.legend(handles=coh_handles, loc="upper right", frameon=False, fontsize=9)

    fig.suptitle("Figure 2: Causal Specificity Contrasts Across Models", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure2_causal_specificity_contrasts.png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_dir / "figure2_causal_specificity_contrasts.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(str(args.results_dir))
    tables_dir = Path(str(args.tables_dir))
    out_dir = Path(str(args.out_dir))

    aom_eval_csv = results_dir / "aom_eval.csv"
    cf_shift_csv = tables_dir / "cf_patching_shift_vs_subinv.csv"

    if not aom_eval_csv.exists():
        raise FileNotFoundError(f"Missing required input: {aom_eval_csv}")
    if not cf_shift_csv.exists():
        raise FileNotFoundError(f"Missing required input: {cf_shift_csv}")

    profiles = load_layerwise_profiles(aom_eval_csv)
    cf_by_model = load_cf_specificity(cf_shift_csv)
    coh_by_model = load_coh_specificity(results_dir)

    plot_layerwise_profiles(profiles, out_dir=out_dir, dpi=int(args.dpi))
    plot_specificity_panels(cf_by_model, coh_by_model, out_dir=out_dir, dpi=int(args.dpi))
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
