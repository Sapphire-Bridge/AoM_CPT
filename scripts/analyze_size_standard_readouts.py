#!/usr/bin/env python3
"""Canonical receiver-level analysis for Size-Standard readouts and traces."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


CONFLICT_PRED_COL = "gate_pred_label"
RAW_CONFLICT_PRED_COL = "pred_label"
DETERMINACY_CORRECTED_COL = "determinacy_pred_label_corrected"
DETERMINACY_RAW_COL = "determinacy_pred_label"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
            rows.append(obj)
    return rows


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: str | Path, obj: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _truthy(value: Any) -> bool:
    text = _clean(value).lower()
    return text in {"1", "true", "t", "yes", "y"}


def _float_or_none(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _mean_optional(values: Iterable[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return float(mean(xs))


def _item_id(row: Mapping[str, Any]) -> str:
    return _clean(row.get("item_id")) or _clean(row.get("case_id"))


def _filtered_item_ids(filtered_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {_item_id(row) for row in filtered_rows if _item_id(row)}


def _is_kept(row: Mapping[str, Any], filtered_ids: set[str] | None = None) -> bool:
    if "kept" in row:
        return _truthy(row.get("kept"))
    if filtered_ids:
        item_id = _item_id(row)
        if item_id:
            return item_id in filtered_ids
    return True


def load_analysis_rows(filtered_jsonl: str | Path, margins_csv: str | Path) -> tuple[list[dict[str, Any]], set[str]]:
    filtered_rows = _read_jsonl(filtered_jsonl)
    filtered_ids = _filtered_item_ids(filtered_rows)
    rows = _read_csv(margins_csv)
    return rows, filtered_ids


def _receiver_id(row: Mapping[str, Any], idx: int) -> str:
    for key in ("conflict_receiver_id", "receiver_group_id", "receiver_id", "item_id", "case_id"):
        value = _clean(row.get(key))
        if value:
            return value
    return f"row_{idx}"


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def _unique_nonempty(rows: Sequence[Mapping[str, Any]], col: str) -> list[str]:
    return sorted({_clean(row.get(col)) for row in rows if _clean(row.get(col))})


def _direction_label_roles(direction: str) -> dict[str, str]:
    if direction == "class_small_standard_large":
        return {"small": "class", "large": "standard"}
    if direction == "class_large_standard_small":
        return {"large": "class", "small": "standard"}
    if direction == "standard_small_class_large":
        return {"small": "standard", "large": "class"}
    if direction == "standard_large_class_small":
        return {"large": "standard", "small": "class"}
    return {}


def _winner_from_prediction(row: Mapping[str, Any], prediction: str) -> str:
    direction = _clean(row.get("conflict_direction"))
    roles = _direction_label_roles(direction)
    return roles.get(str(prediction), str(prediction))


def _conflict_rows(rows: Sequence[Mapping[str, Any]], filtered_ids: set[str] | None) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for row in rows:
        if not _is_kept(row, filtered_ids):
            continue
        if _clean(row.get("family")) and _clean(row.get("family")) != "conflict_swap":
            continue
        if not (_clean(row.get("conflict_receiver_id")) or _clean(row.get("conflict_direction"))):
            continue
        # Margins are side-level; the conflict receiver prediction is the base
        # conflict prompt. Sibling rows are only gate-validity checks.
        if "side" in row and _clean(row.get("side")) != "base":
            continue
        out.append(row)
    return out


def summarize_conflict_winners(
    rows: Sequence[Mapping[str, Any]],
    *,
    filtered_ids: set[str] | None = None,
    pred_col: str = CONFLICT_PRED_COL,
    raw_pred_col: str = RAW_CONFLICT_PRED_COL,
) -> dict[str, Any]:
    conflict_rows = _conflict_rows(rows, filtered_ids)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(conflict_rows):
        groups[_receiver_id(row, idx)].append(row)

    receiver_rows: list[dict[str, Any]] = []
    corrected_winners: Counter[str] = Counter()
    raw_winners: Counter[str] = Counter()
    patch_case_corrected: Counter[str] = Counter()
    patch_case_raw: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    direction_winners: Counter[str] = Counter()
    residual_small_hang = 0

    for row in conflict_rows:
        pred = _clean(row.get(pred_col))
        raw = _clean(row.get(raw_pred_col))
        if pred:
            patch_case_corrected[_winner_from_prediction(row, pred)] += 1
        if raw:
            patch_case_raw[_winner_from_prediction(row, raw)] += 1

    for receiver, group in sorted(groups.items()):
        predictions = _unique_nonempty(group, pred_col)
        if not predictions:
            raise ValueError(f"Missing {pred_col!r} for conflict receiver {receiver!r}")
        if len(predictions) > 1:
            raise ValueError(f"Conflicting {pred_col!r} values for receiver {receiver!r}: {predictions!r}")
        raw_predictions = _unique_nonempty(group, raw_pred_col)
        if len(raw_predictions) > 1:
            raise ValueError(f"Conflicting {raw_pred_col!r} values for receiver {receiver!r}: {raw_predictions!r}")
        directions = _unique_nonempty(group, "conflict_direction")
        direction = directions[0] if len(directions) == 1 else ("mixed" if directions else "unknown")
        corrected = predictions[0]
        raw = raw_predictions[0] if raw_predictions else ""
        exemplar = group[0]
        corrected_winner = _winner_from_prediction(exemplar, corrected)
        raw_winner = _winner_from_prediction(exemplar, raw) if raw else ""

        corrected_winners[corrected_winner] += 1
        if raw_winner:
            raw_winners[raw_winner] += 1
        direction_counts[direction] += 1
        direction_winners[f"{direction}::{corrected_winner}"] += 1
        if direction == "class_small_standard_large" and corrected_winner == "class":
            residual_small_hang += 1
        receiver_rows.append(
            {
                "conflict_receiver_id": receiver,
                "conflict_direction": direction,
                "gate_pred_label": corrected,
                "gate_winner": corrected_winner,
                "pred_label": raw,
                "raw_winner": raw_winner,
                "n_patch_cases": len(group),
            }
        )

    return {
        "n_patch_cases": len(conflict_rows),
        "n_receivers": len(groups),
        "receiver_counts_corrected": _counter_dict(corrected_winners),
        "receiver_counts_raw": _counter_dict(raw_winners),
        "patch_case_counts_corrected_diagnostic": _counter_dict(patch_case_corrected),
        "patch_case_counts_raw_diagnostic": _counter_dict(patch_case_raw),
        "receiver_counts_by_conflict_direction": _counter_dict(direction_counts),
        "receiver_prediction_counts_by_direction": _counter_dict(direction_winners),
        "residual_small_hang_class_wins": int(residual_small_hang),
        "receiver_rows": receiver_rows,
    }


def _prior_bin(row: Mapping[str, Any]) -> str:
    value = _float_or_none(row.get("abs_log_ratio_to_prior"))
    if value is None:
        value = _float_or_none(row.get("abs_log_ratio"))
    if value is None:
        return "missing"
    if value < 0.5:
        return "0-0.5"
    if value < 1.0:
        return "0.5-1"
    if value < 2.0:
        return "1-2"
    return ">=2"


def summarize_determinacy(
    rows: Sequence[Mapping[str, Any]],
    *,
    filtered_ids: set[str] | None = None,
) -> dict[str, Any]:
    has_corrected = any(_clean(row.get(DETERMINACY_CORRECTED_COL)) for row in rows)
    corrected_col = DETERMINACY_CORRECTED_COL if has_corrected else DETERMINACY_RAW_COL
    det_rows = [
        row
        for row in rows
        if _is_kept(row, filtered_ids)
        and (_clean(row.get(corrected_col)) or _clean(row.get(DETERMINACY_RAW_COL)))
    ]

    def counts(col: str) -> Counter[str]:
        return Counter(_clean(row.get(col)) for row in det_rows if _clean(row.get(col)))

    strata_rows: list[dict[str, Any]] = []
    labels = ["def_small", "borderline", "def_large"]
    specs = [
        ("family", lambda r: _clean(r.get("family")) or "missing"),
        ("class_name", lambda r: _clean(r.get("class_name")) or "missing"),
        ("template_id", lambda r: _clean(r.get("template_id")) or "missing"),
        ("abs_log_ratio_to_prior_bin", _prior_bin),
    ]
    for stratum, getter in specs:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in det_rows:
            grouped[str(getter(row))].append(row)
        for value, group in sorted(grouped.items()):
            raw_counts = Counter(_clean(row.get(DETERMINACY_RAW_COL)) for row in group if _clean(row.get(DETERMINACY_RAW_COL)))
            corrected_counts = Counter(_clean(row.get(corrected_col)) for row in group if _clean(row.get(corrected_col)))
            out: dict[str, Any] = {"stratum": stratum, "value": value, "n": len(group)}
            for label in labels:
                out[f"raw_{label}"] = int(raw_counts.get(label, 0))
                out[f"corrected_{label}"] = int(corrected_counts.get(label, 0))
            strata_rows.append(out)

    return {
        "n": len(det_rows),
        "corrected_column": corrected_col,
        "counts_raw": _counter_dict(counts(DETERMINACY_RAW_COL)),
        "counts_corrected": _counter_dict(counts(corrected_col)),
        "mean_borderline_margin_raw": _mean_optional(
            _float_or_none(row.get("determinacy_borderline_margin")) for row in det_rows
        ),
        "mean_borderline_margin_corrected": _mean_optional(
            _float_or_none(row.get("determinacy_borderline_margin_corrected")) for row in det_rows
        ),
        "strata_rows": strata_rows,
    }


def _span(row: Mapping[str, Any]) -> str:
    return _clean(row.get("patch_span")) or _clean(row.get("span")) or "unknown"


def _layer(row: Mapping[str, Any]) -> str:
    return _clean(row.get("layer")) or _clean(row.get("patch_layer")) or "unknown"


def _layer_sort(layer: str) -> tuple[int, str]:
    try:
        return (int(layer), "")
    except ValueError:
        return (10**9, layer)


def summarize_trace(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        receiver = _receiver_id(row, idx)
        cells[(receiver, _span(row), _layer(row))].append(row)

    receiver_cells: list[dict[str, Any]] = []
    for (receiver, span, layer), group in sorted(cells.items()):
        receiver_cells.append(
            {
                "conflict_receiver_id": receiver,
                "patch_span": span,
                "layer": layer,
                "effect": _mean_optional(_float_or_none(row.get("effect")) for row in group),
                "sham_effect": _mean_optional(_float_or_none(row.get("sham_effect")) for row in group),
                "flip_rate": _mean_optional(_float_or_none(row.get("flip")) for row in group),
                "n_patch_cases": len(group),
            }
        )

    layer_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in receiver_cells:
        layer_groups[(str(cell["patch_span"]), str(cell["layer"]))].append(cell)

    layer_rows: list[dict[str, Any]] = []
    for (span, layer), group in sorted(layer_groups.items(), key=lambda kv: (kv[0][0], _layer_sort(kv[0][1]))):
        layer_rows.append(
            {
                "patch_span": span,
                "layer": layer,
                "mean_effect": _mean_optional(_float_or_none(row.get("effect")) for row in group),
                "mean_sham_effect": _mean_optional(_float_or_none(row.get("sham_effect")) for row in group),
                "flip_rate": _mean_optional(_float_or_none(row.get("flip_rate")) for row in group),
                "n_receivers": len({_clean(row.get("conflict_receiver_id")) for row in group}),
            }
        )

    max_by_receiver_span: dict[tuple[str, str], float] = {}
    for cell in receiver_cells:
        effect = _float_or_none(cell.get("effect"))
        if effect is None:
            continue
        key = (str(cell["conflict_receiver_id"]), str(cell["patch_span"]))
        if key not in max_by_receiver_span or effect > max_by_receiver_span[key]:
            max_by_receiver_span[key] = effect

    span_rows: list[dict[str, Any]] = []
    by_span: dict[str, list[float]] = defaultdict(list)
    for (_receiver, span), effect in max_by_receiver_span.items():
        by_span[span].append(effect)
    for span, values in sorted(by_span.items()):
        span_rows.append(
            {
                "patch_span": span,
                "n_receivers": len(values),
                "mean_receiver_max_effect": float(mean(values)) if values else None,
            }
        )

    paired: list[float] = []
    receivers = sorted({receiver for receiver, _span_name in max_by_receiver_span})
    for receiver in receivers:
        standard = max_by_receiver_span.get((receiver, "standard_span"))
        class_effect = max_by_receiver_span.get((receiver, "class_span"))
        if standard is not None and class_effect is not None:
            paired.append(float(standard - class_effect))

    return {
        "n_trace_rows": len(rows),
        "n_receiver_layer_span_cells": len(receiver_cells),
        "layer_rows": layer_rows,
        "span_comparison_rows": span_rows,
        "paired_standard_vs_class": {
            "n_paired_receivers": len(paired),
            "mean_standard_minus_class_max_effect": float(mean(paired)) if paired else None,
        },
    }


def conflict_count_rows(conflict: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, key in (
        ("receiver_corrected", "receiver_counts_corrected"),
        ("receiver_raw", "receiver_counts_raw"),
        ("patch_case_corrected_diagnostic", "patch_case_counts_corrected_diagnostic"),
        ("patch_case_raw_diagnostic", "patch_case_counts_raw_diagnostic"),
        ("receiver_by_conflict_direction", "receiver_counts_by_conflict_direction"),
    ):
        counts = conflict.get(key, {})
        if isinstance(counts, Mapping):
            for value, n in sorted(counts.items()):
                rows.append({"scope": scope, "value": value, "n": n})
    direction_counts = conflict.get("receiver_prediction_counts_by_direction", {})
    if isinstance(direction_counts, Mapping):
        for key, n in sorted(direction_counts.items()):
            direction, _, value = str(key).partition("::")
            rows.append(
                {
                    "scope": "receiver_corrected_by_direction",
                    "conflict_direction": direction,
                    "value": value,
                    "n": n,
                }
            )
    return rows


def determinacy_count_rows(determinacy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, key in (("raw", "counts_raw"), ("corrected", "counts_corrected")):
        counts = determinacy.get(key, {})
        if isinstance(counts, Mapping):
            for value, n in sorted(counts.items()):
                rows.append({"scope": scope, "value": value, "n": n})
    return rows


def write_outputs(
    *,
    filtered_jsonl: str | Path,
    margins_csv: str | Path,
    out_prefix: str | Path,
    trace_csv: str | Path | None = None,
) -> dict[str, Any]:
    rows, filtered_ids = load_analysis_rows(filtered_jsonl, margins_csv)
    conflict = summarize_conflict_winners(rows, filtered_ids=filtered_ids)
    determinacy = summarize_determinacy(rows, filtered_ids=filtered_ids)
    summary: dict[str, Any] = {
        "interpretation_rules": {
            "conflict_count_unit": "receiver",
            "conflict_rows": "kept base rows only; sibling rows are gate diagnostics",
            "conflict_prediction_column": CONFLICT_PRED_COL,
            "patch_case_counts": "diagnostic_only",
            "determinacy_prediction_column": determinacy["corrected_column"],
            "class_small_standard_large_class_wins": "residual_small_hang",
            "conflict_expectation": "open_by_direction; correction may uncover or remove class wins",
        },
        "n_margin_rows": len(rows),
        "n_filtered_items": len(filtered_ids),
        "conflict": {k: v for k, v in conflict.items() if k != "receiver_rows"},
        "determinacy": {k: v for k, v in determinacy.items() if k != "strata_rows"},
    }

    prefix = Path(out_prefix)
    _write_json(f"{prefix}.summary.json", summary)
    _write_csv(f"{prefix}.conflict_receivers.csv", conflict["receiver_rows"])
    _write_csv(f"{prefix}.conflict_counts.csv", conflict_count_rows(conflict))
    _write_csv(f"{prefix}.determinacy_counts.csv", determinacy_count_rows(determinacy))
    _write_csv(f"{prefix}.determinacy_strata.csv", determinacy["strata_rows"])

    if trace_csv is not None:
        trace = summarize_trace(_read_csv(trace_csv))
        summary["trace"] = {k: v for k, v in trace.items() if not k.endswith("_rows")}
        _write_json(f"{prefix}.summary.json", summary)
        _write_csv(f"{prefix}.trace_layers.csv", trace["layer_rows"])
        _write_csv(f"{prefix}.trace_span_comparison.csv", trace["span_comparison_rows"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Size-Standard readouts at receiver level.")
    parser.add_argument("--filtered_jsonl", required=True)
    parser.add_argument("--margins_csv", required=True)
    parser.add_argument("--trace_csv", default=None)
    parser.add_argument("--out_prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_outputs(
        filtered_jsonl=args.filtered_jsonl,
        margins_csv=args.margins_csv,
        trace_csv=args.trace_csv,
        out_prefix=args.out_prefix,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
