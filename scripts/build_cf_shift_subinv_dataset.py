#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.bundle_manifest import compute_bundle_id
from aom.data.dataset_manifest import sha256_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a counterfactual dataset for CF patching that combines "
            "all shift items from a base CF set with substitution-type invariant controls."
        )
    )
    p.add_argument(
        "--base_cf_path",
        type=str,
        default=str(ROOT / "data_paper_hardened_v2" / "counterfactual.jsonl"),
        help="Base CF dataset containing shift items.",
    )
    p.add_argument(
        "--subinv_cf_path",
        type=str,
        default=str(ROOT / "data_paper_hardened_v2" / "counterfactual_invariant_sub.jsonl"),
        help="Substitution-invariant CF dataset (expected_effect=invariant).",
    )
    p.add_argument(
        "--out_cf_path",
        type=str,
        default=str(ROOT / "data_paper_hardened_v2" / "counterfactual_shift_plus_subinv.jsonl"),
        help="Output merged CF JSONL path.",
    )
    p.add_argument(
        "--out_manifest_path",
        type=str,
        default=str(ROOT / "data_paper_hardened_v2" / "DATASET_MANIFEST_cf_shift_plus_subinv.json"),
        help="Output bundle manifest path used by --dataset_manifest_path in patching runs.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="cf_shift_plus_subinv_v1",
        help="Manifest dataset bundle name.",
    )
    p.add_argument(
        "--enforce_counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce expected shift/invariant counts.",
    )
    p.add_argument(
        "--enforce_single_change_block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require substitution-invariant rows to have exactly one contiguous non-empty word-level change block "
            "(prevents multi-locus edits that expand patch spans)."
        ),
    )
    p.add_argument(
        "--shift_intervention_types",
        type=str,
        default="",
        help=(
            "Optional comma-separated shift intervention_type filter (e.g., 'negation,quantifier'). "
            "When set, only matching shift rows are included."
        ),
    )
    p.add_argument(
        "--shift_geometry_filter",
        type=str,
        default="all",
        choices=["all", "single_block_only"],
        help=(
            "Optional geometry filter for shift rows. "
            "'single_block_only' keeps only shift rows with one contiguous non-empty word-level change block."
        ),
    )
    p.add_argument("--expected_shift_count", type=int, default=60, help="Expected number of shift rows from base CF.")
    p.add_argument(
        "--expected_invariant_count",
        type=int,
        default=20,
        help="Expected number of invariant rows from substitution-invariant CF.",
    )
    return p.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _try_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return str(out).strip()
    except Exception:
        return ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel_under_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def _validate_unique_item_ids(rows: list[dict[str, Any]]) -> None:
    ids = [str(r.get("item_id", "")).strip() for r in rows]
    empty = [i for i, v in enumerate(ids) if not v]
    if empty:
        raise ValueError(f"Found {len(empty)} rows with empty item_id.")
    counts = Counter(ids)
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        preview = ", ".join(dups[:8])
        raise ValueError(f"Found duplicated item_id values ({len(dups)} unique duplicates): {preview}")


def _word_level_single_change_block(
    *,
    a_text: str,
    b_text: str,
) -> tuple[bool, str, int, int]:
    a = str(a_text).split()
    b = str(b_text).split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    changed = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if len(changed) != 1:
        return False, f"changed_blocks={len(changed)}", 0, 0
    _tag, i1, i2, j1, j2 = changed[0]
    a_len = int(i2 - i1)
    b_len = int(j2 - j1)
    if a_len <= 0 or b_len <= 0:
        return False, "empty_change_side", a_len, b_len
    return True, "ok", a_len, b_len


def _prompt_pair(row: dict[str, Any]) -> tuple[str, str]:
    base = row.get("base", None)
    cf = row.get("cf", None)
    if not isinstance(base, dict) or not isinstance(cf, dict):
        raise ValueError(f"Row {row.get('item_id', '<missing>')!r} missing base/cf prompt objects")
    a = base.get("prompt", None)
    b = cf.get("prompt", None)
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError(f"Row {row.get('item_id', '<missing>')!r} has non-string prompts")
    return str(a), str(b)


def main() -> None:
    args = parse_args()
    base_cf_path = Path(str(args.base_cf_path))
    subinv_cf_path = Path(str(args.subinv_cf_path))
    out_cf_path = Path(str(args.out_cf_path))
    out_manifest_path = Path(str(args.out_manifest_path))

    if not base_cf_path.exists():
        raise FileNotFoundError(f"Missing base CF dataset: {base_cf_path}")
    if not subinv_cf_path.exists():
        raise FileNotFoundError(f"Missing substitution-invariant CF dataset: {subinv_cf_path}")

    base_rows = _read_jsonl(base_cf_path)
    subinv_rows = _read_jsonl(subinv_cf_path)

    shift_rows_all = [r for r in base_rows if str(r.get("expected_effect", "")).strip() == "shift"]
    shift_types_raw = str(getattr(args, "shift_intervention_types", "") or "").strip()
    shift_type_filter = tuple(s.strip() for s in shift_types_raw.split(",") if s.strip())
    if shift_type_filter:
        shift_type_filter_set = set(shift_type_filter)
        shift_rows_prefilter = [
            r for r in shift_rows_all if str(r.get("intervention_type", "")).strip() in shift_type_filter_set
        ]
    else:
        shift_rows_prefilter = list(shift_rows_all)

    invariant_rows = [r for r in subinv_rows if str(r.get("expected_effect", "")).strip() == "invariant"]
    non_invariant_sub = len(subinv_rows) - len(invariant_rows)
    if non_invariant_sub > 0:
        raise ValueError(
            f"Substitution-invariant source has {non_invariant_sub} non-invariant rows. "
            "All rows in --subinv_cf_path must have expected_effect='invariant'."
        )

    geometry_bad: list[str] = []
    geometry_examples: list[dict[str, Any]] = []
    invariant_a_lens: list[int] = []
    invariant_b_lens: list[int] = []
    shift_single_block_ok = 0
    shift_single_block_bad = 0
    shift_a_lens: list[int] = []
    shift_b_lens: list[int] = []
    shift_rows_single_block: list[dict[str, Any]] = []
    for row in shift_rows_prefilter:
        a_prompt, b_prompt = _prompt_pair(row)
        ok, _reason, a_len, b_len = _word_level_single_change_block(a_text=a_prompt, b_text=b_prompt)
        if ok:
            shift_single_block_ok += 1
            shift_a_lens.append(int(a_len))
            shift_b_lens.append(int(b_len))
            shift_rows_single_block.append(row)
        else:
            shift_single_block_bad += 1

    shift_geometry_filter = str(getattr(args, "shift_geometry_filter", "all"))
    if shift_geometry_filter == "single_block_only":
        shift_rows_used = list(shift_rows_single_block)
    else:
        shift_rows_used = list(shift_rows_prefilter)

    if bool(args.enforce_counts):
        if int(args.expected_shift_count) >= 0 and len(shift_rows_used) != int(args.expected_shift_count):
            raise ValueError(
                f"Shift row count mismatch: got {len(shift_rows_used)}, expected {int(args.expected_shift_count)} "
                f"(after filters) from {base_cf_path}."
            )
        if int(args.expected_invariant_count) >= 0 and len(invariant_rows) != int(args.expected_invariant_count):
            raise ValueError(
                f"Invariant row count mismatch: got {len(invariant_rows)}, expected {int(args.expected_invariant_count)} "
                f"from {subinv_cf_path}."
            )

    for row in invariant_rows:
        item_id = str(row.get("item_id", "")).strip() or "<missing-item-id>"
        a_prompt, b_prompt = _prompt_pair(row)
        ok, reason, a_len, b_len = _word_level_single_change_block(a_text=a_prompt, b_text=b_prompt)
        if not ok:
            geometry_bad.append(item_id)
            if len(geometry_examples) < 10:
                geometry_examples.append(
                    {
                        "item_id": item_id,
                        "reason": reason,
                    }
                )
        else:
            invariant_a_lens.append(int(a_len))
            invariant_b_lens.append(int(b_len))

    if bool(args.enforce_single_change_block) and geometry_bad:
        preview = ", ".join(geometry_bad[:8])
        raise ValueError(
            "Substitution-invariant rows failed single-change-block geometry check: "
            f"{len(geometry_bad)} items (examples: {preview}). "
            "Redesign invariants so each row has exactly one contiguous non-empty change block."
        )

    merged_rows = list(shift_rows_used) + list(invariant_rows)
    _validate_unique_item_ids(merged_rows)
    _write_jsonl(merged_rows, out_cf_path)

    expected_counts = Counter(str(r.get("expected_effect", "")).strip() for r in merged_rows)
    intervention_counts = Counter(str(r.get("intervention_type", "")).strip() for r in merged_rows)

    files = {
        "counterfactual.jsonl": {
            "path": _rel_under_root(out_cf_path),
            "n_lines": int(len(merged_rows)),
            "sha256": sha256_file(out_cf_path),
        }
    }
    bundle_id = compute_bundle_id({"files": files})

    manifest: dict[str, Any] = {
        "name": str(args.name),
        "generated_at_utc": _utc_now_iso(),
        "git_commit": _try_git_commit(),
        "generator": "scripts/build_cf_shift_subinv_dataset.py",
        "bundle_id": str(bundle_id),
        "files": files,
        "construction": {
            "base_cf_path": _rel_under_root(base_cf_path),
            "base_cf_sha256": sha256_file(base_cf_path),
            "base_rows_total": int(len(base_rows)),
            "base_rows_shift_total": int(len(shift_rows_all)),
            "base_rows_shift_after_type_filter": int(len(shift_rows_prefilter)),
            "base_rows_shift_used": int(len(shift_rows_used)),
            "shift_intervention_types_filter": list(shift_type_filter),
            "shift_geometry_filter": str(shift_geometry_filter),
            "subinv_cf_path": _rel_under_root(subinv_cf_path),
            "subinv_cf_sha256": sha256_file(subinv_cf_path),
            "subinv_rows_total": int(len(subinv_rows)),
            "subinv_rows_invariant_used": int(len(invariant_rows)),
        },
        "suite_counts": {
            "cf": {
                "n_items_total": int(len(merged_rows)),
                "expected_effect_counts": {str(k): int(v) for k, v in sorted(expected_counts.items())},
                "intervention_type_counts": {str(k): int(v) for k, v in sorted(intervention_counts.items())},
            }
        },
        "quality_checks": {
            "single_change_block_check": {
                "enabled": bool(args.enforce_single_change_block),
                "shift_rows_total": int(len(shift_rows_prefilter)),
                "shift_rows_single_block_ok": int(shift_single_block_ok),
                "shift_rows_single_block_bad": int(shift_single_block_bad),
                "shift_rows_used": int(len(shift_rows_used)),
                "shift_change_len_a_mean": float(sum(shift_a_lens) / len(shift_a_lens)) if shift_a_lens else None,
                "shift_change_len_b_mean": float(sum(shift_b_lens) / len(shift_b_lens)) if shift_b_lens else None,
                "n_invariant_rows_checked": int(len(invariant_rows)),
                "n_invariant_rows_failed": int(len(geometry_bad)),
                "failed_examples": geometry_examples,
                "invariant_change_len_a_max": max(invariant_a_lens) if invariant_a_lens else None,
                "invariant_change_len_b_max": max(invariant_b_lens) if invariant_b_lens else None,
                "invariant_change_len_a_mean": (
                    float(sum(invariant_a_lens) / len(invariant_a_lens)) if invariant_a_lens else None
                ),
                "invariant_change_len_b_mean": (
                    float(sum(invariant_b_lens) / len(invariant_b_lens)) if invariant_b_lens else None
                ),
            }
        },
    }
    _write_json(out_manifest_path, manifest)

    print(f"[ok] wrote merged CF dataset: {out_cf_path}")
    print(f"[ok] wrote manifest: {out_manifest_path}")
    print(f"[ok] bundle_id: {bundle_id}")
    print(
        "[ok] counts:"
        f" shift={len(shift_rows_used)}"
        f" invariant={len(invariant_rows)}"
        f" total={len(merged_rows)}"
    )


if __name__ == "__main__":
    main()
