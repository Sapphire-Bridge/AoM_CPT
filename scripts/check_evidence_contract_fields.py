from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

_CONTRACT_ROW_RE = re.compile(r"^\|\s*`([ERC]\d+[a-z]?)`\s*\|")
_EVIDENCE_ID_RE = re.compile(r"\b[ERC]\d+[a-z]?\b")

REQUIRED_CSV_PROVENANCE_COLUMNS: tuple[str, ...] = ("dataset_bundle_id", "git_commit", "argv_sha256")
REQUIRED_CSV_UNIQUE_PROVENANCE_COLUMNS: tuple[str, ...] = ("dataset_bundle_id", "git_commit")


@dataclass(frozen=True)
class EvidenceRow:
    evidence_id: str
    metric_cell: str
    artifacts_cell: str


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def _iter_contract_rows(contract_md: str) -> Iterable[EvidenceRow]:
    for line in contract_md.splitlines():
        m = _CONTRACT_ROW_RE.match(line)
        if not m:
            continue
        evidence_id = m.group(1)
        parts = [p.strip() for p in line.split("|")]
        # Markdown table rows are: | col1 | col2 | ... |
        # -> split produces: ["", col1, col2, ..., ""]
        if len(parts) < 9:
            raise ValueError(f"Malformed contract row for {evidence_id}: expected 8 columns, got {len(parts) - 2}")
        metric_cell = parts[3]
        artifacts_cell = parts[4]
        yield EvidenceRow(evidence_id=evidence_id, metric_cell=metric_cell, artifacts_cell=artifacts_cell)


def _extract_backticked_tokens(cell: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"`([^`]+)`", cell)]


def _extract_field_patterns(metric_cell: str) -> list[str]:
    out: list[str] = []
    for tok in _extract_backticked_tokens(metric_cell):
        if _EVIDENCE_ID_RE.fullmatch(tok):
            continue
        out.append(tok)
    return out


def _expand_brace_patterns(pat: str) -> list[str]:
    """
    Expand simple single-brace patterns like 'datasets.{disamb,cf}.n_rows_invalid'.
    Returns [pat] unchanged if no supported brace pattern is present.
    """
    if "{" not in pat or "}" not in pat:
        return [pat]
    # Only support one brace group to keep this deterministic and safe.
    if pat.count("{") != 1 or pat.count("}") != 1:
        return [pat]
    pre, rest = pat.split("{", 1)
    inner, post = rest.split("}", 1)
    options = [x.strip() for x in inner.split(",") if x.strip()]
    if not options:
        return [pat]
    return [f"{pre}{opt}{post}" for opt in options]


def _path_from_contract_token(token: str, *, results_dir: Path) -> Path:
    p = Path(token)
    if p.is_absolute():
        return p

    parts = p.parts
    # Treat `results*/...` as a logical results-root (allow remapping to a different dir).
    if len(parts) > 1 and parts[0].startswith("results"):
        return results_dir / Path(*parts[1:])

    return ROOT / token


def _read_csv_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
    if header is None:
        raise ValueError(f"CSV is empty: {str(path)}")
    return [str(x).strip() for x in header]


def _read_unique_nonempty_value(path: Path, *, column: str, max_rows: int = 5000) -> str:
    values: set[str] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= max_rows:
                break
            v = "" if row.get(column) is None else str(row.get(column)).strip()
            if not v:
                continue
            values.add(v)
            if len(values) > 1:
                break
    if not values:
        raise ValueError(f"CSV {str(path)} has no non-empty values for required column {column!r}")
    if len(values) > 1:
        raise ValueError(f"CSV {str(path)} has multiple values for column {column!r}: {sorted(values)!r}")
    return next(iter(values))


def _require_nonempty_column_values(path: Path, *, column: str) -> None:
    total_rows = 0
    empty_rows = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            total_rows += 1
            v = "" if row.get(column) is None else str(row.get(column)).strip()
            if not v:
                empty_rows += 1
    if total_rows == 0:
        raise ValueError(f"CSV {str(path)} has no rows for required column {column!r}")
    if empty_rows > 0:
        raise ValueError(f"CSV {str(path)} has {empty_rows} empty values for required column {column!r}")


def _json_has_path_pattern(obj: Any, dotted: str) -> bool:
    """
    Return True iff `dotted` pattern exists in nested dicts/lists.

    Supports:
    - dict traversal via key segments
    - list traversal via integer index segments
    - wildcard segment `*` meaning "any key/index at this level"
    """
    segs = [s for s in dotted.split(".") if s]

    def _rec(cur: Any, i: int) -> bool:
        if i >= len(segs):
            return True
        seg = segs[i]

        if seg == "*":
            if isinstance(cur, dict):
                return any(_rec(v, i + 1) for v in cur.values())
            if isinstance(cur, list):
                return any(_rec(v, i + 1) for v in cur)
            return False

        if isinstance(cur, dict):
            if seg not in cur:
                return False
            return _rec(cur[seg], i + 1)

        if isinstance(cur, list):
            try:
                idx = int(seg)
            except Exception:
                return False
            if idx < 0 or idx >= len(cur):
                return False
            return _rec(cur[idx], i + 1)

        return False

    return _rec(obj, 0)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check that evidence-contract fields resolve to real artifacts/columns.")
    p.add_argument("--paper_path", type=str, default=str(ROOT / "AoM_JoLLLI" / "AoM_paper.md"))
    p.add_argument("--contract_path", type=str, default=str(ROOT / "AoM_JoLLLI" / "AoM_evidence_contract.md"))
    p.add_argument(
        "--results_dir",
        type=str,
        default=str(ROOT / "results_submission_full"),
        help="Directory containing generated results artifacts (CSV + .manifest.json).",
    )
    p.add_argument("--all_rows", action="store_true", help="Validate all contract rows (not just those cited in paper).")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    paper_path = Path(str(args.paper_path))
    contract_path = Path(str(args.contract_path))
    results_dir = Path(str(args.results_dir))

    if not paper_path.exists():
        print(f"[FAIL] missing paper: {paper_path}", file=sys.stderr)
        return 2
    if not contract_path.exists():
        print(f"[FAIL] missing evidence contract: {contract_path}", file=sys.stderr)
        return 2
    if not results_dir.exists():
        print(f"[FAIL] missing results_dir: {results_dir}", file=sys.stderr)
        return 2

    paper_ids = {m.group(0) for m in _EVIDENCE_ID_RE.finditer(_read_text(paper_path))}
    contract_md = _read_text(contract_path)

    rows = list(_iter_contract_rows(contract_md))
    rows_by_id = {r.evidence_id: r for r in rows}

    target_ids = sorted(rows_by_id.keys() if bool(args.all_rows) else paper_ids)

    failures: list[str] = []
    for eid in target_ids:
        row = rows_by_id.get(eid)
        if row is None:
            failures.append(f"{eid}: missing row in contract")
            continue

        artifact_tokens = _extract_backticked_tokens(row.artifacts_cell)
        if not artifact_tokens:
            failures.append(f"{eid}: no artifact paths listed")
            continue

        field_pats_raw = _extract_field_patterns(row.metric_cell)
        field_pats: list[str] = []
        for p in field_pats_raw:
            field_pats.extend(_expand_brace_patterns(p))

        for tok in artifact_tokens:
            pth = _path_from_contract_token(tok, results_dir=results_dir)
            if not pth.exists():
                failures.append(f"{eid}: missing artifact {tok!r} (resolved to {str(pth)!r})")
                continue

            if pth.suffix.lower() == ".csv":
                try:
                    header = _read_csv_header(pth)
                    header_set = set(header)
                except Exception as e:
                    failures.append(f"{eid}: failed to read CSV header {str(pth)!r}: {type(e).__name__}: {e}")
                    continue

                missing_cols = [c for c in REQUIRED_CSV_PROVENANCE_COLUMNS if c not in header_set]
                if missing_cols:
                    failures.append(f"{eid}: {str(pth)!r} missing provenance columns {missing_cols!r}")
                    continue
                try:
                    for col in REQUIRED_CSV_PROVENANCE_COLUMNS:
                        _require_nonempty_column_values(pth, column=col)
                    for col in REQUIRED_CSV_UNIQUE_PROVENANCE_COLUMNS:
                        _read_unique_nonempty_value(pth, column=col)
                except Exception as e:
                    failures.append(f"{eid}: {str(pth)!r} provenance check failed: {type(e).__name__}: {e}")
                    continue

                # Field patterns: enforce only on patterns that look like CSV columns.
                for pat in field_pats:
                    if "." in pat:
                        continue
                    if any(ch in pat for ch in "{}"):
                        # Skip complex patterns that are not literal column names.
                        continue
                    if "*" in pat:
                        if not any(fnmatch.fnmatchcase(col, pat) for col in header):
                            failures.append(f"{eid}: {str(pth)!r} missing any columns matching {pat!r}")
                        continue
                    if pat and pat not in header_set:
                        failures.append(f"{eid}: {str(pth)!r} missing column {pat!r}")

            elif pth.suffix.lower() == ".json":
                try:
                    obj = json.loads(_read_text(pth))
                except Exception as e:
                    failures.append(f"{eid}: failed to parse JSON {str(pth)!r}: {type(e).__name__}: {e}")
                    continue

                for pat in field_pats:
                    if "." not in pat:
                        continue
                    if not _json_has_path_pattern(obj, pat):
                        failures.append(f"{eid}: {str(pth)!r} missing JSON path {pat!r}")

            else:
                # Code/markdown artifacts: existence check only.
                continue

    if failures:
        print("[FAIL] evidence contract artifact/field checks failed:", file=sys.stderr)
        for msg in failures[:200]:
            print(" -", msg, file=sys.stderr)
        if len(failures) > 200:
            print(f"... ({len(failures) - 200} more)", file=sys.stderr)
        return 1

    print(f"[ok] validated evidence rows: {len(target_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
