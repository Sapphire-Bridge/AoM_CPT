from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


_EVIDENCE_ID_RE = re.compile(r"\b[ERC]\d+[a-z]?\b")
_CONTRACT_ROW_RE = re.compile(r"^\|\s*`([ERC]\d+[a-z]?)`\s*\|")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def _extract_paper_ids(text: str) -> set[str]:
    return {m.group(0) for m in _EVIDENCE_ID_RE.finditer(text)}


def _extract_contract_ids(text: str) -> list[str]:
    ids: list[str] = []
    for line in text.splitlines():
        m = _CONTRACT_ROW_RE.match(line)
        if m:
            ids.append(m.group(1))
    return ids


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check that paper evidence tags resolve to rows in the evidence contract.")
    p.add_argument("--paper_path", type=str, default=str(ROOT / "AoM_JoLLLI" / "AoM_paper.md"))
    p.add_argument("--contract_path", type=str, default=str(ROOT / "AoM_JoLLLI" / "AoM_evidence_contract.md"))
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    paper_path = Path(str(args.paper_path))
    contract_path = Path(str(args.contract_path))

    if not paper_path.exists():
        print(f"[FAIL] missing paper: {paper_path}", file=sys.stderr)
        return 2
    if not contract_path.exists():
        print(f"[FAIL] missing evidence contract: {contract_path}", file=sys.stderr)
        return 2

    paper = _read_text(paper_path)
    contract = _read_text(contract_path)

    paper_ids = _extract_paper_ids(paper)
    contract_ids = _extract_contract_ids(contract)

    dupes = sorted({x for x in contract_ids if contract_ids.count(x) > 1})
    if dupes:
        print(f"[FAIL] duplicate evidence IDs in contract: {dupes!r}", file=sys.stderr)
        return 1

    missing = sorted(paper_ids - set(contract_ids))
    if missing:
        print(f"[FAIL] evidence IDs cited in paper but missing from contract: {missing!r}", file=sys.stderr)
        return 1

    print(f"[ok] paper evidence IDs: {len(paper_ids)}")
    print(f"[ok] contract evidence IDs: {len(set(contract_ids))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

