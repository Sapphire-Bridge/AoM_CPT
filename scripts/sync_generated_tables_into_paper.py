from __future__ import annotations

import argparse
import difflib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_PATH = ROOT / "AoM_JoLLLI" / "AoM_paper.md"
DEFAULT_GENERATED_MD = ROOT / "tables_generated" / "sdh_specificity.md"
TABLE_ID = "sdh_specificity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync generated Markdown tables into the manuscript.")
    p.add_argument("--generated_md", type=str, default=str(DEFAULT_GENERATED_MD))
    p.add_argument("--paper_path", type=str, default=str(DEFAULT_PAPER_PATH))
    p.add_argument("--check", action="store_true")
    return p.parse_args()


def _markers(table_id: str = TABLE_ID) -> tuple[str, str]:
    return (
        f"<!-- BEGIN GENERATED TABLE: {table_id} -->",
        f"<!-- END GENERATED TABLE: {table_id} -->",
    )


def _normalize_block(text: str) -> str:
    return text.strip("\n") + "\n"


def extract_generated_block(paper_text: str, table_id: str = TABLE_ID) -> str:
    begin_marker, end_marker = _markers(table_id)
    begin_index = paper_text.find(begin_marker)
    end_index = paper_text.find(end_marker)
    if begin_index == -1 or end_index == -1 or end_index < begin_index:
        raise ValueError(f"Could not locate generated table markers for {table_id!r}")
    start = begin_index + len(begin_marker)
    return paper_text[start:end_index].strip("\n") + "\n"


def replace_generated_block(paper_text: str, replacement: str, table_id: str = TABLE_ID) -> str:
    begin_marker, end_marker = _markers(table_id)
    begin_index = paper_text.find(begin_marker)
    end_index = paper_text.find(end_marker)
    if begin_index == -1 or end_index == -1 or end_index < begin_index:
        raise ValueError(f"Could not locate generated table markers for {table_id!r}")
    before = paper_text[: begin_index + len(begin_marker)]
    after = paper_text[end_index:]
    normalized = "\n" + _normalize_block(replacement)
    return before + normalized + after


def _diff(expected: str, actual: str, *, fromfile: str, tofile: str) -> str:
    return "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def main() -> None:
    args = parse_args()
    generated_md = Path(str(args.generated_md))
    paper_path = Path(str(args.paper_path))

    generated_text = _normalize_block(generated_md.read_text(encoding="utf-8"))
    paper_text = paper_path.read_text(encoding="utf-8")
    current_block = extract_generated_block(paper_text)

    if bool(args.check):
        if current_block != generated_text:
            diff = _diff(generated_text, current_block, fromfile=str(generated_md), tofile=str(paper_path))
            raise SystemExit(f"Generated table block is out of sync.\n{diff}")
        return

    updated_text = replace_generated_block(paper_text, generated_text)
    if updated_text != paper_text:
        paper_path.write_text(updated_text, encoding="utf-8")


if __name__ == "__main__":
    main()
