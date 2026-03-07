from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sync_generated_tables_into_paper import DEFAULT_PAPER_PATH, extract_generated_block
from tables.table_sdh_specificity import generate_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check that the generated SDH table matches the manuscript block.")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results_submission_full"))
    p.add_argument("--paper_path", type=str, default=str(DEFAULT_PAPER_PATH))
    return p.parse_args()


def _normalize_block(text: str) -> str:
    return text.strip("\n") + "\n"


def main() -> None:
    args = parse_args()
    results_dir = Path(str(args.results_dir))
    paper_path = Path(str(args.paper_path))
    csv_path = results_dir / "cpt_specificity_disamb_only.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing SDH CSV: {csv_path}")

    _, generated_md = generate_outputs(csv_path)
    generated_text = _normalize_block(generated_md)
    current_text = extract_generated_block(paper_path.read_text(encoding="utf-8"))
    if current_text != generated_text:
        diff = "".join(
            difflib.unified_diff(
                generated_text.splitlines(keepends=True),
                current_text.splitlines(keepends=True),
                fromfile=str(csv_path),
                tofile=str(paper_path),
            )
        )
        raise SystemExit(f"SDH table block is out of sync with {csv_path}.\n{diff}")


if __name__ == "__main__":
    main()
