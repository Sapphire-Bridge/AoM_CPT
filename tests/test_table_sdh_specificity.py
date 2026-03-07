from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tables.table_sdh_specificity import generate_outputs, load_rows, normalize_rows


FIXTURE = ROOT / "tests" / "fixtures" / "sdh_specificity_fixture.csv"


def test_caption_reflects_constant_direction_count() -> None:
    _, md_text = generate_outputs(FIXTURE)
    assert "N directions per model = 104." in md_text


def test_generator_fails_when_strategy_counts_contradict_caption(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad_sdh.csv"
    with FIXTURE.open("r", encoding="utf-8-sig", newline="") as src, bad_csv.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(reader):
            if index == 0:
                row["cpt_spec_n_directions_ctrl_patched_same_index"] = "1"
            writer.writerow(row)

    with pytest.raises(ValueError, match="matched_token-only caption assumption"):
        generate_outputs(bad_csv)


def test_model_order_is_stable() -> None:
    rows = normalize_rows(load_rows(FIXTURE))
    assert [row.model_id for row in rows] == [
        "gpt2",
        "Qwen/Qwen2.5-0.5B",
        "meta-llama/Llama-3.2-1B",
    ]


def test_ci_formatting_is_deterministic() -> None:
    _, md_text = generate_outputs(FIXTURE)
    assert "| GPT‑2 (124M) | 12 | 3 | 0.09 [-0.05, 0.23] | 0.38 [0.13, 0.72] | -0.29 [-0.66, 0.00] | 0.52 [0.42, 0.62] |" in md_text
