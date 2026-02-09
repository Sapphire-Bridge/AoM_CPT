from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _find_results_dir() -> Path | None:
    env = os.environ.get("AOM_RESULTS_DIR", "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None

    candidates = [
        ROOT / "results_submission_full",
        ROOT / "results_submission",
        ROOT / "results",
        ROOT / "results" / "paper_a100",
        ROOT / "results" / "paper_m1max",
        ROOT / "results" / "paper_smoke",
    ]
    for p in candidates:
        if (p / "RUN_MANIFEST.json").exists():
            return p
    return None


def test_evidence_contract_fields_resolve_when_results_present() -> None:
    results_dir = _find_results_dir()
    if results_dir is None:
        pytest.skip("No results directory found; run a paper-mode runner first (or set AOM_RESULTS_DIR).")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_evidence_contract_fields.py"),
        "--results_dir",
        str(results_dir),
    ]
    subprocess.run(cmd, check=True)

