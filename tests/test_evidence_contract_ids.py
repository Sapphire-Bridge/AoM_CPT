from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_evidence_ids_resolve_to_contract_rows() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_evidence_contract.py")], check=True)

