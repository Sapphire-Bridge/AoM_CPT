from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aom.repro_metadata import STRICT_VERIFICATION_PROFILE, manifest_has_strict_verification_profile, read_json_object


ROOT = Path(__file__).resolve().parents[1]
_AUTO_RESULTS_DIR_CANDIDATES = (
    ROOT / "results_submission_full",
    ROOT / "results_submission",
    ROOT / "results",
)


def _load_manifest_for_results_dir(results_dir: Path) -> dict[str, object]:
    manifest_path = results_dir / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        raise ValueError(f"missing RUN_MANIFEST.json in {results_dir}")
    try:
        manifest = read_json_object(manifest_path)
    except Exception as e:
        raise ValueError(f"failed to parse {manifest_path}: {type(e).__name__}: {e}") from e
    return manifest


def _find_auto_results_dir() -> Path | None:
    # Only auto-detect canonical full-results directories here.
    # `run_paper.py` outputs such as `results/paper_smoke`, `results/paper_m1max`,
    # and `results/paper_a100` are intentionally excluded because they do not
    # produce the full paper-facing artifact surface required by the evidence
    # contract field check.
    for p in _AUTO_RESULTS_DIR_CANDIDATES:
        if (p / "RUN_MANIFEST.json").exists():
            manifest = _load_manifest_for_results_dir(p)
            if manifest_has_strict_verification_profile(manifest):
                return p
    return None


def test_evidence_contract_fields_resolve_when_results_present() -> None:
    env = os.environ.get("AOM_RESULTS_DIR", "").strip()
    if env:
        results_dir = Path(env)
        if not results_dir.exists():
            pytest.fail(f"AOM_RESULTS_DIR does not exist: {results_dir}")
        try:
            manifest = _load_manifest_for_results_dir(results_dir)
        except ValueError as e:
            pytest.fail(str(e))
        if not manifest_has_strict_verification_profile(manifest):
            pytest.fail(
                "AOM_RESULTS_DIR does not point to a strict evidence-contract results directory: "
                f"{results_dir} (verification_profile={manifest.get('verification_profile')!r}, "
                f"expected {STRICT_VERIFICATION_PROFILE!r})"
            )
    else:
        try:
            results_dir = _find_auto_results_dir()
        except ValueError as e:
            pytest.fail(str(e))
        if results_dir is None:
            pytest.skip(
                "No canonical strict-results directory found; "
                "run scripts/run_submission_full_strong.sh or set AOM_RESULTS_DIR explicitly."
            )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_evidence_contract_fields.py"),
        "--results_dir",
        str(results_dir),
    ]
    subprocess.run(cmd, check=True)


def test_auto_results_dir_ignores_non_strict_manifests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    smoke_dir = tmp_path / "results_submission_full"
    smoke_dir.mkdir()
    (smoke_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({"mode": "smoke", "verification_profile": "reviewer_smoke"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "_AUTO_RESULTS_DIR_CANDIDATES", (smoke_dir,))
    assert _find_auto_results_dir() is None


def test_auto_results_dir_accepts_legacy_strict_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    strict_dir = tmp_path / "results_submission_full"
    strict_dir.mkdir()
    (strict_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({"mode": "submission_full_strong"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "_AUTO_RESULTS_DIR_CANDIDATES", (strict_dir,))
    assert _find_auto_results_dir() == strict_dir


def test_explicit_non_strict_profile_overrides_legacy_mode() -> None:
    assert not manifest_has_strict_verification_profile(
        {
            "mode": "submission_full_strong",
            "verification_profile": "reviewer_smoke",
        }
    )
