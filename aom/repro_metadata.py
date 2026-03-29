from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


STRICT_VERIFICATION_PROFILE = "paper_evidence_full"
PARTIAL_VERIFICATION_PROFILE = "paper_partial"
SMOKE_VERIFICATION_PROFILE = "reviewer_smoke"
_LEGACY_STRICT_MODES = {"submission_full_strong"}

_DEPENDENCY_PROVENANCE_ENV = {
    "dependency_install_file": "AOM_DEPENDENCY_INSTALL_FILE",
    "dependency_install_profile": "AOM_DEPENDENCY_INSTALL_PROFILE",
    "dependency_install_sha256": "AOM_DEPENDENCY_INSTALL_SHA256",
    "runner_os": "AOM_RUNNER_OS",
    "base_image": "AOM_BASE_IMAGE",
}


def verification_profile_for_mode(mode: str) -> str:
    mode_s = str(mode or "").strip()
    if mode_s == "submission_full_strong":
        return STRICT_VERIFICATION_PROFILE
    if mode_s == "smoke":
        return SMOKE_VERIFICATION_PROFILE
    return PARTIAL_VERIFICATION_PROFILE


def manifest_has_strict_verification_profile(manifest: Mapping[str, Any]) -> bool:
    profile = str(manifest.get("verification_profile", "") or "").strip()
    if profile == STRICT_VERIFICATION_PROFILE:
        return True
    legacy_mode = str(manifest.get("mode", "") or "").strip()
    return legacy_mode in _LEGACY_STRICT_MODES


def collect_dependency_install_provenance_from_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for field, env_name in _DEPENDENCY_PROVENANCE_ENV.items():
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            out[field] = raw
    return out


def read_json_object(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {p}, got {type(obj).__name__}")
    return obj
