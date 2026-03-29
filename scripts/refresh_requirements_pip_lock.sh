#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python3"
OUT_FILE="$ROOT/requirements.pip.lock.txt"
STRICT_PIP_VERSION="24.2"

usage() {
  cat <<'EOF'
Refresh the strict pip lock used for paper-facing clean-room reproduction.

This script is intended to be run from the reference strict environment:
- ubuntu-24.04
- Python 3.12.7
- pip 24.2

Usage:
  bash scripts/refresh_requirements_pip_lock.sh [options]

Options:
  --python BIN                python executable to use (default: python3)
  --output PATH               output path (default: requirements.pip.lock.txt)
  -h, --help                  show this help
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || die "Missing value for --python"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || die "Missing value for --output"
      OUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Missing python executable: $PYTHON_BIN"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aom-pip-lock.XXXXXX")"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

VENV_DIR="$TMP_DIR/venv"
"$PYTHON_BIN" -m venv "$VENV_DIR"
VENV_PY="$VENV_DIR/bin/python"

echo "[run] $VENV_PY -m pip install --upgrade pip==$STRICT_PIP_VERSION"
"$VENV_PY" -m pip install --upgrade "pip==$STRICT_PIP_VERSION"
echo "[run] $VENV_PY -m pip install -r $ROOT/requirements.txt"
"$VENV_PY" -m pip install -r "$ROOT/requirements.txt"

mkdir -p "$(dirname "$OUT_FILE")"
{
  echo "# Strict pip lock for paper-facing clean-room reproduction."
  echo "# Canonical envelope: ubuntu-24.04, Python 3.12.7, pip 24.2."
  echo "# Refresh with: bash scripts/refresh_requirements_pip_lock.sh"
  "$VENV_PY" - <<'PY'
import subprocess
import sys


def normalize_requirement(line: str) -> str:
    if "==" in line:
        name, version = line.split("==", 1)
        return f"{name.lower()}=={version}"
    if " @ " in line:
        name, rest = line.split(" @ ", 1)
        return f"{name.lower()} @ {rest}"
    return line


raw = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
lines = [normalize_requirement(line.strip()) for line in raw.splitlines() if line.strip()]
for line in sorted(lines):
    print(line)
PY
} > "$OUT_FILE"

echo "[ok] wrote $OUT_FILE"
