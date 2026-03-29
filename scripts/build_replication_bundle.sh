#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="submission_full_strong"
RESULTS_DIR="results_submission_full"
TABLES_DIR="tables_submission_full"
ARCHIVE="aom_replication_bundle.tar.gz"
REVISION=""
TOKENIZER_REVISION=""
LOCAL_FILES_ONLY=0
REPRO_MODE="auto"
SKIP_RUN=0
SKIP_RUN_SET=0
SKIP_TABLES=0

usage() {
  cat <<'EOF'
Build a paper-ready AoM replication bundle.

Usage:
  bash scripts/build_replication_bundle.sh [options]

Options:
  --mode MODE                  run mode: smoke | m1max | a100 | submission_full_strong (default: submission_full_strong)
  --results-dir PATH           run/results directory (default: results_submission_full)
  --tables-dir PATH            output directory for strict LaTeX tables (default: tables_submission_full)
  --archive PATH               output archive path (default: aom_replication_bundle.tar.gz)
  --revision REV               HF model revision pin passed to run_paper
  --tokenizer-revision REV     HF tokenizer revision pin passed to run_paper
  --local-files-only           pass --local_files_only to run_paper
  --repro-mode MODE            auto | full_recompute | frozen_artifacts (default: auto)
  --skip-run                   do not run scripts/run_paper.py
  --skip-tables                do not run strict table regeneration
  -h, --help                   show this help

Examples:
  bash scripts/build_replication_bundle.sh --mode submission_full_strong --repro-mode full_recompute
  bash scripts/build_replication_bundle.sh --mode submission_full_strong --repro-mode frozen_artifacts --results-dir results_submission_full
  bash scripts/build_replication_bundle.sh --mode m1max --revision <hf_commit>
  bash scripts/build_replication_bundle.sh --skip-run --results-dir results/paper_m1max --tables-dir tables_submission_full
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing required directory: $1"
}

abs_under_root_to_rel() {
  local abs="$1"
  case "$abs" in
    "$ROOT"/*)
      echo "${abs#"$ROOT"/}"
      ;;
    *)
      die "Path must be inside repo root ($ROOT): $abs"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "Missing value for --mode"
      MODE="$2"
      shift 2
      ;;
    --results-dir)
      [[ $# -ge 2 ]] || die "Missing value for --results-dir"
      RESULTS_DIR="$2"
      shift 2
      ;;
    --tables-dir)
      [[ $# -ge 2 ]] || die "Missing value for --tables-dir"
      TABLES_DIR="$2"
      shift 2
      ;;
    --archive)
      [[ $# -ge 2 ]] || die "Missing value for --archive"
      ARCHIVE="$2"
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || die "Missing value for --revision"
      REVISION="$2"
      shift 2
      ;;
    --tokenizer-revision)
      [[ $# -ge 2 ]] || die "Missing value for --tokenizer-revision"
      TOKENIZER_REVISION="$2"
      shift 2
      ;;
    --local-files-only)
      LOCAL_FILES_ONLY=1
      shift
      ;;
    --repro-mode)
      [[ $# -ge 2 ]] || die "Missing value for --repro-mode"
      REPRO_MODE="$2"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN=1
      SKIP_RUN_SET=1
      shift
      ;;
    --skip-tables)
      SKIP_TABLES=1
      shift
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

case "$MODE" in
  smoke|m1max|a100|submission_full_strong) ;;
  *) die "Invalid --mode: $MODE (expected smoke|m1max|a100|submission_full_strong)" ;;
esac

case "$REPRO_MODE" in
  auto|full_recompute|frozen_artifacts) ;;
  *) die "Invalid --repro-mode: $REPRO_MODE (expected auto|full_recompute|frozen_artifacts)" ;;
esac

if [[ "$REPRO_MODE" == "full_recompute" ]]; then
  [[ "$SKIP_RUN_SET" -eq 0 ]] || die "--repro-mode full_recompute cannot be combined with --skip-run"
  SKIP_RUN=0
fi
if [[ "$REPRO_MODE" == "frozen_artifacts" ]]; then
  SKIP_RUN=1
fi

RESULTS_DIR_ABS="$(cd "$(dirname "$RESULTS_DIR")" && pwd)/$(basename "$RESULTS_DIR")"
TABLES_DIR_ABS="$(cd "$(dirname "$TABLES_DIR")" && pwd)/$(basename "$TABLES_DIR")"
mkdir -p "$(dirname "$ARCHIVE")"
ARCHIVE_ABS="$(cd "$(dirname "$ARCHIVE")" && pwd)/$(basename "$ARCHIVE")"

if [[ "$SKIP_RUN" -eq 0 ]]; then
  mkdir -p "$RESULTS_DIR_ABS"
  if [[ "$MODE" == "submission_full_strong" ]]; then
    cmd=(
      bash
      "$ROOT/scripts/run_submission_full_strong.sh"
      --results-dir
      "$RESULTS_DIR"
      --tables-dir
      "$TABLES_DIR"
    )
    if [[ "$LOCAL_FILES_ONLY" -eq 1 ]]; then
      cmd+=(--local-files-only)
    fi
    if [[ -n "$REVISION" ]]; then
      cmd+=(--revision "$REVISION")
    fi
    if [[ -n "$TOKENIZER_REVISION" ]]; then
      cmd+=(--tokenizer-revision "$TOKENIZER_REVISION")
    fi
    if [[ "$SKIP_TABLES" -eq 1 ]]; then
      cmd+=(--skip-tables)
    fi
    echo "[run] ${cmd[*]}"
    "${cmd[@]}"
  else
    cmd=(
      python
      "$ROOT/scripts/run_paper.py"
      "$MODE"
      --results_dir
      "$RESULTS_DIR_ABS"
    )
    if [[ "$LOCAL_FILES_ONLY" -eq 1 ]]; then
      cmd+=(--local_files_only)
    fi
    if [[ -n "$REVISION" ]]; then
      cmd+=(--revision "$REVISION")
    fi
    if [[ -n "$TOKENIZER_REVISION" ]]; then
      cmd+=(--tokenizer_revision "$TOKENIZER_REVISION")
    fi
    echo "[run] ${cmd[*]}"
    "${cmd[@]}"
  fi
fi

if [[ "$SKIP_TABLES" -eq 0 ]]; then
  # submission_full_strong already runs strict table generation (unless it was asked to skip).
  if [[ "$MODE" != "submission_full_strong" || "$SKIP_RUN" -eq 1 ]]; then
    mkdir -p "$TABLES_DIR_ABS"
    echo "[run] MAKE_TABLES_STRICT=1 make tables RESULTS_DIR=$RESULTS_DIR_ABS TABLES_OUT_DIR=$TABLES_DIR_ABS"
    MAKE_TABLES_STRICT=1 make tables RESULTS_DIR="$RESULTS_DIR_ABS" TABLES_OUT_DIR="$TABLES_DIR_ABS"
  fi
fi

require_dir "$ROOT/data_paper_hardened_v2"
require_file "$ROOT/data_paper_hardened_v2/disamb_pairs.jsonl"
require_file "$ROOT/data_paper_hardened_v2/counterfactual.jsonl"
require_file "$ROOT/data_paper_hardened_v2/coherence.jsonl"
require_file "$ROOT/data_paper_hardened_v2/DATASET_MANIFEST.json"

require_dir "$RESULTS_DIR_ABS"
require_file "$RESULTS_DIR_ABS/RUN_MANIFEST.json"
require_file "$RESULTS_DIR_ABS/results_report.md"
if [[ "$MODE" == "submission_full_strong" ]]; then
  require_file "$ROOT/data_paper_hardened_v2/counterfactual_invariant_sub.jsonl"
  require_file "$RESULTS_DIR_ABS/cf_patching_shift_vs_subinv.csv"
  require_file "$RESULTS_DIR_ABS/cf_patching_shift_vs_subinv.manifest.json"
  require_file "$RESULTS_DIR_ABS/CF_SHIFT_SUBINV_RUN_MANIFEST.json"
  require_file "$RESULTS_DIR_ABS/coh_patching_qwen.csv"
  require_file "$RESULTS_DIR_ABS/coh_patching_qwen.manifest.json"
  require_file "$RESULTS_DIR_ABS/coh_patching_qwen15.csv"
  require_file "$RESULTS_DIR_ABS/coh_patching_qwen15.manifest.json"
  require_file "$RESULTS_DIR_ABS/coh_patching_qwen3b.csv"
  require_file "$RESULTS_DIR_ABS/coh_patching_qwen3b.manifest.json"
fi

shopt -s nullglob
results_csv=("$RESULTS_DIR_ABS"/*.csv)
results_manifests=("$RESULTS_DIR_ABS"/*.manifest.json)
shopt -u nullglob
(( ${#results_csv[@]} > 0 )) || die "No CSV files found in results directory: $RESULTS_DIR_ABS"
(( ${#results_manifests[@]} > 0 )) || die "No .manifest.json files found in results directory: $RESULTS_DIR_ABS"

if [[ "$SKIP_TABLES" -eq 0 ]]; then
  require_dir "$TABLES_DIR_ABS"
  require_file "$TABLES_DIR_ABS/aom_eval.tex"
  require_file "$TABLES_DIR_ABS/cf_patching.tex"
  require_file "$TABLES_DIR_ABS/coh_patching.tex"
  if [[ "$MODE" == "submission_full_strong" ]]; then
    require_file "$TABLES_DIR_ABS/cf_patching_shift_vs_subinv.tex"
  fi
fi

results_rel="$(abs_under_root_to_rel "$RESULTS_DIR_ABS")"
tables_rel=""
if [[ -d "$TABLES_DIR_ABS" ]]; then
  tables_rel="$(abs_under_root_to_rel "$TABLES_DIR_ABS")"
fi

bundle_paths=(
  "README.md"
  "LICENSE"
  "CITATION.cff"
  "PAPER_MODES.md"
  "PAPER_VERIFICATION_GUIDE.md"
  "AoM_JoLLLI"
  "Makefile"
  "requirements.txt"
  "requirements.pip.lock.txt"
  "requirements.lock.txt"
  "aom"
  "aom_eval.py"
  "aom_cf_patching.py"
  "aom_coh_patching.py"
  "scripts/run_submission_full_strong.sh"
  "scripts/run_paper.py"
  "scripts/report_results.py"
  "scripts/make_tables.py"
  "scripts/build_replication_bundle.sh"
  "scripts/final_repro_cleanroom.sh"
  "scripts/refresh_requirements_pip_lock.sh"
  "scripts/run_cf_shift_vs_subinv.sh"
  "scripts/build_cf_shift_subinv_dataset.py"
  "scripts/check_evidence_contract.py"
  "scripts/check_evidence_contract_fields.py"
  "tests"
  "tables/table_aom_eval.py"
  "tables/table_cf_patching.py"
  "tables/table_cf_shift_vs_subinv.py"
  "tables/table_coh_patching.py"
  "data/README.md"
  "data_paper_hardened_v2"
  "$results_rel"
)

if [[ -f "$ROOT/CHANGELOG.md" ]]; then
  bundle_paths+=("CHANGELOG.md")
fi
if [[ -f "$ROOT/CONTRIBUTING.md" ]]; then
  bundle_paths+=("CONTRIBUTING.md")
fi
if [[ -f "$ROOT/SECURITY.md" ]]; then
  bundle_paths+=("SECURITY.md")
fi
if [[ -f "$ROOT/.zenodo.json" ]]; then
  bundle_paths+=(".zenodo.json")
fi
if [[ -d "$ROOT/docs/release" ]]; then
  bundle_paths+=("docs/release")
fi

if [[ -f "$ROOT/SOURCE_GIT_COMMIT.txt" ]]; then
  bundle_paths+=("SOURCE_GIT_COMMIT.txt")
fi

if [[ -n "$tables_rel" ]]; then
  bundle_paths+=("$tables_rel")
fi

mkdir -p "$(dirname "$ARCHIVE_ABS")"
echo "[run] build archive $ARCHIVE_ABS (python tarfile)"
python3 - "$ROOT" "$ARCHIVE_ABS" "${bundle_paths[@]}" <<'PY'
import os
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
archive = Path(sys.argv[2]).resolve()
paths = sys.argv[3:]

def include_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = info.name.split("/")
    base = os.path.basename(info.name)
    if "__pycache__" in parts:
        return None
    if base.endswith(".pyc") or base == ".DS_Store":
        return None
    return info

with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
    for rel in paths:
        src = root / rel
        if not src.exists():
            raise SystemExit(f"[error] Missing required bundle path: {src}")
        tar.add(src, arcname=rel, recursive=True, filter=include_filter)
PY

checksum_path="${ARCHIVE_ABS}.sha256"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$ARCHIVE_ABS" > "$checksum_path"
elif command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ARCHIVE_ABS" > "$checksum_path"
else
  die "No SHA-256 tool found (expected shasum or sha256sum)"
fi

echo "[ok] wrote archive: $ARCHIVE_ABS"
echo "[ok] wrote digest:  $checksum_path"
