#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="data_paper_hardened_v2"
RESULTS_DIR="results_submission_full"
TABLES_DIR="tables_submission_full"
DEVICE="cpu"
BOOTSTRAP_N=1000
BOOTSTRAP_SEED=42
CI=0.95
DATASET_PROFILE="full"
SKIP_BUILD_DATA=0
LOCAL_FILES_ONLY=0
REVISION=""
TOKENIZER_REVISION=""
SWEEP_SEEDS=(0)

MODELS=(
  gpt2
  Qwen/Qwen2.5-0.5B
  Qwen/Qwen2.5-1.5B
  Qwen/Qwen2.5-3B
)

usage() {
  cat <<'EOF'
Run CF patching for shift vs substitution-invariant controls across four models.

Outputs:
  <results-dir>/cf_patching_<profile>.csv
  <results-dir>/cf_patching_<profile>.manifest.json
  <results-dir>/CF_SHIFT_SUBINV_RUN_MANIFEST.json
  <tables-dir>/cf_patching_<profile>.md
  <tables-dir>/cf_patching_<profile>.csv
  <tables-dir>/cf_patching_<profile>.tex

Usage:
  bash scripts/run_cf_shift_vs_subinv.sh [options]

Options:
  --dataset-profile NAME      one of: full, single_block (default: full)
  --data-dir PATH             dataset directory (default: data_paper_hardened_v2)
  --results-dir PATH          output results directory (default: results_submission_full)
  --tables-dir PATH           output table directory (default: tables_submission_full)
  --device DEV                device for patching (default: cpu)
  --bootstrap-n N             bootstrap replicates (default: 1000)
  --bootstrap-seed N          bootstrap seed (default: 42)
  --ci FLOAT                  confidence interval level (default: 0.95)
  --sweep-seeds LIST          comma-separated seed list (default: 0; must contain exactly one seed for table output)
  --local-files-only          pass --local_files_only to model loading
  --revision REV              HF model revision (optional)
  --tokenizer-revision REV    HF tokenizer revision (optional)
  --skip-build-data           skip dataset merge step and reuse existing merged file/manifest
  -h, --help                  show this help
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-profile)
      [[ $# -ge 2 ]] || die "Missing value for --dataset-profile"
      DATASET_PROFILE="$2"
      shift 2
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || die "Missing value for --data-dir"
      DATA_DIR="$2"
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
    --device)
      [[ $# -ge 2 ]] || die "Missing value for --device"
      DEVICE="$2"
      shift 2
      ;;
    --bootstrap-n)
      [[ $# -ge 2 ]] || die "Missing value for --bootstrap-n"
      BOOTSTRAP_N="$2"
      shift 2
      ;;
    --bootstrap-seed)
      [[ $# -ge 2 ]] || die "Missing value for --bootstrap-seed"
      BOOTSTRAP_SEED="$2"
      shift 2
      ;;
    --ci)
      [[ $# -ge 2 ]] || die "Missing value for --ci"
      CI="$2"
      shift 2
      ;;
    --sweep-seeds)
      [[ $# -ge 2 ]] || die "Missing value for --sweep-seeds"
      IFS=',' read -r -a SWEEP_SEEDS <<< "$2"
      shift 2
      ;;
    --local-files-only)
      LOCAL_FILES_ONLY=1
      shift
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
    --skip-build-data)
      SKIP_BUILD_DATA=1
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

BASE_CF_PATH="$DATA_DIR/counterfactual.jsonl"
SUBINV_CF_PATH="$DATA_DIR/counterfactual_invariant_sub.jsonl"
case "$DATASET_PROFILE" in
  full)
    PROFILE_TAG="shift_vs_subinv"
    MERGED_CF_PATH="$DATA_DIR/counterfactual_shift_plus_subinv.jsonl"
    MERGED_MANIFEST_PATH="$DATA_DIR/DATASET_MANIFEST_cf_shift_plus_subinv.json"
    DATASET_NAME="cf_shift_plus_subinv_v1"
    EXPECTED_SHIFT_COUNT=60
    SHIFT_GEOMETRY_FILTER=all
    ;;
  single_block)
    PROFILE_TAG="shift_singleblock_vs_subinv"
    MERGED_CF_PATH="$DATA_DIR/counterfactual_shift_singleblock_plus_subinv.jsonl"
    MERGED_MANIFEST_PATH="$DATA_DIR/DATASET_MANIFEST_cf_shift_singleblock_plus_subinv.json"
    DATASET_NAME="cf_shift_singleblock_vs_subinv_v1"
    EXPECTED_SHIFT_COUNT=40
    SHIFT_GEOMETRY_FILTER=single_block_only
    ;;
  *)
    die "Unknown --dataset-profile: $DATASET_PROFILE (expected: full|single_block)"
    ;;
esac

CSV_PATH="$RESULTS_DIR/cf_patching_${PROFILE_TAG}.csv"
TABLE_MD_PATH="$TABLES_DIR/cf_patching_${PROFILE_TAG}.md"
TABLE_CSV_PATH="$TABLES_DIR/cf_patching_${PROFILE_TAG}.csv"
TABLE_TEX_PATH="$TABLES_DIR/cf_patching_${PROFILE_TAG}.tex"

[[ -f "$BASE_CF_PATH" ]] || die "Missing base CF dataset: $BASE_CF_PATH"
[[ -f "$SUBINV_CF_PATH" ]] || die "Missing substitution-invariant CF dataset: $SUBINV_CF_PATH"
[[ "${#SWEEP_SEEDS[@]}" -eq 1 ]] || die "Exactly one sweep seed is required for publication table output (multi-seed CI pooling not implemented)."

mkdir -p "$RESULTS_DIR" "$TABLES_DIR"

COMMAND_LOG="$RESULTS_DIR/commands_cf_shift_vs_subinv.log"
: > "$COMMAND_LOG"

run_and_log() {
  local -a cmd=("$@")
  echo "[run] ${cmd[*]}"
  printf "%s\n" "${cmd[*]}" >> "$COMMAND_LOG"
  "${cmd[@]}"
}

if [[ "$SKIP_BUILD_DATA" -eq 0 ]]; then
  run_and_log python scripts/build_cf_shift_subinv_dataset.py \
    --base_cf_path "$BASE_CF_PATH" \
    --subinv_cf_path "$SUBINV_CF_PATH" \
    --out_cf_path "$MERGED_CF_PATH" \
    --out_manifest_path "$MERGED_MANIFEST_PATH" \
    --name "$DATASET_NAME" \
    --expected_shift_count "$EXPECTED_SHIFT_COUNT" \
    --expected_invariant_count 20 \
    --shift_geometry_filter "$SHIFT_GEOMETRY_FILTER"
fi

[[ -f "$MERGED_CF_PATH" ]] || die "Missing merged CF dataset: $MERGED_CF_PATH"
[[ -f "$MERGED_MANIFEST_PATH" ]] || die "Missing merged dataset manifest: $MERGED_MANIFEST_PATH"

cmd_common=(
  --bootstrap_n "$BOOTSTRAP_N"
  --bootstrap_seed "$BOOTSTRAP_SEED"
  --ci "$CI"
  --dataset_manifest_path "$MERGED_MANIFEST_PATH"
  --error_policy raise
  --max_fail_rate 0
  --strict_data
  --no-require_git
)
if [[ "$LOCAL_FILES_ONLY" -eq 1 ]]; then
  cmd_common+=(--local_files_only)
fi
if [[ -n "$REVISION" ]]; then
  cmd_common+=(--revision "$REVISION")
fi
if [[ -n "$TOKENIZER_REVISION" ]]; then
  cmd_common+=(--tokenizer_revision "$TOKENIZER_REVISION")
fi

run_and_log python aom_cf_patching.py \
  --models "${MODELS[@]}" \
  --device "$DEVICE" \
  --attn_implementation eager \
  --cf_path "$MERGED_CF_PATH" \
  --span_mode left_aligned_truncated \
  --include_expected_effects shift,invariant \
  --sweep_seeds "${SWEEP_SEEDS[@]}" \
  --csv_path "$CSV_PATH" \
  "${cmd_common[@]}"

table_cmd=(
  python tables/table_cf_shift_vs_subinv.py
  --in_csv "$CSV_PATH"
  --out_md "$TABLE_MD_PATH"
  --out_csv "$TABLE_CSV_PATH"
  --out_tex "$TABLE_TEX_PATH"
)
run_and_log "${table_cmd[@]}"

SOURCE_GIT_COMMIT=""
if [[ -f "$ROOT/SOURCE_GIT_COMMIT.txt" ]]; then
  SOURCE_GIT_COMMIT="$(cat "$ROOT/SOURCE_GIT_COMMIT.txt")"
elif git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
fi

python - <<'PY' "$RESULTS_DIR" "$CSV_PATH" "$MERGED_CF_PATH" "$MERGED_MANIFEST_PATH" "$SOURCE_GIT_COMMIT" "$COMMAND_LOG" "$ROOT"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

results_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
merged_cf_path = Path(sys.argv[3])
merged_manifest_path = Path(sys.argv[4])
git_commit = sys.argv[5]
command_log = Path(sys.argv[6])
repo_root = Path(sys.argv[7]).resolve()

commands = [line.strip() for line in command_log.read_text(encoding="utf-8").splitlines() if line.strip()]

from aom.data.dataset_manifest import sha256_file
from aom.repro import collect_versions
from aom.repro_metadata import collect_dependency_install_provenance_from_env, verification_profile_for_mode

def _rel_under_root(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(repo_root))
    except Exception:
        return str(p)

manifest = {
    "mode": "cf_shift_vs_subinv",
    "verification_profile": verification_profile_for_mode("cf_shift_vs_subinv"),
    "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "git_commit": str(git_commit),
    "results_csv_path": _rel_under_root(csv_path),
    "results_csv_sha256": sha256_file(csv_path) if csv_path.exists() else "",
    "merged_cf_path": _rel_under_root(merged_cf_path),
    "merged_cf_sha256": sha256_file(merged_cf_path) if merged_cf_path.exists() else "",
    "dataset_manifest_path": _rel_under_root(merged_manifest_path),
    "dataset_manifest_sha256": sha256_file(merged_manifest_path) if merged_manifest_path.exists() else "",
    "runtime_versions": collect_versions(),
    "commands": commands,
}
manifest.update(collect_dependency_install_provenance_from_env())
(results_dir / "CF_SHIFT_SUBINV_RUN_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "[ok] cf patching CSV: $CSV_PATH"
echo "[ok] table markdown: $TABLE_MD_PATH"
echo "[ok] table csv: $TABLE_CSV_PATH"
echo "[ok] table tex: $TABLE_TEX_PATH"
echo "[ok] run manifest: $RESULTS_DIR/CF_SHIFT_SUBINV_RUN_MANIFEST.json"
