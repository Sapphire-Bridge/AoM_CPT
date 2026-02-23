#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TOTAL_BUDGET_SEC="${TOTAL_BUDGET_SEC:-28800}"
FIELDS_TIMEOUT_SEC="${FIELDS_TIMEOUT_SEC:-1200}"
STRICT_SHA="${STRICT_SHA:-0}"
FREEZE_TAG="${FREEZE_TAG:-}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
RESULTS_DIR="${RESULTS_DIR:-results_submission_full}"
TABLES_DIR="${TABLES_DIR:-tables_submission_full}"
ARCHIVE_NAME="${ARCHIVE_NAME:-aom_replication_bundle_fast8h.tar.gz}"

START_TS="$(date +%s)"

die() {
  echo "[FAIL] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

elapsed_sec() {
  local now
  now="$(date +%s)"
  echo $(( now - START_TS ))
}

remaining_sec() {
  local rem
  rem=$(( TOTAL_BUDGET_SEC - $(elapsed_sec) ))
  if (( rem < 0 )); then
    rem=0
  fi
  echo "$rem"
}

run_with_timeout() {
  local timeout_sec="$1"
  shift
  python3 - "$timeout_sec" "$@" <<'PY'
import subprocess
import sys

timeout_sec = int(sys.argv[1])
cmd = sys.argv[2:]
if timeout_sec <= 0:
    raise SystemExit(124)
try:
    proc = subprocess.run(cmd, timeout=timeout_sec)
    raise SystemExit(proc.returncode)
except subprocess.TimeoutExpired:
    print(f"[FAIL] timeout after {timeout_sec}s: {' '.join(cmd)}", file=sys.stderr)
    raise SystemExit(124)
PY
}

run_step() {
  local label="$1"
  shift
  local rem
  rem="$(remaining_sec)"
  (( rem > 0 )) || die "Budget exhausted before step: $label"
  echo "[run] $label (remaining=${rem}s)"
  run_with_timeout "$rem" "$@"
}

run_step_capped() {
  local label="$1"
  local cap="$2"
  shift 2
  local rem
  local limit
  rem="$(remaining_sec)"
  (( rem > 0 )) || die "Budget exhausted before step: $label"
  limit="$rem"
  if (( cap > 0 && cap < limit )); then
    limit="$cap"
  fi
  echo "[run] $label (timeout=${limit}s remaining=${rem}s)"
  run_with_timeout "$limit" "$@"
}

need_cmd git
need_cmd python3
need_cmd bash
need_cmd make

if [[ "$ALLOW_DIRTY" != "1" ]]; then
  STATUS="$(git status --porcelain || true)"
  [[ -z "$STATUS" ]] || die "Worktree is dirty. Commit/stash first, or run with ALLOW_DIRTY=1."
fi

HEAD_SHA="$(git rev-parse HEAD)"
echo "[info] release gate fast-8h"
echo "[info] head_sha=$HEAD_SHA"
echo "[info] budget_sec=$TOTAL_BUDGET_SEC"
echo "[info] strict_sha=$STRICT_SHA"
echo "[info] results_dir=$RESULTS_DIR"
echo "[info] tables_dir=$TABLES_DIR"
echo "[info] archive=$ARCHIVE_NAME"

for f in \
  "$RESULTS_DIR/aom_eval.csv" \
  "$RESULTS_DIR/cf_patching_shift_vs_subinv.csv" \
  "$RESULTS_DIR/coh_patching_qwen.csv" \
  "$RESULTS_DIR/coh_patching_qwen15.csv" \
  "$RESULTS_DIR/coh_patching_qwen3b.csv"
do
  [[ -f "$f" ]] || die "Missing required artifact: $f"
done

run_step "Preflight E10d/E10e row integrity + device notes" \
  python3 -c '
from pathlib import Path
p = Path("AoM_JoLLLI/AoM_evidence_contract.md")
lines = p.read_text(encoding="utf-8").splitlines()
for eid in ("E10d", "E10e"):
    hits = [ln for ln in lines if ln.lstrip().startswith(f"| `{eid}` |")]
    if len(hits) != 1:
        raise SystemExit(f"[FAIL] {eid} row_lines_found={len(hits)} (expected 1)")
    s = hits[0].lower()
    has_cpu = "cpu" in s
    has_mps = "mps" in s
    if not (has_cpu or has_mps):
        raise SystemExit(f"[FAIL] {eid} notes must include cpu or mps")
    token = "mps" if has_mps else "cpu"
    print(f"[ok] {eid} row parsed; notes_token={token}")
'

run_step "Preflight artifact device auditability (E10d/E10e)" \
  python3 -c '
import csv
from pathlib import Path
import sys

results_dir = Path(sys.argv[1])
contract_path = Path("AoM_JoLLLI/AoM_evidence_contract.md")

def row_expected_device(eid: str) -> str:
    lines = contract_path.read_text(encoding="utf-8").splitlines()
    hits = [ln for ln in lines if ln.lstrip().startswith(f"| `{eid}` |")]
    if len(hits) != 1:
        raise SystemExit(f"[FAIL] {eid} row_lines_found={len(hits)} (expected 1)")
    s = hits[0].lower()
    if "mps" in s:
        return "mps"
    if "cpu" in s:
        return "cpu"
    raise SystemExit(f"[FAIL] {eid} notes must include cpu or mps")

def devs(path: Path):
    out = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("device") or "").strip().lower()
            if d:
                out.add(d)
    return sorted(out)

expected_e10d = row_expected_device("E10d")
expected_e10e = row_expected_device("E10e")

cf_path = results_dir / "cf_patching_shift_vs_subinv.csv"
coh_paths = [
    results_dir / "coh_patching_qwen.csv",
    results_dir / "coh_patching_qwen15.csv",
    results_dir / "coh_patching_qwen3b.csv",
]

cf_devs = devs(cf_path)
if cf_devs != [expected_e10e]:
    raise SystemExit(f"[FAIL] E10e expected [{expected_e10e!r}], got {cf_devs}")
print(f"[ok] E10e device(s): {cf_devs} (from notes={expected_e10e})")

coh_sets = [devs(p) for p in coh_paths]
if any(d != [expected_e10d] for d in coh_sets):
    raise SystemExit(f"[FAIL] E10d expected all [{expected_e10d!r}], got {coh_sets}")
print(f"[ok] E10d device(s): {coh_sets} (from notes={expected_e10d})")
' "$RESULTS_DIR"

run_step "Check evidence contract IDs" python3 scripts/check_evidence_contract.py
run_step_capped "Check evidence contract fields" "$FIELDS_TIMEOUT_SEC" python3 scripts/check_evidence_contract_fields.py
run_step "Pytest (evidence contract suite)" pytest -q tests/test_evidence_contract_ids.py tests/test_evidence_contract_fields.py

run_step "Strict table regeneration" env MAKE_TABLES_STRICT=1 make tables "RESULTS_DIR=$RESULTS_DIR" "TABLES_OUT_DIR=$TABLES_DIR"

if [[ "$STRICT_SHA" == "1" ]]; then
  run_step "STRICT_SHA check (results CSV git_commit == HEAD)" \
    python3 -c '
import csv
from pathlib import Path
import sys
head = sys.argv[1]
results = Path(sys.argv[2])
vals = set()
for p in sorted(results.glob("*.csv")):
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("git_commit") or "").strip()
            if v:
                vals.add(v)
if not vals:
    raise SystemExit("[FAIL] No non-empty git_commit values in CSV artifacts")
if vals != {head}:
    raise SystemExit(f"[FAIL] STRICT_SHA mismatch: head={head} csv_git_commit={sorted(vals)}")
print(f"[ok] STRICT_SHA matched head={head}")
' "$HEAD_SHA" "$RESULTS_DIR"
fi

run_step "Build publication bundle from frozen artifacts" \
  bash scripts/build_replication_bundle.sh \
  --mode submission_full_strong \
  --results-dir "$RESULTS_DIR" \
  --tables-dir "$TABLES_DIR" \
  --archive "$ARCHIVE_NAME" \
  --skip-run

ARCHIVE_SHA="${ARCHIVE_NAME}.sha256"
[[ -f "$ARCHIVE_NAME" ]] || die "Archive not found: $ARCHIVE_NAME"
[[ -f "$ARCHIVE_SHA" ]] || die "Archive sha256 not found: $ARCHIVE_SHA"

if [[ -n "$FREEZE_TAG" ]]; then
  if git rev-parse -q --verify "refs/tags/$FREEZE_TAG" >/dev/null; then
    die "Tag already exists: $FREEZE_TAG"
  fi
  run_step "Create freeze tag" git tag "$FREEZE_TAG" "$HEAD_SHA"
  echo "[ok] created tag=$FREEZE_TAG @ $HEAD_SHA"
fi

TOTAL_ELAPSED="$(elapsed_sec)"
echo "[PASS] release_gate_fast_8h completed"
echo "[PASS] head_sha=$HEAD_SHA"
echo "[PASS] elapsed_sec=$TOTAL_ELAPSED budget_sec=$TOTAL_BUDGET_SEC"
echo "[PASS] archive=$ROOT/$ARCHIVE_NAME"
echo "[PASS] archive_sha256=$ROOT/$ARCHIVE_SHA"
