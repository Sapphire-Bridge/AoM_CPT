# Zenodo Release Checklist

## 1) Prepare validated artifacts

From a clean worktree:

```bash
TOTAL_BUDGET_SEC=28800 FIELDS_TIMEOUT_SEC=1200 STRICT_SHA=1 ALLOW_DIRTY=0 \
RESULTS_DIR=results_submission_full TABLES_DIR=tables_submission_full \
ARCHIVE_NAME=aom_replication_bundle_fast8h.tar.gz \
bash scripts/release_gate_fast_8h.sh
```

This produces:

- `aom_replication_bundle_fast8h.tar.gz`
- `aom_replication_bundle_fast8h.tar.gz.sha256`

## 2) Create a GitHub release

- Tag the validated commit (for example `v1.0.1`).
- Create a GitHub Release from that tag.
- Attach the bundle `.tar.gz` and `.sha256`.

## 3) Publish to Zenodo

- Ensure this repository is enabled in Zenodo GitHub integration.
- Confirm `.zenodo.json` metadata is present in the tagged commit.
- Trigger Zenodo ingestion from the GitHub release.
- Verify DOI and metadata fields.

## 4) Post-release verification

- Add DOI badge/link to `README.md` and `CITATION.cff` when DOI is minted.
- Confirm archive checksum matches:

```bash
shasum -a 256 aom_replication_bundle_fast8h.tar.gz
```

- Record release commit/tag and DOI in project notes.

## Repro Modes

Use `scripts/build_replication_bundle.sh --repro-mode`:

- `full_recompute`: regenerate artifacts from model runs before bundling.
- `frozen_artifacts`: reuse existing results/tables and rebuild the archive only.
