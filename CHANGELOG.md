# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- Explicit reproducibility mode selection for replication bundles via `--repro-mode` (`full_recompute` or `frozen_artifacts`).
- Public release governance files: `CONTRIBUTING.md`, `SECURITY.md`, `.github/CODEOWNERS`.
- Zenodo release metadata scaffold in `.zenodo.json`.
- GitHub Actions workflows for evidence checks and manual replication-bundle builds.

### Changed
- Replication bundle now includes public-facing metadata files (`LICENSE`, `CITATION.cff`, and optional governance/Zenodo files when present).
- Reproducibility documentation now distinguishes full recompute from frozen-artifact verification.
