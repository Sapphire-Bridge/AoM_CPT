.PHONY: tables sync-paper-tables

RESULTS_DIR ?= results
TABLES_OUT_DIR ?= tables_out

# MAKE_TABLES_STRICT=1 -> fail on missing inputs (no --skip_missing).
# Default -> skip missing inputs but warn to stderr.
tables:
	python scripts/make_tables.py --results_dir $(RESULTS_DIR) --out_dir $(TABLES_OUT_DIR) $(if $(filter 1,$(MAKE_TABLES_STRICT)),,--skip_missing)

sync-paper-tables:
	python scripts/sync_generated_tables_into_paper.py --generated_md $(TABLES_OUT_DIR)/sdh_specificity.md
