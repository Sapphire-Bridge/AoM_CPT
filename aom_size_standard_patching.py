from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from aom.data.loaders import load_counterfactual_pairs_with_manifest
from aom.interventions.patching.base import run_activation_patching
from aom.interventions.patching.size_standard_protocol import (
    SizeStandardNamedSpanProtocol,
    SizeStandardPatchingConfig,
)
from aom.models.loader import load_causal_lm
from aom.repro import ReproConfig, collect_versions, get_git_commit_hash, seed_everything
from aom.run_manifest import build_run_manifest, redact_argv, write_run_manifest
from aom.run_summary import ErrorPolicy, RunSummary, normalize_error_thresholds
from aom.utils import configure_logprob_computation, get_best_device, set_seed
from aom_cf_patching import (
    _infer_input_device,
    _model_param_dtype,
    _parse_int_list,
    _sha256_file,
    write_csv,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(s: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return out.strip("_") or "x"


def _load_size_filter_manifest(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Size filter manifest must be a JSON object: {path}")
    if str(obj.get("manifest_type", "")) != "size_standard_filter_manifest":
        raise ValueError(f"Unexpected size filter manifest type in {path}: {obj.get('manifest_type')!r}")
    return obj


def _filter_entry_for_model(manifest: Mapping[str, Any], model_name: str) -> dict[str, Any] | None:
    by_name = manifest.get("models_by_name", {})
    if isinstance(by_name, Mapping):
        entry = by_name.get(str(model_name))
        if isinstance(entry, Mapping):
            return dict(entry)

    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        return None
    exact = [dict(e) for e in entries if isinstance(e, Mapping) and str(e.get("model", "")) == str(model_name)]
    if len(exact) == 1:
        return exact[0]

    model_slug = _slug(model_name)
    slug_matches = [
        dict(e)
        for e in entries
        if isinstance(e, Mapping) and str(e.get("model_slug", "")) == str(model_slug)
    ]
    if len(slug_matches) == 1:
        return slug_matches[0]
    return None


def _resolve_size_dataset_for_model(
    *,
    model_name: str,
    n_models: int,
    size_path: str | Path,
    filter_manifest: Mapping[str, Any] | None,
    allow_shared_size_path: bool,
    filter_manifest_path: Path | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    if filter_manifest is not None:
        entry = _filter_entry_for_model(filter_manifest, str(model_name))
        if entry is None:
            raise ValueError(f"Size filter manifest has no entry for model {model_name!r}")
        raw_path = str(entry.get("filtered_jsonl", "") or "").strip()
        if not raw_path:
            raise ValueError(f"Size filter manifest entry for {model_name!r} has no filtered_jsonl")
        selected = Path(raw_path)
        if not selected.is_absolute() and filter_manifest_path is not None and not selected.exists():
            selected = filter_manifest_path.parent / selected
        return selected, entry

    if int(n_models) > 1 and not bool(allow_shared_size_path):
        raise ValueError(
            "Multi-model size patching requires --size_filter_manifest or --allow_shared_size_path"
        )
    return Path(str(size_path)), None


def _trace_path_for_attempt(base: Path, *, model_name: str, seed: int, multi_attempt: bool) -> Path:
    if not bool(multi_attempt):
        return base
    suffix = base.suffix or ".csv"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}.{_slug(model_name)}.seed{int(seed)}{suffix}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Size-standard named-span causal patching runner.")

    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--models", nargs="*", type=str, default=None, help="Optional list of models to sweep.")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Activation patching requires eager attention.",
    )
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep_seeds", nargs="*", type=int, default=None)
    p.add_argument(
        "--determinism",
        type=str,
        default="best_effort",
        choices=["strict", "best_effort", "off"],
    )

    p.add_argument("--size_path", type=str, default=str(root / "data_size_standard" / "size_standard.filtered.jsonl"))
    p.add_argument(
        "--size_filter_manifest",
        type=str,
        default="",
        help="Per-behavior-model filter manifest emitted by check_size_standard_acceptance.py.",
    )
    p.add_argument(
        "--allow_shared_size_path",
        action="store_true",
        help="Permit --models sweeps to reuse one --size_path instead of a per-model filter manifest.",
    )
    p.add_argument(
        "--include_expected_effects",
        type=str,
        default="shift",
        help="Comma-separated expected_effect values to include (default: shift).",
    )
    p.add_argument("--patch_layers", type=str, default="", help="Comma-separated layers to patch (default: all).")
    p.add_argument(
        "--include_value_placebo",
        action="store_true",
        help="Also patch the unchanged value_span on standard_swap rows as a digit-span placebo control.",
    )
    p.add_argument("--no_length_norm", action="store_true")
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)

    p.add_argument(
        "--error_policy",
        type=str,
        default="warn_skip",
        choices=["raise", "warn_skip", "skip_silent"],
    )
    p.add_argument(
        "--data_error_policy",
        type=str,
        default="warn_skip",
        choices=["raise", "warn_skip"],
    )
    p.add_argument("--strict_data", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--strict_errors", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max_fail_rate", type=float, default=1.0)
    p.add_argument("--max_skips", type=int, default=-1)
    p.add_argument("--require_git", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--csv_path", type=str, default="")
    p.add_argument(
        "--trace_path",
        type=str,
        default="",
        help="Optional per-case/per-layer trace CSV path. For multi-model/seed runs, model/seed suffixes are added.",
    )
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default="size_standard_patching")
    return p.parse_args()


def main() -> None:
    run_t0 = time.perf_counter()
    run_started_at_utc = _utc_now_iso()
    args = parse_args()
    error_policy: ErrorPolicy = str(getattr(args, "error_policy", "warn_skip"))  # type: ignore[assignment]
    strict_data = bool(getattr(args, "strict_data", False))
    data_error_policy = "raise" if strict_data else str(getattr(args, "data_error_policy", "warn_skip"))

    summary = RunSummary()
    fatal_error: Exception | None = None
    rows: List[Dict[str, Any]] = []
    attempted_expected: int | None = None
    n_models: int | None = None
    n_seeds: int | None = None
    repro: dict[str, Any] | None = None
    versions: dict[str, str] | None = None
    device_backend: str | None = None
    datasets: dict[str, Any] | None = None
    dataset_warnings: list[str] = []

    def _warn(msg: str) -> None:
        if error_policy != "skip_silent":
            print(str(msg), file=sys.stderr, flush=True)

    csv_path = str(getattr(args, "csv_path", "") or "").strip()
    if not csv_path:
        results_dir = Path(str(getattr(args, "results_dir", "results")))
        run_name = str(getattr(args, "run_name", "") or "").strip() or "size_standard_patching"
        csv_path = str(results_dir / f"{run_name}.csv")
    csv_p = Path(csv_path)
    manifest_path = str(csv_p.with_suffix(".manifest.json"))

    def _flush_partial_csv() -> None:
        if not rows:
            return
        ended_at_utc = _utc_now_iso()
        wall_time_sec = float(time.perf_counter() - run_t0)
        for row in rows:
            row["started_at_utc"] = str(run_started_at_utc)
            row["ended_at_utc"] = str(ended_at_utc)
            row["wall_time_sec"] = float(wall_time_sec)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, str(csv_p))

    try:
        import torch

        configure_logprob_computation(logprobs_dtype=getattr(torch, "float32"), strict_finite=True)
        if str(getattr(args, "attn_implementation", "eager")) != "eager":
            raise ValueError("Activation patching requires --attn_implementation eager")

        models = args.models if args.models is not None and len(args.models) > 0 else [args.model_name_or_path]
        seeds = args.sweep_seeds if args.sweep_seeds is not None and len(args.sweep_seeds) > 0 else [args.seed]
        n_models = int(len(models))
        n_seeds = int(len(seeds))
        attempted_expected = int(n_models * n_seeds)
        trace_path_raw = str(getattr(args, "trace_path", "") or "").strip()
        trace_base_path = Path(trace_path_raw) if trace_path_raw else None
        datasets = {}
        size_filter_manifest_path_raw = str(getattr(args, "size_filter_manifest", "") or "").strip()
        size_filter_manifest_path = Path(size_filter_manifest_path_raw) if size_filter_manifest_path_raw else None
        size_filter_manifest = (
            _load_size_filter_manifest(size_filter_manifest_path)
            if size_filter_manifest_path is not None
            else None
        )

        repo_root = Path(__file__).resolve().parent
        git_commit_hash = get_git_commit_hash(repo_root=repo_root, required=bool(getattr(args, "require_git", True)))
        argv_redacted_list = redact_argv(sys.argv)
        argv_redacted_json = json.dumps(argv_redacted_list, ensure_ascii=False)
        argv_sha256 = hashlib.sha256(argv_redacted_json.encode("utf-8")).hexdigest()

        if args.device == "auto":
            base_device = get_best_device()
        else:
            base_device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[args.device])
        device_backend = str(getattr(base_device, "type", str(base_device)))

        determinism = str(getattr(args, "determinism", "best_effort"))
        repro = seed_everything(ReproConfig(seed=int(seeds[0]), determinism=determinism), device=base_device)
        repro["seeds"] = [int(s) for s in seeds]
        versions = collect_versions()

        layer_list = _parse_int_list(str(getattr(args, "patch_layers", "") or ""))
        include_effects_raw = str(getattr(args, "include_expected_effects", "shift"))
        include_effects = tuple(s.strip() for s in include_effects_raw.split(",") if s.strip())
        protocol = SizeStandardNamedSpanProtocol(
            config=SizeStandardPatchingConfig(
                include_expected_effects=include_effects,
                include_value_placebo=bool(getattr(args, "include_value_placebo", False)),
            )
        )

        for model_name in models:
            try:
                selected_size_path, size_filter_entry = _resolve_size_dataset_for_model(
                    model_name=str(model_name),
                    n_models=int(n_models),
                    size_path=str(args.size_path),
                    filter_manifest=size_filter_manifest,
                    allow_shared_size_path=bool(getattr(args, "allow_shared_size_path", False)),
                    filter_manifest_path=size_filter_manifest_path,
                )
                size_items, size_manifest = load_counterfactual_pairs_with_manifest(
                    str(selected_size_path),
                    role="size_standard",
                    error_policy=data_error_policy,
                )
                dataset_key = f"size_standard__{_slug(str(model_name))}"
                datasets[dataset_key] = size_manifest.as_dict()
                datasets[dataset_key]["selected_for_model"] = str(model_name)
                datasets[dataset_key]["selected_size_path"] = str(selected_size_path)
                if size_filter_manifest_path is not None:
                    datasets[dataset_key]["size_filter_manifest"] = str(size_filter_manifest_path)
                invalid = int(datasets[dataset_key].get("n_rows_invalid", 0) or 0)
                total = int(datasets[dataset_key].get("n_rows_total", 0) or 0)
                valid = int(datasets[dataset_key].get("n_rows_valid", 0) or 0)
                if not size_items:
                    raise ValueError(f"Dataset size_standard for model={model_name!r} has zero valid rows")
                if total > 0 and valid == 0:
                    raise ValueError(
                        f"Dataset size_standard for model={model_name!r} has "
                        f"{invalid}/{total} invalid rows and 0 valid rows"
                    )
                if invalid > 0:
                    msg = f"dataset size_standard model={model_name!r}: invalid_rows {invalid} of {total}"
                    dataset_warnings.append(msg)
                    _warn(f"[WARN] {msg}")
            except Exception as e:
                for _s in seeds:
                    summary.record_failure(e)
                _warn(f"[fail] model={model_name!r} size_dataset: {type(e).__name__}: {e}")
                if error_policy == "raise" or data_error_policy == "raise":
                    raise
                continue

            print(
                "Loading model "
                f"model={model_name!r} "
                f"size_path={str(selected_size_path)!r} "
                f"local_files_only={bool(args.local_files_only)} "
                f"torch_dtype={args.torch_dtype!r} "
                f"attn_implementation={args.attn_implementation!r} "
                f"device_map={args.device_map!r}",
                flush=True,
            )
            try:
                loaded = load_causal_lm(
                    model_name,
                    device=base_device,
                    torch_dtype=args.torch_dtype,
                    revision=getattr(args, "revision", None),
                    tokenizer_revision=getattr(args, "tokenizer_revision", None),
                    local_files_only=args.local_files_only,
                    trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
                    attn_implementation=args.attn_implementation,
                    device_map=args.device_map,
                )
            except Exception as e:
                for _s in seeds:
                    summary.record_failure(e)
                _warn(f"[fail] model={model_name!r}: {type(e).__name__}: {e}")
                if error_policy == "raise":
                    raise
                continue

            model = loaded.model
            tokenizer = loaded.tokenizer
            print(f"Loaded {model_name} (arch={loaded.architecture})", flush=True)
            tensor_device = _infer_input_device(model) if args.device_map is not None else base_device

            for s in seeds:
                eval_t0 = time.perf_counter()
                eval_started_at_utc = _utc_now_iso()
                try:
                    set_seed(int(s))
                    attempt_trace_rows: list[dict[str, Any]] | None = [] if trace_base_path is not None else None
                    patch_res = run_activation_patching(
                        model=model,
                        tokenizer=tokenizer,
                        protocol=protocol,
                        items=size_items,
                        device=tensor_device,
                        layers=layer_list,
                        normalize_by_length=not bool(getattr(args, "no_length_norm", False)),
                        ci=float(getattr(args, "ci", 0.95)),
                        bootstrap_n=int(getattr(args, "bootstrap_n", 1000)),
                        bootstrap_seed=int(getattr(args, "bootstrap_seed", 42)),
                        trace_rows=attempt_trace_rows,
                    )
                except Exception as e:
                    summary.record_failure(e)
                    _warn(f"[fail] model={model_name!r} seed={int(s)}: {type(e).__name__}: {e}")
                    if error_policy == "raise":
                        raise
                    continue
                eval_ended_at_utc = _utc_now_iso()
                eval_wall_time_sec = float(time.perf_counter() - eval_t0)
                trace_path_for_row = ""
                trace_sha_for_row = ""
                trace_n_rows = ""
                if trace_base_path is not None and attempt_trace_rows is not None:
                    trace_p = _trace_path_for_attempt(
                        trace_base_path,
                        model_name=str(model_name),
                        seed=int(s),
                        multi_attempt=bool(int(n_models or 0) * int(n_seeds or 0) > 1),
                    )
                    trace_p.parent.mkdir(parents=True, exist_ok=True)
                    write_csv(attempt_trace_rows, str(trace_p))
                    trace_path_for_row = str(trace_p)
                    trace_sha_for_row = _sha256_file(trace_p)
                    trace_n_rows = int(len(attempt_trace_rows))

                row: Dict[str, Any] = {
                    "model": str(model_name),
                    "arch": str(loaded.architecture),
                    "seed": int(s),
                    "torch_version": "" if versions is None else str(versions.get("torch", "")),
                    "transformers_version": "" if versions is None else str(versions.get("transformers", "")),
                    "tokenizers_version": "" if versions is None else str(versions.get("tokenizers", "")),
                    "python_version": "" if versions is None else str(versions.get("python", "")),
                    "size_path": str(selected_size_path),
                    "size_sha256": _sha256_file(Path(str(selected_size_path))),
                    "size_filter_manifest": "" if size_filter_manifest_path is None else str(size_filter_manifest_path),
                    "size_filter_summary": ""
                    if size_filter_entry is None
                    else str(size_filter_entry.get("filtered_summary", "") or ""),
                    "size_filter_model": ""
                    if size_filter_entry is None
                    else str(size_filter_entry.get("model", "") or ""),
                    "size_filter_gate_mode": ""
                    if size_filter_entry is None
                    else str(size_filter_entry.get("gate_mode", "") or ""),
                    "size_filter_n_kept_rows": ""
                    if size_filter_entry is None
                    else int(size_filter_entry.get("n_kept_rows", 0) or 0),
                    "size_include_value_placebo": bool(getattr(args, "include_value_placebo", False)),
                    "device": str(tensor_device),
                    "requested_device": str(getattr(args, "device", "")),
                    "device_map": str(getattr(args, "device_map", "")),
                    "attn_implementation": str(getattr(args, "attn_implementation", "")),
                    "torch_dtype_requested": str(getattr(args, "torch_dtype", "")),
                    "model_param_dtype": _model_param_dtype(model),
                    "hf_revision_requested": str(getattr(args, "revision", "") or ""),
                    "hf_tokenizer_revision_requested": str(getattr(args, "tokenizer_revision", "") or ""),
                    "hf_tokenizer_revision_effective": str(loaded.tokenizer_revision_effective or ""),
                    "hf_local_files_only": bool(getattr(args, "local_files_only", False)),
                    "hf_trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
                    "hf_model_commit_hash": str(loaded.model_commit_hash or ""),
                    "git_commit": str(git_commit_hash),
                    "argv_redacted_json": str(argv_redacted_json),
                    "argv_sha256": str(argv_sha256),
                    "eval_started_at_utc": str(eval_started_at_utc),
                    "eval_ended_at_utc": str(eval_ended_at_utc),
                    "eval_wall_time_sec": float(eval_wall_time_sec),
                    "bootstrap_n": int(getattr(args, "bootstrap_n", 0)),
                    "bootstrap_seed": int(getattr(args, "bootstrap_seed", 0)),
                    "ci": float(getattr(args, "ci", 0.0)),
                    "trace_path": str(trace_path_for_row),
                    "trace_sha256": str(trace_sha_for_row),
                    "trace_n_rows": trace_n_rows,
                }
                row.update({f"size_patch_{k}": v for k, v in patch_res.items()})
                rows.append(row)
                summary.record_success()
                _flush_partial_csv()
                print(
                    f"model={row['model']} seed={row['seed']} "
                    f"size_patch_mean_max_effect={row.get('size_patch_mean_max_effect', float('nan'))}",
                    flush=True,
                )

            del model
            del tokenizer
            del loaded

    except Exception as e:
        fatal_error = e
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"[FATAL] Aborting; run manifest will be written to {manifest_path}", file=sys.stderr, flush=True)

    run_ended_at_utc = _utc_now_iso()
    run_wall_time_sec = float(time.perf_counter() - run_t0)
    for r in rows:
        r["started_at_utc"] = str(run_started_at_utc)
        r["ended_at_utc"] = str(run_ended_at_utc)
        r["wall_time_sec"] = float(run_wall_time_sec)

    csv_sha256: str | None = None
    if rows:
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, str(csv_p))
        if csv_p.exists():
            csv_sha256 = _sha256_file(csv_p)

    max_skips_raw = int(getattr(args, "max_skips", -1))
    strict_errors = bool(getattr(args, "strict_errors", False))
    max_fail_rate, max_skips = normalize_error_thresholds(
        max_fail_rate=float(getattr(args, "max_fail_rate", 1.0)),
        max_skips=max_skips_raw,
        strict_errors=strict_errors,
    )
    max_skips_manifest = int(max_skips_raw if max_skips is None else max_skips)
    if fatal_error is not None:
        run_status = "FAIL"
        run_status_reasons = [f"fatal: {type(fatal_error).__name__}"]
    else:
        run_status, run_status_reasons = summary.evaluate(max_fail_rate=max_fail_rate, max_skips=max_skips)
        if dataset_warnings:
            if run_status == "PASS":
                run_status = "WARN"
            for msg in dataset_warnings:
                if msg not in run_status_reasons:
                    run_status_reasons.append(str(msg))

    if fatal_error is None and attempted_expected is not None and int(summary.attempted) != int(attempted_expected):
        msg = f"attempted {int(summary.attempted)} != attempted_expected {int(attempted_expected)}"
        if run_status == "PASS":
            run_status = "WARN"
        if msg not in run_status_reasons:
            run_status_reasons.append(msg)

    if rows:
        manifest_row: Dict[str, Any] = dict(rows[0])
        if len(rows) != 1:
            manifest_row["_manifest_note"] = f"CSV contains {len(rows)} rows; manifest stores the first row only."
    else:
        manifest_row = {"_manifest_note": "No successful results rows were produced."}
        manifest_row["error_policy"] = str(error_policy)
        manifest_row["strict_errors"] = bool(strict_errors)
        manifest_row["max_fail_rate"] = float(max_fail_rate)
        manifest_row["max_skips"] = int(max_skips_manifest)
        if fatal_error is not None:
            manifest_row["fatal_error_type"] = str(type(fatal_error).__name__)
            manifest_row["fatal_error"] = str(fatal_error)

    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=manifest_row,
        dataset_manifest_path=None,
        csv_path=str(csv_p) if csv_p.exists() else None,
        csv_sha256=str(csv_sha256) if csv_sha256 is not None else None,
        csv_n_rows=int(len(rows)),
    )
    manifest["started_at_utc"] = str(run_started_at_utc)
    manifest["ended_at_utc"] = str(run_ended_at_utc)
    manifest["wall_time_sec"] = float(run_wall_time_sec)
    manifest["error_policy"] = str(error_policy)
    manifest["strict_errors"] = bool(strict_errors)
    manifest["max_fail_rate"] = float(max_fail_rate)
    manifest["max_skips"] = int(max_skips_manifest)
    manifest["data_error_policy"] = str(data_error_policy)
    manifest["strict_data"] = bool(strict_data)
    manifest["attempt_unit"] = "model_seed"
    manifest["attempted_expected"] = attempted_expected
    manifest["n_models"] = n_models
    manifest["n_seeds"] = n_seeds
    if repro is not None:
        manifest["repro"] = dict(repro)
    if versions is not None:
        manifest["versions"] = dict(versions)
    if device_backend is not None:
        manifest["device_backend"] = str(device_backend)
    if datasets is not None:
        manifest["datasets"] = dict(datasets)
    manifest["run_status"] = str(run_status)
    manifest["run_status_reasons"] = list(run_status_reasons)
    manifest["run_summary"] = summary.as_dict()
    write_run_manifest(manifest_path, manifest)

    if run_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
