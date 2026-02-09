from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.io import write_jsonl
from aom.repro import collect_versions, get_git_commit_hash


M1MAX_MODELS: list[str] = [
    "gpt2",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
]

A100_EXTRA_MODELS: list[str] = [
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

CF_PATCHING_MODELS: list[str] = [
    "gpt2",
    "Qwen/Qwen2.5-3B",
]

COH_PATCHING_MODELS: list[str] = [
    "gpt2",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _shlex_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in argv)


def _run(argv: List[str], *, cwd: Path, dry_run: bool) -> None:
    print(_shlex_join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, cwd=str(cwd), check=True)


def _try_git_commit() -> str:
    return get_git_commit_hash(repo_root=ROOT, required=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _parse_int_csv(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        vals.append(int(s))
    return vals


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_paper_dataset(
    *,
    data_dir: Path,
    seed: int,
    disamb_mode: str = "hardened",
    cf_include_shams: bool = True,
    cf_include_graded: bool = True,
    n_coh: int = 80,
    coh_include_controls: bool = True,
    dry_run: bool,
) -> Path:
    """
    Ensure a paper-canonical dataset exists, and write a deterministic manifest.

    This is intentionally separate from the repo's checked-in `data/` so you can pin a "paper dataset"
    without overwriting examples.
    """
    disamb_path = data_dir / "disamb_pairs.jsonl"
    cf_path = data_dir / "counterfactual.jsonl"
    coh_path = data_dir / "coherence.jsonl"

    files_exist = bool(disamb_path.exists() and cf_path.exists() and coh_path.exists())

    if not files_exist:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "generate_data.py"),
            "--out_dir",
            str(data_dir),
            "--seed",
            str(int(seed)),
            "--disamb_mode",
            str(disamb_mode),
            "--n_coh",
            str(int(n_coh)),
        ]
        if cf_include_shams:
            cmd.append("--cf_include_shams")
        if cf_include_graded:
            cmd.append("--cf_include_graded")
        if coh_include_controls:
            cmd.append("--coh_include_controls")
        _run(cmd, cwd=ROOT, dry_run=dry_run)

    manifest_path = data_dir / "DATASET_MANIFEST.json"
    files_exist = bool(disamb_path.exists() and cf_path.exists() and coh_path.exists())
    if files_exist:
        from collections import Counter

        from aom.data.bundle_manifest import compute_bundle_id

        def _paper_bundle_name() -> str:
            cfg = (
                str(disamb_mode),
                bool(cf_include_shams),
                bool(cf_include_graded),
                int(n_coh),
                bool(coh_include_controls),
            )
            if cfg == ("hardened", True, False, 40, True):
                return "paper_hardened_v1"
            if cfg == ("hardened", True, True, 80, True):
                return "paper_hardened_v2"
            if str(disamb_mode) == "hardened":
                return "paper_hardened_custom"
            return "paper_dataset"

        def _iter_jsonl(path: Path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

        def _rel(path: Path) -> str:
            try:
                return str(path.resolve().relative_to(ROOT.resolve()))
            except Exception:
                try:
                    return str(path.relative_to(ROOT))
                except Exception:
                    return str(path.name)

        files = {
            "disamb_pairs.jsonl": {
                "path": _rel(disamb_path),
                "n_lines": int(_count_nonempty_lines(disamb_path)),
                "sha256": _sha256_file(disamb_path),
            },
            "counterfactual.jsonl": {
                "path": _rel(cf_path),
                "n_lines": int(_count_nonempty_lines(cf_path)),
                "sha256": _sha256_file(cf_path),
            },
            "coherence.jsonl": {
                "path": _rel(coh_path),
                "n_lines": int(_count_nonempty_lines(coh_path)),
                "sha256": _sha256_file(coh_path),
            },
        }
        bundle_id = compute_bundle_id({"files": files})

        dis_pair_variant = Counter()
        for row in _iter_jsonl(disamb_path):
            meta = row.get("metadata", None)
            if isinstance(meta, dict):
                dis_pair_variant[str(meta.get("pair_variant", ""))] += 1
        cf_expected_effect = Counter()
        for row in _iter_jsonl(cf_path):
            cf_expected_effect[str(row.get("expected_effect", "shift"))] += 1
        coh_group = Counter()
        coh_constraint_type = Counter()
        coh_n_constraints = Counter()
        coh_constraint_type_main = Counter()
        coh_n_constraints_main = Counter()
        for row in _iter_jsonl(coh_path):
            grp = str(row.get("group", "main"))
            coh_group[grp] += 1
            coh_constraint_type[str(row.get("constraint_type", ""))] += 1
            meta = row.get("metadata", None)
            nc = 1
            if isinstance(meta, dict):
                try:
                    nc = int(meta.get("n_constraints", 1))
                except Exception:
                    nc = 1
            coh_n_constraints[str(int(nc))] += 1
            if grp == "main":
                coh_constraint_type_main[str(row.get("constraint_type", ""))] += 1
                coh_n_constraints_main[str(int(nc))] += 1

        suite_counts = {
            "disamb": {
                "n_pairs_total": int(files["disamb_pairs.jsonl"]["n_lines"]),
                "pair_variant_counts": {str(k): int(v) for k, v in sorted(dis_pair_variant.items())},
            },
            "cf": {
                "n_items_total": int(files["counterfactual.jsonl"]["n_lines"]),
                "expected_effect_counts": {str(k): int(v) for k, v in sorted(cf_expected_effect.items())},
            },
            "coh": {
                "n_rows_total": int(files["coherence.jsonl"]["n_lines"]),
                "group_counts": {str(k): int(v) for k, v in sorted(coh_group.items())},
                "constraint_type_counts": {str(k): int(v) for k, v in sorted(coh_constraint_type.items())},
                "n_constraints_counts": {str(k): int(v) for k, v in sorted(coh_n_constraints.items())},
                "main": {
                    "n_items": int(coh_group.get("main", 0)),
                    "constraint_type_counts": {str(k): int(v) for k, v in sorted(coh_constraint_type_main.items())},
                    "n_constraints_counts": {str(k): int(v) for k, v in sorted(coh_n_constraints_main.items())},
                },
            },
        }

        expected_cfg = {
            "name": _paper_bundle_name(),
            "seed": int(seed),
            "disamb_mode": str(disamb_mode),
            "cf_include_shams": bool(cf_include_shams),
            "cf_include_graded": bool(cf_include_graded),
            "n_coh": int(n_coh),
            "coh_include_controls": bool(coh_include_controls),
        }

        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError(f"Existing DATASET_MANIFEST.json is not a JSON object: {str(manifest_path)}")
            mismatches = []
            for k, v in expected_cfg.items():
                if k in existing and existing.get(k) != v:
                    mismatches.append(f"{k}: have={existing.get(k)!r} want={v!r}")
            for filename, info in files.items():
                have = existing.get("files", {}).get(filename, {}).get("sha256", None)
                if have is not None and str(have).strip() != str(info["sha256"]).strip():
                    mismatches.append(f"files.{filename}.sha256: have={str(have).strip()} want={str(info['sha256']).strip()}")
            if mismatches:
                raise ValueError(
                    "Dataset directory already contains a different dataset/manifest configuration. "
                    f"Use a fresh --data_dir or delete the existing one. Mismatches: {mismatches}"
                )

            updated = dict(existing)
            for k, v in expected_cfg.items():
                updated.setdefault(k, v)
            updated.setdefault("generator", "scripts/generate_data.py")
            updated.setdefault("git_commit", _try_git_commit())
            updated.setdefault("generated_at_utc", _utc_now_iso())
            updated["files"] = files
            updated["bundle_id"] = str(bundle_id)
            updated["suite_counts"] = suite_counts
            if not dry_run and updated != existing:
                _write_json(manifest_path, updated)
        else:
            manifest = {
                **expected_cfg,
                "generated_at_utc": _utc_now_iso(),
                "git_commit": _try_git_commit(),
                "generator": "scripts/generate_data.py",
                "files": files,
                "bundle_id": str(bundle_id),
                "suite_counts": suite_counts,
            }
            if not dry_run:
                _write_json(manifest_path, manifest)

    return manifest_path


def _make_wordlevel_tokenizer_dir(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "[PAD]": 0,
        "[EOS]": 1,
        "[UNK]": 2,
        "Alice": 3,
        "went": 4,
        "to": 5,
        "the": 6,
        "bank": 7,
        "and": 8,
        "then": 9,
        "sat": 10,
        "by": 11,
        "river": 12,
        "loan": 13,
        "did": 14,
        "not": 15,
        "go": 16,
        "store": 17,
        "Therefore": 18,
        "John": 19,
        "died": 20,
        "Later": 21,
        "was": 22,
        "remembered": 23,
        "walked": 24,
        "quietly": 25,
        "today": 26,
    }

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)


def _make_tiny_local_gpt2_model_dir(path: Path) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=32,
        vocab_size=64,
        n_positions=64,
        pad_token_id=0,
        eos_token_id=1,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    model.save_pretrained(path)


def ensure_smoke_local_model(model_dir: Path) -> None:
    """
    Create a tiny local model+tokenizer so `aom_eval.py` can run fully offline.
    """
    if (model_dir / "config.json").exists() and (model_dir / "tokenizer.json").exists():
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    _make_wordlevel_tokenizer_dir(model_dir)
    _make_tiny_local_gpt2_model_dir(model_dir)


def ensure_smoke_datasets(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    disamb_path = data_dir / "disamb_pairs.jsonl"
    cf_path = data_dir / "counterfactual.jsonl"
    coh_path = data_dir / "coherence.jsonl"

    if not disamb_path.exists():
        write_jsonl(
            [
                {
                    "pair_id": "bank-smoke-0",
                    "target": "bank",
                    "target_occurrence": 0,
                    "a": {"prompt": "Alice went to the bank and then Alice went to the store Therefore Alice", "expected_label": "loan"},
                    "b": {"prompt": "Alice sat by the bank and then Alice sat by the river Later Alice", "expected_label": "river"},
                    "choices": {"loan": [" loan"], "river": [" river"]},
                    "metadata": {"type": "smoke"},
                }
            ],
            disamb_path,
        )

    if not cf_path.exists():
        write_jsonl(
            [
                {
                    "item_id": "negation-smoke-0",
                    "base": {"prompt": "Alice did go to the store Therefore Alice", "expected_label": "did"},
                    "cf": {"prompt": "Alice did not go to the store Therefore Alice", "expected_label": "not"},
                    "choices": {"did": [" did"], "not": [" not"]},
                    "intervention_type": "negation",
                    "contrast_labels": ["did", "not"],
                    "expected_effect": "shift",
                    "metadata": {"type": "smoke"},
                },
                {
                    "item_id": "negation-smoke-0__sham",
                    "base": {"prompt": "Alice did go to the store Therefore Alice", "expected_label": "did"},
                    "cf": {"prompt": "Alice did go to the store Therefore, Alice", "expected_label": "did"},
                    "choices": {"did": [" did"], "not": [" not"]},
                    "intervention_type": "sham_punctuation",
                    "contrast_labels": ["did", "not"],
                    "expected_effect": "invariant",
                    "metadata": {"type": "smoke", "control": "sham"},
                },
            ],
            cf_path,
        )

    if not coh_path.exists():
        write_jsonl(
            [
                {
                    "item_id": "entity-smoke-0__main",
                    "context": "John died Later John",
                    "valid_continuations": [" was remembered"],
                    "invalid_continuations": [" walked"],
                    "constraint_type": "entity_state",
                    "group": "main",
                    "metadata": {"type": "smoke"},
                }
            ],
            coh_path,
        )

    return {"disamb": disamb_path, "cf": cf_path, "coh": coh_path}


@dataclass(frozen=True)
class PaperRun:
    mode: str
    results_dir: Path
    dataset_manifest_path: Optional[Path]
    commands: List[List[str]]


def _write_run_manifest(run: PaperRun) -> None:
    req_txt = ROOT / "requirements.txt"
    req_lock = ROOT / "requirements.lock.txt"
    out = {
        "mode": str(run.mode),
        "generated_at_utc": _utc_now_iso(),
        "git_commit": _try_git_commit(),
        "results_dir": str(run.results_dir),
        "dataset_manifest_path": "" if run.dataset_manifest_path is None else str(run.dataset_manifest_path),
        "runtime_versions": collect_versions(),
        "requirements_txt_sha256": "" if not req_txt.exists() else _sha256_file(req_txt),
        "requirements_lock_sha256": "" if not req_lock.exists() else _sha256_file(req_lock),
        "commands": [list(cmd) for cmd in run.commands],
    }
    _write_json(run.results_dir / "RUN_MANIFEST.json", out)


def _aom_eval_cmd(
    *,
    models: List[str],
    device: str,
    torch_dtype: Optional[str],
    attn_implementation: str,
    local_files_only: bool,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    disamb_path: str,
    cf_path: str,
    coh_path: str,
    dataset_manifest_path: Optional[str] = None,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    csv_path: str,
    run_patching: bool = False,
    patch_layers: str = "",
    run_patching_specificity: bool = False,
    patch_specificity_depth_frac: Optional[float] = None,
    patch_specificity_layer: Optional[int] = None,
    patch_specificity_buffer: Optional[int] = None,
    patch_specificity_position_window: Optional[int] = None,
    patch_specificity_seed: Optional[int] = None,
    device_map: Optional[str] = None,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_eval.py")]
    argv += ["--models", *models]
    argv += ["--device", str(device)]
    argv += ["--attn_implementation", str(attn_implementation)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if device_map is not None:
        argv += ["--device_map", str(device_map)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if revision is not None and str(revision).strip():
        argv += ["--revision", str(revision)]
    if tokenizer_revision is not None and str(tokenizer_revision).strip():
        argv += ["--tokenizer_revision", str(tokenizer_revision)]
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--cf_path", str(cf_path)]
    argv += ["--coh_path", str(coh_path)]
    if dataset_manifest_path is not None and str(dataset_manifest_path).strip():
        argv += ["--dataset_manifest_path", str(dataset_manifest_path)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    if run_patching:
        argv.append("--run_patching")
        if str(patch_layers).strip():
            argv += ["--patch_layers", str(patch_layers)]
    if run_patching_specificity:
        argv.append("--run_patching_specificity")
        if patch_specificity_layer is not None:
            argv += ["--patch_specificity_layer", str(int(patch_specificity_layer))]
        if patch_specificity_depth_frac is not None:
            argv += ["--patch_specificity_depth_frac", str(float(patch_specificity_depth_frac))]
        if patch_specificity_buffer is not None:
            argv += ["--patch_specificity_buffer", str(int(patch_specificity_buffer))]
        if patch_specificity_position_window is not None:
            argv += ["--patch_specificity_position_window", str(int(patch_specificity_position_window))]
        if patch_specificity_seed is not None:
            argv += ["--patch_specificity_seed", str(int(patch_specificity_seed))]
    argv += ["--csv_path", str(csv_path)]
    return argv


def _cf_patching_cmd(
    *,
    models: List[str],
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    cf_path: str,
    dataset_manifest_path: Optional[str] = None,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    csv_path: str,
    patch_layers: str = "",
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_cf_patching.py")]
    argv += ["--models", *models]
    argv += ["--device", str(device)]
    argv += ["--attn_implementation", "eager"]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if revision is not None and str(revision).strip():
        argv += ["--revision", str(revision)]
    if tokenizer_revision is not None and str(tokenizer_revision).strip():
        argv += ["--tokenizer_revision", str(tokenizer_revision)]
    argv += ["--cf_path", str(cf_path)]
    if dataset_manifest_path is not None and str(dataset_manifest_path).strip():
        argv += ["--dataset_manifest_path", str(dataset_manifest_path)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    if str(patch_layers).strip():
        argv += ["--patch_layers", str(patch_layers)]
    argv += ["--csv_path", str(csv_path)]
    return argv


def _coh_patching_cmd(
    *,
    models: List[str],
    device: str,
    torch_dtype: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    coh_path: str,
    dataset_manifest_path: Optional[str] = None,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    csv_path: str,
    patch_layers: str = "",
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_coh_patching.py")]
    argv += ["--models", *models]
    argv += ["--device", str(device)]
    argv += ["--attn_implementation", "eager"]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    if revision is not None and str(revision).strip():
        argv += ["--revision", str(revision)]
    if tokenizer_revision is not None and str(tokenizer_revision).strip():
        argv += ["--tokenizer_revision", str(tokenizer_revision)]
    argv += ["--coh_path", str(coh_path)]
    if dataset_manifest_path is not None and str(dataset_manifest_path).strip():
        argv += ["--dataset_manifest_path", str(dataset_manifest_path)]
    argv += ["--bootstrap_n", str(int(bootstrap_n))]
    argv += ["--bootstrap_seed", str(int(bootstrap_seed))]
    argv += ["--ci", str(float(ci))]
    if str(patch_layers).strip():
        argv += ["--patch_layers", str(patch_layers)]
    argv += ["--csv_path", str(csv_path)]
    return argv


def run_smoke(args: argparse.Namespace) -> PaperRun:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = None
    if not bool(args.skip_dataset):
        dataset_manifest_path = ensure_paper_dataset(
            data_dir=Path(args.data_dir),
            seed=int(args.dataset_seed),
            disamb_mode="hardened",
            cf_include_shams=True,
            coh_include_controls=True,
            dry_run=bool(args.dry_run),
        )

    smoke_model_dir = results_dir / "local_model"
    smoke_data_dir = results_dir / "smoke_data"
    smoke_manifest_path = smoke_data_dir / "DATASET_MANIFEST.json"
    if not bool(args.dry_run):
        ensure_smoke_local_model(smoke_model_dir)
        smoke_paths = ensure_smoke_datasets(smoke_data_dir)
        files = {
            "disamb_pairs.jsonl": {
                "path": str(smoke_paths["disamb"]),
                "n_lines": int(_count_nonempty_lines(smoke_paths["disamb"])),
                "sha256": _sha256_file(smoke_paths["disamb"]),
            },
            "counterfactual.jsonl": {
                "path": str(smoke_paths["cf"]),
                "n_lines": int(_count_nonempty_lines(smoke_paths["cf"])),
                "sha256": _sha256_file(smoke_paths["cf"]),
            },
            "coherence.jsonl": {
                "path": str(smoke_paths["coh"]),
                "n_lines": int(_count_nonempty_lines(smoke_paths["coh"])),
                "sha256": _sha256_file(smoke_paths["coh"]),
            },
        }
        from aom.data.bundle_manifest import compute_bundle_id

        smoke_manifest = {
            "name": "smoke_bundle",
            "generated_at_utc": _utc_now_iso(),
            "git_commit": _try_git_commit(),
            "generator": "scripts/run_paper.py:ensure_smoke_datasets",
            "files": files,
            "bundle_id": compute_bundle_id({"files": files}),
        }
        _write_json(smoke_manifest_path, smoke_manifest)
    else:
        smoke_paths = {
            "disamb": smoke_data_dir / "disamb_pairs.jsonl",
            "cf": smoke_data_dir / "counterfactual.jsonl",
            "coh": smoke_data_dir / "coherence.jsonl",
        }

    commands: List[List[str]] = []
    csv_path = results_dir / "aom_eval.csv"
    cmd = _aom_eval_cmd(
        models=[str(smoke_model_dir)],
        device="cpu",
        torch_dtype="float32",
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        disamb_path=str(smoke_paths["disamb"]),
        cf_path=str(smoke_paths["cf"]),
        coh_path=str(smoke_paths["coh"]),
        dataset_manifest_path=str(smoke_manifest_path) if not bool(args.dry_run) else None,
        bootstrap_n=50,
        bootstrap_seed=42,
        ci=0.95,
        run_patching=True,
        patch_layers="0",
        run_patching_specificity=True,
        patch_specificity_depth_frac=0.25,
        patch_specificity_buffer=2,
        patch_specificity_position_window=8,
        patch_specificity_seed=0,
        csv_path=str(csv_path),
    )
    commands.append(cmd)
    _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    report_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "report_results.py"),
        "--results_dir",
        str(results_dir),
        "--out_path",
        str(results_dir / "results_report.md"),
    ]
    commands.append(report_cmd)
    _run(report_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    run = PaperRun(mode="smoke", results_dir=results_dir, dataset_manifest_path=dataset_manifest_path, commands=commands)
    if not bool(args.dry_run):
        _write_run_manifest(run)
    return run


def run_m1max(args: argparse.Namespace) -> PaperRun:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.data_dir)
    dataset_manifest_path: Optional[Path] = None
    if not bool(args.skip_dataset):
        dataset_manifest_path = ensure_paper_dataset(
            data_dir=dataset_dir,
            seed=int(args.dataset_seed),
            disamb_mode="hardened",
            cf_include_shams=True,
            coh_include_controls=True,
            dry_run=bool(args.dry_run),
        )
    else:
        p = dataset_dir / "DATASET_MANIFEST.json"
        dataset_manifest_path = p if p.exists() else None

    models = list(M1MAX_MODELS)
    commands: List[List[str]] = []

    disamb_path = dataset_dir / "disamb_pairs.jsonl"
    cf_path = dataset_dir / "counterfactual.jsonl"
    coh_path = dataset_dir / "coherence.jsonl"

    # 1) Full AoM behavioral suite on MPS (fast attention allowed).
    behavioral_csv = results_dir / "aom_eval.csv"
    cmd = _aom_eval_cmd(
        models=models,
        device="mps",
        torch_dtype="float16",
        attn_implementation="sdpa",
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        revision=getattr(args, "revision", None),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        disamb_path=str(disamb_path),
        cf_path=str(cf_path),
        coh_path=str(coh_path),
        dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
        ci=float(args.ci),
        csv_path=str(behavioral_csv),
        device_map=str(args.device_map) if args.device_map else None,
    )
    commands.append(cmd)
    _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 1b) CF/COH causal patching (eager attention required).
    if not bool(getattr(args, "skip_cf_patching", False)):
        cf_patching_csv = results_dir / "cf_patching.csv"
        cmd = _cf_patching_cmd(
            models=list(CF_PATCHING_MODELS),
            device="mps",
            torch_dtype="float16",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            cf_path=str(cf_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(cf_patching_csv),
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))
    if not bool(getattr(args, "skip_coh_patching", False)):
        coh_patching_csv = results_dir / "coh_patching.csv"
        cmd = _coh_patching_cmd(
            models=list(COH_PATCHING_MODELS),
            device="mps",
            torch_dtype="float16",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            coh_path=str(coh_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(coh_patching_csv),
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 2) CPT layer sweep on "small" models only (DISAMB-only, eager attention required).
    if not bool(args.skip_patching):
        sweep_models = ["gpt2", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
        sweep_csv = results_dir / "cpt_layer_sweep_disamb_only.csv"
        cmd = _aom_eval_cmd(
            models=sweep_models,
            device="mps",
            torch_dtype="float16",
            attn_implementation="eager",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            disamb_path=str(disamb_path),
            cf_path="",
            coh_path="",
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(sweep_csv),
            run_patching=True,
            patch_layers="",
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 3) CPT target-specificity control (DISAMB-only, fixed depth).
    if not bool(args.skip_specificity):
        seeds = _parse_int_csv(str(getattr(args, "specificity_selection_seeds", "0")))
        if not seeds:
            seeds = [0]
        for spec_seed in seeds:
            if len(seeds) > 1:
                spec_csv = results_dir / f"cpt_specificity_seed{int(spec_seed)}_disamb_only.csv"
            else:
                spec_csv = results_dir / "cpt_specificity_disamb_only.csv"
            cmd = _aom_eval_cmd(
                models=models,
                device="mps",
                torch_dtype="float16",
                attn_implementation="eager",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                revision=getattr(args, "revision", None),
                tokenizer_revision=getattr(args, "tokenizer_revision", None),
                disamb_path=str(disamb_path),
                cf_path="",
                coh_path="",
                dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                csv_path=str(spec_csv),
                run_patching_specificity=True,
                patch_specificity_depth_frac=float(getattr(args, "specificity_depth_frac", 0.25)),
                patch_specificity_buffer=int(getattr(args, "specificity_buffer", 2)),
                patch_specificity_position_window=int(getattr(args, "specificity_position_window", 8)),
                patch_specificity_seed=int(spec_seed),
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    report_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "report_results.py"),
        "--results_dir",
        str(results_dir),
        "--out_path",
        str(results_dir / "results_report.md"),
    ]
    commands.append(report_cmd)
    _run(report_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    run = PaperRun(mode="m1max", results_dir=results_dir, dataset_manifest_path=dataset_manifest_path, commands=commands)
    if not bool(args.dry_run):
        _write_run_manifest(run)
    return run


def run_a100(args: argparse.Namespace) -> PaperRun:
    if bool(args.device_map) and not bool(args.skip_patching):
        raise ValueError("For patching runs, avoid --device_map (patching assumes a single-device model). Use --skip_patching.")
    if bool(args.device_map) and not bool(args.skip_specificity):
        raise ValueError(
            "For patching-specificity runs, avoid --device_map (patching assumes a single-device model). Use --skip_specificity."
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.data_dir)
    dataset_manifest_path: Optional[Path] = None
    if not bool(args.skip_dataset):
        dataset_manifest_path = ensure_paper_dataset(
            data_dir=dataset_dir,
            seed=int(args.dataset_seed),
            disamb_mode="hardened",
            cf_include_shams=True,
            coh_include_controls=True,
            dry_run=bool(args.dry_run),
        )
    else:
        p = dataset_dir / "DATASET_MANIFEST.json"
        dataset_manifest_path = p if p.exists() else None

    models = list(M1MAX_MODELS) + list(A100_EXTRA_MODELS)
    commands: List[List[str]] = []

    disamb_path = dataset_dir / "disamb_pairs.jsonl"
    cf_path = dataset_dir / "counterfactual.jsonl"
    coh_path = dataset_dir / "coherence.jsonl"

    # 1) Full AoM behavioral suite (fast attention encouraged).
    behavioral_csv = results_dir / "aom_eval.csv"
    cmd = _aom_eval_cmd(
        models=models,
        device="cuda",
        torch_dtype="bfloat16",
        attn_implementation=str(args.attn_behavioral),
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        revision=getattr(args, "revision", None),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        disamb_path=str(disamb_path),
        cf_path=str(cf_path),
        coh_path=str(coh_path),
        dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
        ci=float(args.ci),
        csv_path=str(behavioral_csv),
        device_map=str(args.device_map) if args.device_map else None,
    )
    commands.append(cmd)
    _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 1b) CF/COH causal patching (eager attention required).
    if not bool(getattr(args, "skip_cf_patching", False)):
        cf_patching_csv = results_dir / "cf_patching.csv"
        cmd = _cf_patching_cmd(
            models=list(CF_PATCHING_MODELS),
            device="cuda",
            torch_dtype="bfloat16",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            cf_path=str(cf_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(cf_patching_csv),
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))
    if not bool(getattr(args, "skip_coh_patching", False)):
        coh_patching_csv = results_dir / "coh_patching.csv"
        cmd = _coh_patching_cmd(
            models=list(COH_PATCHING_MODELS),
            device="cuda",
            torch_dtype="bfloat16",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            coh_path=str(coh_path),
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(coh_patching_csv),
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 2) CPT layer sweep across all models (DISAMB-only, eager attention required).
    if not bool(args.skip_patching):
        sweep_csv = results_dir / "cpt_layer_sweep_disamb_only.csv"
        cmd = _aom_eval_cmd(
            models=models,
            device="cuda",
            torch_dtype="bfloat16",
            attn_implementation="eager",
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            revision=getattr(args, "revision", None),
            tokenizer_revision=getattr(args, "tokenizer_revision", None),
            disamb_path=str(disamb_path),
            cf_path="",
            coh_path="",
            dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            csv_path=str(sweep_csv),
            run_patching=True,
            patch_layers="",
        )
        commands.append(cmd)
        _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    # 3) CPT target-specificity control (DISAMB-only).
    if not bool(args.skip_specificity):
        seeds = _parse_int_csv(str(getattr(args, "specificity_selection_seeds", "0")))
        if not seeds:
            seeds = [0]
        for spec_seed in seeds:
            if len(seeds) > 1:
                spec_csv = results_dir / f"cpt_specificity_seed{int(spec_seed)}_disamb_only.csv"
            else:
                spec_csv = results_dir / "cpt_specificity_disamb_only.csv"
            cmd = _aom_eval_cmd(
                models=models,
                device="cuda",
                torch_dtype="bfloat16",
                attn_implementation="eager",
                local_files_only=bool(args.local_files_only),
                trust_remote_code=bool(args.trust_remote_code),
                revision=getattr(args, "revision", None),
                tokenizer_revision=getattr(args, "tokenizer_revision", None),
                disamb_path=str(disamb_path),
                cf_path="",
                coh_path="",
                dataset_manifest_path=str(dataset_manifest_path) if dataset_manifest_path is not None else None,
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
                ci=float(args.ci),
                csv_path=str(spec_csv),
                run_patching_specificity=True,
                patch_specificity_depth_frac=float(getattr(args, "specificity_depth_frac", 0.25)),
                patch_specificity_buffer=int(getattr(args, "specificity_buffer", 2)),
                patch_specificity_position_window=int(getattr(args, "specificity_position_window", 8)),
                patch_specificity_seed=int(spec_seed),
            )
            commands.append(cmd)
            _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    report_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "report_results.py"),
        "--results_dir",
        str(results_dir),
        "--out_path",
        str(results_dir / "results_report.md"),
    ]
    commands.append(report_cmd)
    _run(report_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    run = PaperRun(mode="a100", results_dir=results_dir, dataset_manifest_path=dataset_manifest_path, commands=commands)
    if not bool(args.dry_run):
        _write_run_manifest(run)
    return run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paper-style evaluations in three reproducible modes.")
    p.add_argument("mode", type=str, choices=["smoke", "m1max", "a100"])
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data_paper_hardened_v2"))
    p.add_argument("--results_dir", type=str, default="")
    p.add_argument("--dataset_seed", type=int, default=0)

    p.add_argument("--local_files_only", action="store_true", help="Disallow model downloads (offline mode).")
    p.add_argument("--trust_remote_code", action="store_true", help="Allow custom HF model code (use with care).")
    p.add_argument("--revision", type=str, default=None, help="Optional HF model revision (branch/tag/commit SHA).")
    p.add_argument(
        "--tokenizer_revision",
        type=str,
        default=None,
        help="Optional HF tokenizer revision (defaults to --revision when unset).",
    )
    p.add_argument("--device_map", type=str, default=None, help="Device map for multi-GPU (behavioral runs only).")

    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)

    p.add_argument("--attn_behavioral", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--specificity_depth_frac", type=float, default=0.25)
    p.add_argument("--specificity_buffer", type=int, default=2)
    p.add_argument("--specificity_position_window", type=int, default=8)
    p.add_argument(
        "--specificity_selection_seeds",
        type=str,
        default="0",
        help="Comma-separated selection seeds for specificity control off-target pair selection (default: 0).",
    )

    p.add_argument("--skip_dataset", action="store_true", help="Skip dataset generation/manifest writing.")
    p.add_argument("--skip_patching", action="store_true", help="Skip CPT layer-sweep patching runs.")
    p.add_argument("--skip_specificity", action="store_true", help="Skip CPT target-specificity runs.")
    p.add_argument("--skip_cf_patching", action="store_true", help="Skip AoM-CF causal patching runs.")
    p.add_argument("--skip_coh_patching", action="store_true", help="Skip AoM-COH causal patching runs.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_dir:
        args.results_dir = str(ROOT / "results" / f"paper_{args.mode}")

    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "m1max":
        run_m1max(args)
    elif args.mode == "a100":
        run_a100(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")


if __name__ == "__main__":
    main()
