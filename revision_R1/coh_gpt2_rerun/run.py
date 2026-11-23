"""Replay the 80 original AoM COH triplets. Only native continuation scoring."""

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import mean, median
import subprocess
import sys
from time import perf_counter

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import tokenizers
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
ARCHIVED_COMMIT = "ea24714787ac14d6641f15fc08483a463dfb70b9"
GROUPS = ("main", "ablate_relevant", "ablate_irrelevant")
TOL = 1e-5


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def summarize(rows):
    assert rows
    result = {"n": len(rows)}
    for group in GROUPS:
        result["margin_" + group] = mean(r["margin_" + group] for r in rows)
        result["accuracy_" + group] = mean(r["margin_" + group] > 0 for r in rows)
    for key in ("loss_relevant", "loss_irrelevant", "contrast"):
        result[key] = mean(r[key] for r in rows)
    contrasts = [r["contrast"] for r in rows]
    result.update(positive=sum(d > 0 for d in contrasts), negative=sum(d < 0 for d in contrasts),
                  ties=sum(d == 0 for d in contrasts), median=median(contrasts),
                  minimum=min(contrasts), maximum=max(contrasts))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True, help="archived aom_eval.csv")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, archive, snapshot, out = (p.resolve() for p in (args.repo, args.archive, args.snapshot, args.out))
    out.mkdir(exist_ok=True, parents=True)
    if (out / "summary.json").exists():
        raise ValueError("Select a fresh output directory for a replay.")
    sys.path.insert(0, str(repo))
    from aom.data.loaders import load_coherence_items
    from aom.metrics.disamb import score_labels_next_continuations
    from aom.utils import configure_logprob_computation

    started = datetime.now(timezone.utc).isoformat()
    clock = perf_counter()
    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(repo), *cmd], text=True).strip()
    with archive.open() as f:
        matches = [r for r in csv.DictReader(f) if r["model"] == "gpt2"]
    assert len(matches) == 1
    reference = matches[0]
    data = repo / reference["coh_path"]
    assert digest(data) == reference["coh_sha256"]
    assert reference["hf_model_commit_hash"] == snapshot.name == REV
    paths = ["aom/metrics/coherence.py", "aom/metrics/disamb.py", "aom/utils.py", "aom/data/loaders.py"]
    for path in paths:
        assert not git("diff", ARCHIVED_COMMIT, "--", path), f"Archived source differs: {path}"

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    configure_logprob_computation(logprobs_dtype=torch.float32, strict_finite=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=True, local_files_only=True)
    items = load_coherence_items(str(data))
    families = {}
    for it in items:
        base, group = it.item_id.rsplit("__", 1)
        family = families.setdefault(base, {})
        assert group == it.group and group in GROUPS and group not in family
        family[group] = it
    assert len(items) == 240 and len(families) == 80
    lengths = {}
    for base, family in families.items():
        assert set(family) == set(GROUPS)
        orig = family["main"]
        for it in family.values():
            assert it.constraint_type == orig.constraint_type
            assert (it.valid_continuations, it.invalid_continuations) == (orig.valid_continuations, orig.invalid_continuations)
        lengths[base] = {g: len(tokenizer.encode(it.context, add_special_tokens=False)) for g, it in family.items()}
    matched = [b for b, l in lengths.items() if l["ablate_relevant"] == l["ablate_irrelevant"]]
    states = [b for b in matched if families[b]["main"].constraint_type == "entity_state"]
    assert Counter(families[b]["main"].constraint_type for b in matched) == {"possession": 2, "location": 8, "entity_state": 5}
    assert len(matched) == 15 and len(states) == 5
    write_json(out / "selection.json", {"data_sha256": digest(data), "matched_ids": matched,
               "matched_entity_state_ids": states, "tokens_by_triplet": lengths,
               "rule": "GPT-2 prompt tokens without specials: ablate_relevant == ablate_irrelevant; subset within these: entity_state",
               "estimand": "mean of within-triplet margin(irrelevant replacement) minus margin(relevant replacement)",
               "archived_reproduction_absolute_tolerance": TOL})
    print("Preflight passed: 80 triplets, 15 matched, 5 matched entity_state.", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
                                                  torch_dtype=torch.float32, attn_implementation="eager").to("cpu").eval()
    assert next(model.parameters()).dtype == torch.float32 and model.config._attn_implementation == "eager"
    prompt_rows = []
    triplets = []
    with torch.inference_mode():
        for base, family in families.items():
            r = {"id": base, "frame": family["main"].constraint_type, "matched": base in matched}
            for group in GROUPS:
                it = family[group]
                info = {"id": it.item_id, "group": group, "prompt_tokens": lengths[base][group]}
                for label, alternatives in (("valid", it.valid_continuations), ("invalid", it.invalid_continuations)):
                    choices = {str(i): [text] for i, text in enumerate(alternatives)}
                    scores = score_labels_next_continuations(model, tokenizer, it.context, choices,
                                                             torch.device("cpu"), normalize_by_length=True).by_label
                    info[label] = [{"text": text, "score": scores[str(i)]} for i, text in enumerate(alternatives)]
                margin = mean(a["score"] for a in info["valid"]) - mean(a["score"] for a in info["invalid"])
                assert math.isfinite(margin)
                info["margin"] = r["margin_" + group] = margin
                r["tokens_" + group] = lengths[base][group]
                prompt_rows.append(info)
            r["loss_relevant"] = r["margin_main"] - r["margin_ablate_relevant"]
            r["loss_irrelevant"] = r["margin_main"] - r["margin_ablate_irrelevant"]
            r["contrast"] = r["margin_ablate_irrelevant"] - r["margin_ablate_relevant"]
            triplets.append(r)
            if len(triplets) % 20 == 0:
                print(f"Scored {len(triplets)}/80 triplets", flush=True)

    write_json(out / "prompt_scores.json", prompt_rows)
    with (out / "triplets.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(triplets[0]))
        writer.writeheader()
        writer.writerows(triplets)
    sets = {"all_80": triplets, "matched_15": [r for r in triplets if r["matched"]],
            "matched_entity_state_5": [r for r in triplets if r["id"] in states]}
    results = {name: summarize(rows) for name, rows in sets.items()}
    for name in ("all_80", "matched_15"):
        frames = defaultdict(list)
        for row in sets[name]:
            frames[row["frame"]].append(row)
        results[name]["by_frame"] = {frame: summarize(rows) for frame, rows in frames.items()}
    archived = {"loss_" + key: float(reference["coh_paired_delta_ablate_" + key + "_mean_gap"])
                for key in ("relevant", "irrelevant")}
    archived["contrast"] = archived["loss_relevant"] - archived["loss_irrelevant"]
    for group in GROUPS:
        archived["accuracy_" + group] = float(reference["coh_group_" + group + "_constraint_accuracy"])
    differences = {k: results["all_80"][k] - value for k, value in archived.items()}
    results["reproduction"] = {"archived": archived, "replay_minus_archive": differences,
                               "passed": all(abs(d) <= TOL for d in differences.values()), "tolerance": TOL}
    write_json(out / "summary.json", results)
    files = ["selection.json", "prompt_scores.json", "triplets.csv", "summary.json"]
    write_json(out / "manifest.json", {"started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": perf_counter()-clock, "command": [sys.executable, *sys.argv], "python": sys.version,
        "torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "numpy": np.__version__, "platform": platform.platform(), "hardware": subprocess.getoutput("sysctl -n machdep.cpu.brand_string"),
        "memory_bytes": subprocess.getoutput("sysctl -n hw.memsize"), "threads": torch.get_num_threads(),
        "seed": 0, "device": "cpu", "dtype": "float32", "logprobs_dtype": "float32", "attention": "eager",
        "model": "gpt2", "model_and_tokenizer_revision": REV, "snapshot": str(snapshot), "runner_sha256": digest(__file__),
        "weights_sha256": digest(snapshot/"model.safetensors"), "tokenizer_json_sha256": digest(snapshot/"tokenizer.json"),
        "data_path": str(data), "data_sha256": digest(data), "archive_path": str(archive), "archive_sha256": digest(archive),
        "repo_head": git("rev-parse", "HEAD"), "archive_repo_head": ARCHIVED_COMMIT,
        "scoring_source_files_verified_identical_to_archive": {p: digest(repo/p) for p in paths},
        "behavior_only": True, "hashes": {name: digest(out/name) for name in files}})
    print(json.dumps(results["reproduction"], indent=2), flush=True)


if __name__ == "__main__":
    main()
