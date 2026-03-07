#!/usr/bin/env python3
"""B1a: Design and validate substitution-type invariant CF items.

These items have non-empty base spans under the CF patching span definition.
The intervention is a synonym/paraphrase substitution that preserves meaning.
Expected effect: invariant (no label flip).

Validates that:
1. Base and CF tokenize to sequences with a non-empty divergence span on both sides
2. Span lengths match (ideal) or are close (acceptable under left_aligned_truncated)
3. Items pass the same validation as existing CF items

Usage:
    python scripts/b1a_substitution_invariants.py --validate_only
    python scripts/b1a_substitution_invariants.py --output data_paper_hardened_v2/counterfactual_invariant_sub.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aom.token_diff import divergence_span

# ──────────────────────────────────────────────────────────────
# Substitution-type invariant items
# Design principle: synonym or syntactic rearrangement in the
# intervention span that preserves meaning and expected label.
# The divergent span must be non-empty on both base and CF sides.
# ──────────────────────────────────────────────────────────────

ITEMS: List[Dict] = [
    # === Role-swap analogs: verb synonym ===
    {
        "item_id": "sub-inv-role-verb-0",
        "base": {"prompt": "The teacher attacked the student. The attacker was the", "expected_label": "teacher"},
        "cf": {"prompt": "The teacher assaulted the student. The attacker was the", "expected_label": "teacher"},
        "choices": {"teacher": [" teacher"], "student": [" student"]},
        "intervention_type": "substitution_verb_synonym",
        "contrast_labels": ["teacher", "student"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-role-verb-1",
        "base": {"prompt": "The chef attacked the critic. The attacker was the", "expected_label": "chef"},
        "cf": {"prompt": "The chef assaulted the critic. The attacker was the", "expected_label": "chef"},
        "choices": {"chef": [" chef"], "critic": [" critic"]},
        "intervention_type": "substitution_verb_synonym",
        "contrast_labels": ["chef", "critic"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-role-verb-2",
        "base": {"prompt": "The guard attacked the visitor. The attacker was the", "expected_label": "guard"},
        "cf": {"prompt": "The guard assaulted the visitor. The attacker was the", "expected_label": "guard"},
        "choices": {"guard": [" guard"], "visitor": [" visitor"]},
        "intervention_type": "substitution_verb_synonym",
        "contrast_labels": ["guard", "visitor"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-role-verb-3",
        "base": {"prompt": "The pilot attacked the passenger. The attacker was the", "expected_label": "pilot"},
        "cf": {"prompt": "The pilot assaulted the passenger. The attacker was the", "expected_label": "pilot"},
        "choices": {"pilot": [" pilot"], "passenger": [" passenger"]},
        "intervention_type": "substitution_verb_synonym",
        "contrast_labels": ["pilot", "passenger"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },

    # === Quantifier analogs: adjective synonym ===
    {
        "item_id": "sub-inv-quant-adj-0",
        "base": {"prompt": "All the files are complete. Therefore at least one file is", "expected_label": "complete"},
        "cf": {"prompt": "All the files are finished. Therefore at least one file is", "expected_label": "complete"},
        "choices": {"complete": [" complete"], "missing": [" missing"]},
        "intervention_type": "substitution_adj_synonym",
        "contrast_labels": ["complete", "missing"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-quant-adj-1",
        "base": {"prompt": "All the phones are charged. Therefore at least one phone is", "expected_label": "charged"},
        "cf": {"prompt": "All the phones are powered. Therefore at least one phone is", "expected_label": "charged"},
        "choices": {"charged": [" charged"], "dead": [" dead"]},
        "intervention_type": "substitution_adj_synonym",
        "contrast_labels": ["charged", "dead"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-quant-adj-2",
        "base": {"prompt": "All the orders are approved. Therefore at least one order is", "expected_label": "approved"},
        "cf": {"prompt": "All the orders are accepted. Therefore at least one order is", "expected_label": "approved"},
        "choices": {"approved": [" approved"], "rejected": [" rejected"]},
        "intervention_type": "substitution_adj_synonym",
        "contrast_labels": ["approved", "rejected"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-quant-adj-3",
        "base": {"prompt": "All the servers are online. Therefore at least one server is", "expected_label": "online"},
        "cf": {"prompt": "All the servers are active. Therefore at least one server is", "expected_label": "online"},
        "choices": {"online": [" online"], "offline": [" offline"]},
        "intervention_type": "substitution_adj_synonym",
        "contrast_labels": ["online", "offline"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-quant-adj-4",
        "base": {"prompt": "All the tables are clean. Therefore at least one table is", "expected_label": "clean"},
        "cf": {"prompt": "All the tables are tidy. Therefore at least one table is", "expected_label": "clean"},
        "choices": {"clean": [" clean"], "dirty": [" dirty"]},
        "intervention_type": "substitution_adj_synonym",
        "contrast_labels": ["clean", "dirty"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },

    # === Negation analogs: verb synonym ===
    {
        "item_id": "sub-inv-neg-verb-0",
        "base": {"prompt": "John did call the client. Therefore John", "expected_label": "did"},
        "cf": {"prompt": "John did ring the client. Therefore John", "expected_label": "did"},
        "choices": {"did": [" did"], "did_not": [" did not"]},
        "intervention_type": "substitution_verb_synonym",
        "contrast_labels": ["did", "did_not"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-neg-verb-1",
        "base": {"prompt": "Nina did pay the bill. Therefore Nina", "expected_label": "did"},
        "cf": {"prompt": "Nina did cover the bill. Therefore Nina", "expected_label": "did"},
        "choices": {"did": [" did"], "did_not": [" did not"]},
        "intervention_type": "substitution_verb_synonym",
        "contrast_labels": ["did", "did_not"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-neg-verb-2",
        "base": {"prompt": "Liam did start the engine. Therefore Liam", "expected_label": "did"},
        "cf": {"prompt": "Liam did start the motor. Therefore Liam", "expected_label": "did"},
        "choices": {"did": [" did"], "did_not": [" did not"]},
        "intervention_type": "substitution_verb_synonym",
        "contrast_labels": ["did", "did_not"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-neg-verb-3",
        "base": {"prompt": "Mia did write the code. Therefore Mia", "expected_label": "did"},
        "cf": {"prompt": "Mia did write the script. Therefore Mia", "expected_label": "did"},
        "choices": {"did": [" did"], "did_not": [" did not"]},
        "intervention_type": "substitution_verb_synonym",
        "contrast_labels": ["did", "did_not"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },

    # === Name substitution (all types) ===
    {
        "item_id": "sub-inv-name-0",
        "base": {"prompt": "John did call the office. Therefore this person", "expected_label": "did"},
        "cf": {"prompt": "Mark did call the office. Therefore this person", "expected_label": "did"},
        "choices": {"did": [" did"], "did_not": [" did not"]},
        "intervention_type": "substitution_name",
        "contrast_labels": ["did", "did_not"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-name-1",
        "base": {"prompt": "Kate did pay the bill. Therefore this person", "expected_label": "did"},
        "cf": {"prompt": "Jane did pay the bill. Therefore this person", "expected_label": "did"},
        "choices": {"did": [" did"], "did_not": [" did not"]},
        "intervention_type": "substitution_name",
        "contrast_labels": ["did", "did_not"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-name-2",
        "base": {"prompt": "Jack did start the engine. Therefore this person", "expected_label": "did"},
        "cf": {"prompt": "Paul did start the engine. Therefore this person", "expected_label": "did"},
        "choices": {"did": [" did"], "did_not": [" did not"]},
        "intervention_type": "substitution_name",
        "contrast_labels": ["did", "did_not"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-name-3",
        "base": {"prompt": "Luke did write the code. Therefore this person", "expected_label": "did"},
        "cf": {"prompt": "Alex did write the code. Therefore this person", "expected_label": "did"},
        "choices": {"did": [" did"], "did_not": [" did not"]},
        "intervention_type": "substitution_name",
        "contrast_labels": ["did", "did_not"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },

    # === Object noun synonym ===
    {
        "item_id": "sub-inv-obj-0",
        "base": {"prompt": "All the reports are finished. Therefore at least one of them is", "expected_label": "finished"},
        "cf": {"prompt": "All the papers are finished. Therefore at least one of them is", "expected_label": "finished"},
        "choices": {"finished": [" finished"], "unfinished": [" unfinished"]},
        "intervention_type": "substitution_noun_synonym",
        "contrast_labels": ["finished", "unfinished"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-obj-1",
        "base": {"prompt": "All the keys are available. Therefore at least one of them is", "expected_label": "available"},
        "cf": {"prompt": "All the cards are available. Therefore at least one of them is", "expected_label": "available"},
        "choices": {"available": [" available"], "lost": [" lost"]},
        "intervention_type": "substitution_noun_synonym",
        "contrast_labels": ["available", "lost"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
    {
        "item_id": "sub-inv-obj-2",
        "base": {"prompt": "All the seats are free. Therefore at least one of them is", "expected_label": "free"},
        "cf": {"prompt": "All the rooms are free. Therefore at least one of them is", "expected_label": "free"},
        "choices": {"free": [" free"], "taken": [" taken"]},
        "intervention_type": "substitution_noun_synonym",
        "contrast_labels": ["free", "taken"],
        "expected_effect": "invariant",
        "metadata": {"source": "b1a_design", "control": "substitution_invariant"},
    },
]


def validate_tokenization(items: List[Dict], tokenizer_names: List[str]) -> bool:
    """Validate that all items have non-empty spans across all tokenizers."""
    from transformers import AutoTokenizer

    all_pass = True
    for tname in tokenizer_names:
        print(f"\n{'='*60}")
        print(f"Tokenizer: {tname}")
        print(f"{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(tname)

        for item in items:
            iid = item["item_id"]
            base_text = item["base"]["prompt"]
            cf_text = item["cf"]["prompt"]

            base_ids = tokenizer(base_text, add_special_tokens=False)["input_ids"]
            cf_ids = tokenizer(cf_text, add_special_tokens=False)["input_ids"]

            span = divergence_span(base_ids, cf_ids)
            if span is None:
                print(f"  FAIL  {iid}: no divergence (identical tokenization)")
                all_pass = False
                continue

            base_span_len = span.a_len
            cf_span_len = span.b_len
            len_match = "MATCH" if base_span_len == cf_span_len else f"MISMATCH ({base_span_len} vs {cf_span_len})"

            if base_span_len == 0:
                print(f"  FAIL  {iid}: empty base span (insertion-type)")
                all_pass = False
            elif cf_span_len == 0:
                print(f"  FAIL  {iid}: empty CF span (deletion-type)")
                all_pass = False
            else:
                status = "OK" if base_span_len == cf_span_len else "WARN"
                # Show the actual tokens in the span
                base_span_tokens = tokenizer.convert_ids_to_tokens(base_ids[span.a_start:span.a_end])
                cf_span_tokens = tokenizer.convert_ids_to_tokens(cf_ids[span.b_start:span.b_end])
                print(f"  {status:5s} {iid:30s} base_span={base_span_tokens} cf_span={cf_span_tokens} len={len_match}")

    return all_pass


def write_jsonl(items: List[Dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {path}")


def main():
    parser = argparse.ArgumentParser(description="B1a: substitution-type invariant CF items")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--tokenizers", nargs="+",
                        default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    args = parser.parse_args()

    print(f"Designed {len(ITEMS)} substitution-type invariant items")
    print(f"Types: {sorted(set(it['intervention_type'] for it in ITEMS))}")

    ok = validate_tokenization(ITEMS, args.tokenizers)

    if args.output and not args.validate_only:
        write_jsonl(ITEMS, args.output)

    if ok:
        print("\nAll items PASS tokenization validation.")
    else:
        print("\nSome items FAIL tokenization validation. Review above.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
