"""Score the two worked examples of the revised manuscript with GPT-2.

DISAMB pair bank-lex-0 (Section 1.1) and CF item role-swap-5 (Section 2).
Uses the repository's own label scorer (length-normalized, log-mean-exp).

Run from the repository root:
    python revision_R1/single_item_scores/score_examples.py
"""

import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aom.metrics.disamb import score_labels_next_continuations  # noqa: E402

REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATA = ROOT / "data_paper_hardened_v2"


def find(path, key, value):
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            if item[key] == value:
                return item
    raise KeyError(value)


def main():
    tok = AutoTokenizer.from_pretrained("gpt2", revision=REV)
    model = AutoModelForCausalLM.from_pretrained("gpt2", revision=REV, dtype=torch.float32).eval()
    device = torch.device("cpu")

    pair = find(DATA / "disamb_pairs.jsonl", "pair_id", "bank-lex-0")
    print("DISAMB bank-lex-0")
    with torch.no_grad():
        for side in ("a", "b"):
            s = score_labels_next_continuations(model, tok, pair[side]["prompt"], pair["choices"], device).by_label
            margin = s["finance"] - s["river"]
            print(f"  {side}: finance={s['finance']:.3f} river={s['river']:.3f} "
                  f"finance-river={margin:.3f} ratio={math.exp(abs(margin)):.1f}")

    item = find(DATA / "counterfactual.jsonl", "item_id", "role-swap-5")
    print("CF role-swap-5")
    prefs = {}
    with torch.no_grad():
        for side in ("base", "cf"):
            s = score_labels_next_continuations(model, tok, item[side]["prompt"], item["choices"], device).by_label
            prefs[side] = s["passenger"] - s["pilot"]
            print(f"  {side}: pilot={s['pilot']:.3f} passenger={s['passenger']:.3f} "
                  f"passenger-pilot={prefs[side]:.3f}")
    print(f"  shift toward passenger: {prefs['cf'] - prefs['base']:.3f}")


if __name__ == "__main__":
    main()
