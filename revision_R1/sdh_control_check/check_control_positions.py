"""Classify the SDH control positions chosen by the archived protocol (tokenizer only, no model).

Uses the repository's own selection function with the archived settings
(buffer=2, donor_buffer=2, position_window=8, selection_seed=0).
For each of the 104 donor->receiver directions it reports whether the
control patch is a no-op (donor and receiver are identical up to and
including the control position, at the same index) and whether the
control position is the final token of the receiver prompt.

Run from the repository root:
    python revision_R1/sdh_control_check/check_control_positions.py <tokenizer name or local snapshot path>
"""
import collections
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aom.data.loaders import load_disamb_pairs  # noqa: E402
from aom.metrics.disamb import (  # noqa: E402
    _encode_prompt,
    _select_offtarget_pairs_with_fallback,
    _stable_u32_seed,
)
from aom.token_spans import token_span_for_substring  # noqa: E402

name = sys.argv[1]
tok = AutoTokenizer.from_pretrained(name)
items = load_disamb_pairs(str(ROOT / "data_paper_hardened_v2" / "disamb_pairs.jsonl"))
dev = torch.device("cpu")
cnt = collections.Counter()
finals = []
for it in items:
    for di, (donor, recv) in enumerate(((it.a, it.b), (it.b, it.a))):
        tag = "a_to_b" if di == 0 else "b_to_a"
        dspan, dt = token_span_for_substring(tok, donor.prompt, it.target, it.target_occurrence)
        rspan, rt = token_span_for_substring(tok, recv.prompt, it.target, it.target_occurrence)
        if len(dspan) != len(rspan) or dt != rt:
            cnt["skipped"] += 1
            continue
        d = [int(x) for x in _encode_prompt(tok, donor.prompt, dev)[0].tolist()]
        r = [int(x) for x in _encode_prompt(tok, recv.prompt, dev)[0].tolist()]
        rng = random.Random(int(_stable_u32_seed(it.pair_id, tag, base_seed=0)))
        pairs, strat = _select_offtarget_pairs_with_fallback(
            donor_ids=d, recv_ids=r, donor_span=dspan, recv_span=rspan,
            k=len(rspan), buffer=2, donor_buffer=2, position_window=8, rng=rng,
        )
        cnt["directions"] += 1
        cnt[strat] += 1
        cnt["noop_identical_prefix"] += all(q == p and d[: q + 1] == r[: p + 1] for q, p in pairs)
        final = any(p == len(r) - 1 for q, p in pairs)
        cnt["ctrl_is_final_token"] += final
        cnt["ctrl_before_target"] += all(p < min(rspan) for q, p in pairs)
        cnt["target_index_differs"] += dspan != rspan
        if final:
            finals.append(f"{it.pair_id} {tag}")
print(dict(cnt))
print("final-token controls:", finals)
