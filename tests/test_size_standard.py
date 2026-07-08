from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from aom.data.loaders import load_counterfactual_pairs_with_manifest
from aom.interventions.patching.size_standard_protocol import (
    SizeStandardNamedSpanProtocol,
    SizeStandardPatchingConfig,
    SpanMarker,
    token_span_for_marker,
)
from scripts.generate_size_standard import (
    SpanMarker as GeneratorSpanMarker,
    _find_anchored_char_span,
    _single_edit_assertion,
    _single_edit_error,
    _load_model_priors_cm,
    generate_size_standard_rows,
    token_span_for_marker as generator_token_span_for_marker,
)
from aom_size_standard_patching import (
    _filter_entry_for_model,
    _resolve_size_dataset_for_model,
    _slug as runner_slug,
)
from scripts.check_size_standard_acceptance import (
    _argmax_label,
    _binary_label_metrics,
    _corrected_scores,
    _determinacy_bias_from_scored_contexts,
    _parse_min_kept_by_family,
    _is_soft_penumbra_side,
    _slug as acceptance_slug,
    filtered_jsonl_path,
)
from scripts.analyze_size_standard_readouts import (
    summarize_conflict_winners,
    summarize_determinacy,
    summarize_trace,
    summarize_uncertainty,
)


class FakeFastTokenizer:
    is_fast = True

    def __init__(self, *, digit_tokens: bool = False):
        self.digit_tokens = bool(digit_tokens)
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab) + 1
            self.vocab[token] = idx
            self.id_to_token[idx] = token
        return self.vocab[token]

    def _scan(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit():
                if self.digit_tokens:
                    ids.append(self._id(ch))
                    offsets.append((i, i + 1))
                    i += 1
                else:
                    j = i + 1
                    while j < len(text) and text[j].isdigit():
                        j += 1
                    ids.append(self._id(text[i:j]))
                    offsets.append((i, j))
                    i = j
                continue
            if ch.isalpha():
                j = i + 1
                while j < len(text) and text[j].isalpha():
                    j += 1
                ids.append(self._id(text[i:j]))
                offsets.append((i, j))
                i = j
                continue
            ids.append(self._id(ch))
            offsets.append((i, i + 1))
            i += 1
        return ids, offsets

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        ids, offsets = self._scan(str(text))
        out: dict[str, torch.Tensor] = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        return out


class FixedOffsetTokenizer:
    is_fast = True

    def __init__(self, offsets: list[tuple[int, int]]):
        self.offsets = list(offsets)

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        ids = list(range(1, len(self.offsets) + 1))
        out: dict[str, torch.Tensor] = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor([self.offsets], dtype=torch.long)
        return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def test_anchored_span_does_not_match_number_inside_larger_number() -> None:
    tok = FakeFastTokenizer()
    text = "The measured object is 350 cm. The reference is 50 cm. Therefore it is"
    marker = GeneratorSpanMarker(text="50", left="The reference is ", right=" cm")
    span, ids = generator_token_span_for_marker(tok, text, marker)
    assert span
    assert [tok.id_to_token[int(i)] for i in ids] == ["50"]


def test_anchor_must_be_unique() -> None:
    text = "The reference is 50 cm. The reference is 50 cm."
    marker = GeneratorSpanMarker(text="50", left="The reference is ", right=" cm")
    try:
        _find_anchored_char_span(text, marker)
    except ValueError as e:
        assert "Anchor not unique" in str(e)
    else:
        raise AssertionError("Expected duplicate anchor to fail")


def test_numeric_span_does_not_bleed_into_unit_with_digit_tokenizer() -> None:
    tok = FakeFastTokenizer(digit_tokens=True)
    text = "The reference is 50 cm. Therefore it is"
    marker = SpanMarker(text="50", left="The reference is ", right=" cm")
    span, ids = token_span_for_marker(tok, text, marker)
    assert len(span) == 2
    assert [tok.id_to_token[int(i)] for i in ids] == ["5", "0"]


def test_leading_whitespace_overlap_allowed_but_unit_bleed_rejected() -> None:
    text = "is 50 cm"
    marker = SpanMarker(text="50", left="is ", right=" cm")
    span, _ids = token_span_for_marker(FixedOffsetTokenizer([(0, 2), (2, 5), (5, 8)]), text, marker)
    assert span == [1]

    try:
        token_span_for_marker(FixedOffsetTokenizer([(0, 2), (3, 6), (6, 8)]), text, marker)
    except ValueError as e:
        assert "named_span_token_bleed" in str(e)
    else:
        raise AssertionError("Expected numeric token overlap into following space/unit to fail")


def test_generated_rows_load_and_protocol_builds_without_span_mismatch(tmp_path: Path) -> None:
    tokenizers = {
        "whole": FakeFastTokenizer(digit_tokens=False),
        "digit": FakeFastTokenizer(digit_tokens=True),
    }
    rows, summary = generate_size_standard_rows(tokenizers=tokenizers)
    assert rows
    assert summary["skip_reason_counts"] == {}
    assert summary["generation_error_counts"] == {}

    jsonl_path = tmp_path / "size_standard.jsonl"
    _write_jsonl(jsonl_path, rows)
    items, manifest = load_counterfactual_pairs_with_manifest(jsonl_path, error_policy="raise")
    assert manifest.n_rows_invalid == 0
    assert len(items) == len(rows)
    assert all(set(r["base"].keys()) == {"prompt", "expected_label"} for r in rows)
    assert all(set(r["cf"].keys()) == {"prompt", "expected_label"} for r in rows)

    protocol = SizeStandardNamedSpanProtocol()
    for tok in tokenizers.values():
        cases, skips = protocol.build_cases(tokenizer=tok, items=items, device=torch.device("cpu"))
        assert not skips
        assert len(cases) == len(items)
        assert all(len(c.receiver_span) == len(c.donor_span) for c in cases)
        assert {c.strata["patch_span"] for c in cases} == {"class_span", "standard_span"}
        assert "penumbra_standard" in {c.strata["family"] for c in cases}
        assert "class_penumbra" in {c.strata["family"] for c in cases}


def test_protocol_can_add_value_span_placebo_cases(tmp_path: Path) -> None:
    tokenizers = {
        "whole": FakeFastTokenizer(digit_tokens=False),
        "digit": FakeFastTokenizer(digit_tokens=True),
    }
    rows, _summary = generate_size_standard_rows(tokenizers=tokenizers)
    standard_rows = [
        r
        for r in rows
        if r["metadata"]["family"] in {"standard_swap", "penumbra_standard", "class_penumbra"}
    ]
    primary_standard_span_rows = [r for r in rows if r["metadata"]["patch_span"] == "standard_span"]
    jsonl_path = tmp_path / "size_standard.jsonl"
    _write_jsonl(jsonl_path, rows)
    items, _manifest = load_counterfactual_pairs_with_manifest(jsonl_path, error_policy="raise")

    protocol = SizeStandardNamedSpanProtocol(
        config=SizeStandardPatchingConfig(include_value_placebo=True)
    )
    for tok in tokenizers.values():
        cases, skips = protocol.build_cases(tokenizer=tok, items=items, device=torch.device("cpu"))
        assert not skips
        assert len(cases) == len(items) + len(standard_rows)
        patch_span_counts = {span: sum(c.strata["patch_span"] == span for c in cases) for span in {"class_span", "standard_span", "value_span"}}
        assert patch_span_counts["value_span"] == len(standard_rows)
        assert patch_span_counts["standard_span"] == len(primary_standard_span_rows)
        assert all(c.strata.get("case_type") == "placebo" for c in cases if c.strata["patch_span"] == "value_span")
        assert all(c.case_id.endswith("__value_span_placebo") for c in cases if c.strata["patch_span"] == "value_span")


def test_class_swap_rows_are_number_free_and_standard_swap_rows_keep_standard_metadata() -> None:
    rows, _summary = generate_size_standard_rows(tokenizers={"whole": FakeFastTokenizer(digit_tokens=False)})
    class_rows = [r for r in rows if r["metadata"]["family"] == "class_swap"]
    standard_rows = [r for r in rows if r["metadata"]["family"] == "standard_swap"]
    assert class_rows
    assert standard_rows

    for row in class_rows:
        md = row["metadata"]
        prompt_blob = f"{row['base']['prompt']} {row['cf']['prompt']}".lower()
        assert "standard" not in prompt_blob
        assert "reference" not in prompt_blob
        assert "trial" not in prompt_blob
        assert md["label_source"] == "class_prior"
        assert md["margin_bin"] == "prior_only"
        assert "standard_cm" not in md
        assert "cf_standard_cm" not in md
        assert "log_ratio" not in md
        assert "cf_log_ratio" not in md
        assert "stated_standard_label" not in md
        assert "cf_stated_standard_label" not in md
        assert "standard_span" not in md["span_markers"]
        assert "For a elephant" not in row["base"]["prompt"]
        assert "For a ant" not in row["base"]["prompt"]
        assert "For a elephant" not in row["cf"]["prompt"]
        assert "For a ant" not in row["cf"]["prompt"]

    for row in standard_rows:
        md = row["metadata"]
        assert md["label_source"] == "stated_standard"
        assert "standard_cm" in md
        assert "cf_standard_cm" in md
        assert "log_ratio" in md
        assert "standard_span" in md["span_markers"]


def test_penumbra_standard_rows_are_dense_soft_gate_candidates() -> None:
    rows, summary = generate_size_standard_rows(tokenizers={"whole": FakeFastTokenizer(digit_tokens=False)})
    pen_rows = [r for r in rows if r["metadata"]["family"] == "penumbra_standard"]
    assert pen_rows
    assert summary["counts_by_family"]["penumbra_standard"] == len(pen_rows)
    assert {r["metadata"]["patch_span"] for r in pen_rows} == {"standard_span"}
    assert {"small_to_large", "large_to_small"} <= {r["metadata"]["polarity"] for r in pen_rows}

    near_rows = [r for r in pen_rows if float(r["metadata"]["abs_log_ratio"]) <= 0.15]
    assert near_rows
    for row in near_rows[:10]:
        md = row["metadata"]
        assert _is_soft_penumbra_side(md, "base", threshold=0.15)
        assert _is_soft_penumbra_side(md, "cf", threshold=0.15)
        assert md["penumbra_axis"] == "stated_standard"
        assert md["penumbral_relation"] == "standard_straddle"
        assert md["value_pair_id"]


def test_class_penumbra_rows_are_near_prior_and_soft_gated() -> None:
    priors = {
        "ant": 0.5,
        "mouse": 8.0,
        "worm": 10.0,
        "frog": 12.0,
        "fish": 30.0,
        "cat": 45.0,
        "dog": 80.0,
        "human": 170.0,
        "horse": 240.0,
        "elephant": 300.0,
        "car": 450.0,
        "plane": 4000.0,
        "ship": 30000.0,
    }
    rows, summary = generate_size_standard_rows(
        tokenizers={"whole": FakeFastTokenizer(digit_tokens=False)},
        model_priors_cm=priors,
        require_model_priors=True,
    )
    pen_rows = [r for r in rows if r["metadata"]["family"] == "class_penumbra"]
    assert pen_rows
    assert summary["counts_by_family"]["class_penumbra"] == len(pen_rows)
    assert {r["metadata"]["patch_span"] for r in pen_rows} == {"class_span"}

    value_texts = {str(r["metadata"]["value_text"]) for r in pen_rows}
    assert any("." in v for v in value_texts)
    assert any(float(r["metadata"]["value_cm"]) < 1.0 for r in pen_rows)

    near_rows = [r for r in pen_rows if float(r["metadata"]["target_abs_log_ratio"]) <= 0.15]
    assert near_rows
    for row in near_rows[:20]:
        md = row["metadata"]
        assert _is_soft_penumbra_side(md, "base", threshold=0.15)
        assert not _is_soft_penumbra_side(md, "cf", threshold=0.15)
        assert md["penumbra_axis"] == "class_prior"
        assert md["penumbral_relation"] == "class_prior_neighborhood"
        assert md["label_source"] == "class_prior_penumbra"
        assert "standard" not in row["base"]["prompt"].lower()
        assert "reference" not in row["base"]["prompt"].lower()
        assert "value_span" in md["span_markers"]
        assert float(md["abs_log_ratio_to_prior"]) <= 0.2
        assert float(md["cf_abs_log_ratio_to_prior"]) >= 0.35


def test_model_prior_loader_and_required_prior_mode(tmp_path: Path) -> None:
    priors_path = tmp_path / "priors.json"
    priors_path.write_text(
        json.dumps(
            {
                "model_priors": {
                    "ant": {"model_prior_cm_argmax": 1, "model_prior_cm_expected_log": 0.8, "prior_valid": True},
                    "human": {"model_prior_cm_argmax": 180, "model_prior_cm_expected_log": 170, "prior_valid": False},
                    "elephant": {"model_prior_cm_argmax": 400, "model_prior_cm_expected_log": 350, "prior_valid": True},
                }
            }
        ),
        encoding="utf-8",
    )
    priors = _load_model_priors_cm(priors_path)
    assert priors == {"ant": 1.0, "human": 180.0, "elephant": 400.0}
    priors_expected = _load_model_priors_cm(priors_path, prior_field="expected_log")
    assert priors_expected == {"ant": 0.8, "human": 170.0, "elephant": 350.0}
    priors_valid = _load_model_priors_cm(priors_path, prior_field="argmax", valid_only=True)
    assert priors_valid == {"ant": 1.0, "elephant": 400.0}

    with pytest.raises(ValueError, match="requires finite positive priors"):
        generate_size_standard_rows(
            tokenizers={"whole": FakeFastTokenizer(digit_tokens=False)},
            model_priors_cm={"ant": 1.0},
            require_model_priors=True,
        )


def test_conflict_rows_have_sibling_gate_and_two_donor_metadata() -> None:
    rows, summary = generate_size_standard_rows(tokenizers={"whole": FakeFastTokenizer(digit_tokens=False)})
    conflict_rows = [r for r in rows if r["metadata"]["family"] == "conflict_swap"]
    assert conflict_rows
    assert summary["counts_by_family"]["conflict_swap"] == len(conflict_rows)

    by_receiver: dict[str, set[str]] = {}
    for row in conflict_rows:
        md = row["metadata"]
        assert md["label_source"] == "cue_conflict"
        assert md["patch_span"] in {"class_span", "standard_span"}
        assert md["donor_kind"] in {"class_donor", "standard_donor"}
        assert md["receiver_group_id"]
        assert set(md["conflict_gate_siblings"]) == {"class", "standard"}
        for sib in md["conflict_gate_siblings"].values():
            assert set(sib) == {"prompt", "expected_label"}
            assert sib["expected_label"] in {"small", "large"}
        by_receiver.setdefault(md["receiver_group_id"], set()).add(md["donor_kind"])

    assert by_receiver
    assert all(kinds == {"class_donor", "standard_donor"} for kinds in by_receiver.values())


def test_single_edit_error_can_be_used_as_skip_reason() -> None:
    reason = _single_edit_error("value 150 cm", "value 250 cm", "50", "60")
    assert reason is not None
    try:
        _single_edit_assertion("value 150 cm", "value 250 cm", "50", "60")
    except ValueError as e:
        assert str(e) == reason
    else:
        raise AssertionError("Expected single-edit assertion to raise")


def test_bias_correction_changes_gate_argmax_when_label_prior_dominates() -> None:
    raw_scores = {"small": -1.0, "large": -2.0}
    label_bias = {"small": -0.7, "large": -2.2}
    corrected = _corrected_scores(raw_scores, label_bias)
    assert _argmax_label(raw_scores) == "small"
    assert corrected["small"] == pytest.approx(-0.3)
    assert corrected["large"] == pytest.approx(0.2)
    assert _argmax_label(corrected) == "large"


def test_determinacy_bias_uses_determinate_non_content_contexts() -> None:
    scored_contexts = [
        {
            "expected_label": "def_small",
            "scores": {"def_small": 10.0, "borderline": 1.0, "def_large": 2.0},
        },
        {
            "expected_label": "def_large",
            "scores": {"def_small": 3.0, "borderline": 5.0, "def_large": 20.0},
        },
    ]
    choices = {
        "def_small": [" definitely small"],
        "borderline": [" borderline"],
        "def_large": [" definitely large"],
    }

    bias = _determinacy_bias_from_scored_contexts(scored_contexts, choices)

    assert bias["borderline"] == pytest.approx(3.0)
    assert bias["def_small"] == pytest.approx(3.0)
    assert bias["def_large"] == pytest.approx(2.0)


def test_per_model_filtered_path_slugging(tmp_path: Path) -> None:
    model_slug = acceptance_slug("Qwen/Qwen2.5-3B")
    assert model_slug == "qwen_qwen2_5_3b"
    assert runner_slug("Qwen/Qwen2.5-3B") == model_slug

    default_path = filtered_jsonl_path(out_dir=tmp_path, base_path="", model_slug=model_slug)
    assert default_path == tmp_path / "size_standard.filtered.qwen_qwen2_5_3b.jsonl"

    custom_path = filtered_jsonl_path(
        out_dir=tmp_path,
        base_path=str(tmp_path / "filtered.jsonl"),
        model_slug=model_slug,
    )
    assert custom_path == tmp_path / "filtered.qwen_qwen2_5_3b.jsonl"


def test_min_kept_by_family_parser() -> None:
    assert _parse_min_kept_by_family(["standard_swap:25", "class_swap:10"]) == {
        "standard_swap": 25,
        "class_swap": 10,
    }
    with pytest.raises(ValueError):
        _parse_min_kept_by_family(["bad"])
    with pytest.raises(ValueError):
        _parse_min_kept_by_family(["class_swap:-1"])


def test_filter_manifest_resolves_matching_model_entry(tmp_path: Path) -> None:
    manifest = {
        "models_by_name": {
            "gpt2": {
                "model": "gpt2",
                "model_slug": "gpt2",
                "filtered_jsonl": "size_standard.filtered.gpt2.jsonl",
            }
        },
        "entries": [],
    }
    entry = _filter_entry_for_model(manifest, "gpt2")
    assert entry is not None
    assert entry["filtered_jsonl"] == "size_standard.filtered.gpt2.jsonl"
    assert _filter_entry_for_model(manifest, "Qwen/Qwen2.5-3B") is None

    selected, selected_entry = _resolve_size_dataset_for_model(
        model_name="gpt2",
        n_models=2,
        size_path="/unused/shared.jsonl",
        filter_manifest=manifest,
        allow_shared_size_path=False,
        filter_manifest_path=tmp_path / "size_standard.filter_manifest.json",
    )
    assert selected == tmp_path / "size_standard.filtered.gpt2.jsonl"
    assert selected_entry == entry


def test_filter_manifest_does_not_double_prefix_existing_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data_size_standard").mkdir()
    expected = tmp_path / "data_size_standard" / "size_standard.filtered.qwen_qwen2_5_3b.jsonl"
    expected.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    manifest = {
        "models_by_name": {
            "Qwen/Qwen2.5-3B": {
                "model": "Qwen/Qwen2.5-3B",
                "model_slug": "qwen_qwen2_5_3b",
                "filtered_jsonl": "data_size_standard/size_standard.filtered.qwen_qwen2_5_3b.jsonl",
            }
        },
        "entries": [],
    }
    selected, _entry = _resolve_size_dataset_for_model(
        model_name="Qwen/Qwen2.5-3B",
        n_models=1,
        size_path="/unused/shared.jsonl",
        filter_manifest=manifest,
        allow_shared_size_path=False,
        filter_manifest_path=Path("data_size_standard/size_standard.filter_manifest.json"),
    )
    assert selected == Path("data_size_standard/size_standard.filtered.qwen_qwen2_5_3b.jsonl")


def test_multi_model_runner_requires_filter_manifest_or_explicit_shared_path() -> None:
    with pytest.raises(ValueError, match="Multi-model size patching requires"):
        _resolve_size_dataset_for_model(
            model_name="gpt2",
            n_models=2,
            size_path="/tmp/shared.jsonl",
            filter_manifest=None,
            allow_shared_size_path=False,
        )

    selected, entry = _resolve_size_dataset_for_model(
        model_name="gpt2",
        n_models=2,
        size_path="/tmp/shared.jsonl",
        filter_manifest=None,
        allow_shared_size_path=True,
    )
    assert selected == Path("/tmp/shared.jsonl")
    assert entry is None


def test_readout_analysis_counts_conflicts_receiver_level_from_base_rows() -> None:
    rows = [
        {
            "item_id": "case-1",
            "kept": "True",
            "family": "conflict_swap",
            "side": "base",
            "conflict_receiver_id": "r1",
            "conflict_direction": "class_small_standard_large",
            "pred_label": "small",
            "gate_pred_label": "large",
        },
        {
            "item_id": "case-1",
            "kept": "True",
            "family": "conflict_swap",
            "side": "sibling_class",
            "conflict_receiver_id": "r1",
            "conflict_direction": "class_small_standard_large",
            "pred_label": "small",
            "gate_pred_label": "small",
        },
        {
            "item_id": "case-2",
            "kept": "True",
            "family": "conflict_swap",
            "side": "base",
            "conflict_receiver_id": "r1",
            "conflict_direction": "class_small_standard_large",
            "pred_label": "small",
            "gate_pred_label": "large",
        },
        {
            "item_id": "case-3",
            "kept": "True",
            "family": "conflict_swap",
            "side": "base",
            "conflict_receiver_id": "r2",
            "conflict_direction": "standard_small_class_large",
            "pred_label": "small",
            "gate_pred_label": "small",
        },
    ]

    summary = summarize_conflict_winners(rows)

    assert summary["n_patch_cases"] == 3
    assert summary["n_receivers"] == 2
    assert summary["receiver_counts_corrected"] == {"standard": 2}
    assert summary["receiver_counts_raw"] == {"class": 1, "standard": 1}
    assert summary["patch_case_counts_corrected_diagnostic"] == {"standard": 3}


def test_readout_analysis_rejects_inconsistent_duplicate_receiver_predictions() -> None:
    rows = [
        {
            "kept": "True",
            "family": "conflict_swap",
            "side": "base",
            "conflict_receiver_id": "r1",
            "conflict_direction": "class_small_standard_large",
            "gate_pred_label": "small",
        },
        {
            "kept": "True",
            "family": "conflict_swap",
            "side": "base",
            "conflict_receiver_id": "r1",
            "conflict_direction": "class_small_standard_large",
            "gate_pred_label": "large",
        },
    ]
    with pytest.raises(ValueError, match="Conflicting 'gate_pred_label'"):
        summarize_conflict_winners(rows)


def test_readout_analysis_prefers_corrected_determinacy() -> None:
    rows = [
        {
            "kept": "True",
            "family": "class_penumbra",
            "class_name": "cup",
            "template_id": "1",
            "abs_log_ratio_to_prior": "0.1",
            "determinacy_pred_label": "borderline",
            "determinacy_pred_label_corrected": "def_small",
            "determinacy_borderline_margin": "0.2",
            "determinacy_borderline_margin_corrected": "-0.3",
        },
        {
            "kept": "False",
            "family": "class_penumbra",
            "determinacy_pred_label": "def_large",
            "determinacy_pred_label_corrected": "def_large",
        },
    ]

    summary = summarize_determinacy(rows)

    assert summary["corrected_column"] == "determinacy_pred_label_corrected"
    assert summary["counts_raw"] == {"borderline": 1}
    assert summary["counts_corrected"] == {"def_small": 1}
    assert summary["mean_borderline_margin_corrected"] == pytest.approx(-0.3)


def test_binary_determinacy_metrics_use_margin_magnitude_and_entropy() -> None:
    raw = {"def_small": -2.0, "def_large": -2.5, "borderline": -0.1}
    corrected = _corrected_scores(raw, {"def_small": -1.0, "def_large": -2.0, "borderline": 0.0})

    raw_metrics = _binary_label_metrics(raw)
    corrected_metrics = _binary_label_metrics(corrected)

    assert raw_metrics["pred_label"] == "def_small"
    assert raw_metrics["margin"] == pytest.approx(-0.5)
    assert raw_metrics["abs_margin"] == pytest.approx(0.5)
    assert 0.0 < float(raw_metrics["entropy_norm"]) < 1.0

    assert corrected_metrics["pred_label"] == "def_large"
    assert corrected_metrics["margin"] == pytest.approx(0.5)
    assert corrected_metrics["abs_margin"] == pytest.approx(0.5)
    assert corrected_metrics["prob_def_large"] == pytest.approx(float(raw_metrics["prob_def_small"]))


def test_uncertainty_analysis_derives_two_way_scores_from_existing_determinacy_fields() -> None:
    rows = [
        {
            "kept": "True",
            "family": "class_penumbra",
            "side": "base",
            "class_name": "cup",
            "abs_log_ratio": "0.1",
            "abs_log_ratio_to_prior": "2.5",
            "det_score_def_small": "-2.0",
            "det_score_def_large": "-2.2",
            "det_score_def_small_corrected": "-2.0",
            "det_score_def_large_corrected": "-2.1",
        },
        {
            "kept": "True",
            "family": "class_penumbra",
            "side": "cf",
            "class_name": "cup",
            "abs_log_ratio": "2.5",
            "abs_log_ratio_to_prior": "0.1",
            "det_score_def_small": "-1.0",
            "det_score_def_large": "-4.0",
            "det_score_def_small_corrected": "-1.0",
            "det_score_def_large_corrected": "-4.0",
        },
        {
            "kept": "False",
            "family": "class_penumbra",
            "abs_log_ratio_to_prior": "0.0",
            "det_score_def_small": "-1.0",
            "det_score_def_large": "-1.0",
        },
    ]

    summary = summarize_uncertainty(rows)

    assert summary["n"] == 2
    assert summary["has_corrected"] is True
    near = next(
        row
        for row in summary["strata_rows"]
        if row["stratum"] == "abs_log_ratio_to_prior_bin" and row["value"] == "0-0.5"
    )
    far = next(
        row
        for row in summary["strata_rows"]
        if row["stratum"] == "abs_log_ratio_to_prior_bin" and row["value"] == ">=2"
    )
    assert near["mean_abs_margin_corrected"] == pytest.approx(0.1)
    assert far["mean_abs_margin_corrected"] == pytest.approx(3.0)
    assert float(near["mean_entropy_norm_corrected"]) > float(far["mean_entropy_norm_corrected"])
    family_side_values = {
        row["value"]
        for row in summary["strata_rows"]
        if row["stratum"] == "family_side"
    }
    assert family_side_values == {"class_penumbra::base", "class_penumbra::cf"}


def test_trace_analysis_aggregates_duplicate_patch_cases_receiver_level() -> None:
    rows = [
        {"conflict_receiver_id": "r1", "patch_span": "standard_span", "layer": "0", "effect": "1.0", "sham_effect": "0.1", "flip": "1.0"},
        {"conflict_receiver_id": "r1", "patch_span": "standard_span", "layer": "0", "effect": "3.0", "sham_effect": "0.3", "flip": "0.0"},
        {"conflict_receiver_id": "r1", "patch_span": "standard_span", "layer": "1", "effect": "4.0", "sham_effect": "0.4", "flip": "1.0"},
        {"conflict_receiver_id": "r1", "patch_span": "class_span", "layer": "0", "effect": "1.0", "sham_effect": "0.0", "flip": "0.0"},
        {"conflict_receiver_id": "r1", "patch_span": "class_span", "layer": "1", "effect": "2.0", "sham_effect": "0.0", "flip": "1.0"},
    ]

    summary = summarize_trace(rows)

    assert summary["n_trace_rows"] == 5
    assert summary["n_receiver_layer_span_cells"] == 4
    std_layer0 = next(
        row for row in summary["layer_rows"] if row["patch_span"] == "standard_span" and row["layer"] == "0"
    )
    assert std_layer0["mean_effect"] == pytest.approx(2.0)
    assert std_layer0["mean_sham_effect"] == pytest.approx(0.2)
    assert std_layer0["flip_rate"] == pytest.approx(0.5)
    assert summary["paired_standard_vs_class"]["mean_standard_minus_class_max_effect"] == pytest.approx(2.0)
