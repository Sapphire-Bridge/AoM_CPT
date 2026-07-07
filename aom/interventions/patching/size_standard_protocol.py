from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from aom.data.schemas import CounterfactualPair

from .base import ActivationPatchingProtocol, CaseSkip, ComparisonSpec, PatchingCase


@dataclass(frozen=True)
class SizeStandardPatchingConfig:
    include_expected_effects: Tuple[str, ...] = ("shift",)
    require_readout: str = "bigness_logscore_margin"
    include_value_placebo: bool = False


@dataclass(frozen=True)
class SpanMarker:
    text: str
    left: str
    right: str


def _marker_from_obj(obj: Any) -> SpanMarker:
    if not isinstance(obj, Mapping):
        raise ValueError("span marker must be a mapping")
    return SpanMarker(text=str(obj["text"]), left=str(obj.get("left", "")), right=str(obj.get("right", "")))


def _find_anchored_char_span(text: str, marker: SpanMarker) -> tuple[int, int]:
    anchor = f"{marker.left}{marker.text}{marker.right}"
    start = str(text).find(anchor)
    if start < 0:
        raise ValueError(f"anchor_not_found:{anchor!r}")
    second = str(text).find(anchor, start + 1)
    if second >= 0:
        raise ValueError(f"anchor_not_unique:{anchor!r}")
    span_start = int(start + len(marker.left))
    span_end = int(span_start + len(marker.text))
    return span_start, span_end


def token_span_for_marker(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    marker: SpanMarker,
) -> tuple[list[int], list[int]]:
    char_start, char_end = _find_anchored_char_span(text, marker)
    enc = tokenizer(str(text), return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", None)
    if offsets is None:
        raise ValueError("Tokenizer must support return_offsets_mapping for named-span patching.")
    offsets_list = offsets[0].tolist()
    input_ids = enc["input_ids"][0].tolist()
    span = [i for i, (s, e) in enumerate(offsets_list) if (s < char_end and e > char_start)]
    if not span:
        raise ValueError("empty_named_span")
    if span != list(range(span[0], span[-1] + 1)):
        raise ValueError("non_contiguous_named_span")
    for i in span:
        s, e = offsets_list[i]
        left_extra = str(text)[int(s) : char_start] if int(s) < char_start else ""
        if (left_extra and left_extra.strip()) or int(e) > char_end:
            raise ValueError(
                f"named_span_token_bleed:text={marker.text!r}:token_offset={(int(s), int(e))}:char_span={(char_start, char_end)}"
            )
    return [int(i) for i in span], [int(input_ids[i]) for i in span]


def _stratum_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


class SizeStandardNamedSpanProtocol(ActivationPatchingProtocol):
    """
    Named-span donor->receiver patching for size/standard dissociation rows.

    Receiver is `base`, donor is `cf`, and positive effect means the receiver
    moved toward `cf.expected_label` under:
        effect = patched_margin(cf.expected_label) - base_margin(cf.expected_label)

    The shared runner already computes the receiver-from-receiver sham baseline
    for every case.
    """

    name = "size_standard_named_span"

    def __init__(self, *, config: SizeStandardPatchingConfig | None = None):
        self.config = SizeStandardPatchingConfig() if config is None else config

    def primary_comparisons(self) -> Sequence[ComparisonSpec]:
        comparisons: list[ComparisonSpec] = [
            ComparisonSpec(
                name="patch_span_standard_vs_class",
                stratum_key="patch_span",
                a_value="standard_span",
                b_value="class_span",
            ),
        ]
        if bool(self.config.include_value_placebo):
            comparisons.append(
                ComparisonSpec(
                    name="patch_span_standard_vs_value_placebo",
                    stratum_key="patch_span",
                    a_value="standard_span",
                    b_value="value_span",
                )
            )
        return tuple(comparisons)

    def build_cases(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        items: Sequence[Any],
        device: torch.device,
    ) -> tuple[list[PatchingCase], list[CaseSkip]]:
        cases: list[PatchingCase] = []
        skips: list[CaseSkip] = []

        for raw in items:
            if not isinstance(raw, CounterfactualPair):
                skips.append(CaseSkip(case_id=str(getattr(raw, "item_id", "size_item")), reason="type_mismatch"))
                continue
            it: CounterfactualPair = raw
            if str(it.expected_effect) not in set(self.config.include_expected_effects):
                skips.append(CaseSkip(case_id=str(it.item_id), reason=f"excluded_expected_effect:{it.expected_effect}"))
                continue
            md = it.metadata or {}
            if str(md.get("readout", "")) != str(self.config.require_readout):
                skips.append(CaseSkip(case_id=str(it.item_id), reason="readout_mismatch"))
                continue
            family = str(md.get("family", ""))
            if family not in {"standard_swap", "class_swap", "conflict_swap"}:
                skips.append(CaseSkip(case_id=str(it.item_id), reason=f"invalid_family:{family}"))
                continue
            patch_span = str(md.get("patch_span", ""))
            if patch_span not in {"class_span", "standard_span"}:
                skips.append(CaseSkip(case_id=str(it.item_id), reason=f"invalid_patch_span:{patch_span}"))
                continue
            expected_patch_span = None
            if family == "standard_swap":
                expected_patch_span = "standard_span"
            elif family == "class_swap":
                expected_patch_span = "class_span"
            if expected_patch_span is not None and patch_span != expected_patch_span:
                skips.append(
                    CaseSkip(
                        case_id=str(it.item_id),
                        reason=f"family_patch_span_mismatch:{family}:{patch_span}",
                    )
                )
                continue
            markers_obj = md.get("span_markers", None)
            if not isinstance(markers_obj, Mapping):
                skips.append(CaseSkip(case_id=str(it.item_id), reason="missing_span_markers"))
                continue

            strata_keys = (
                "family",
                "patch_span",
                "case_type",
                "label_source",
                "base_item_id",
                "receiver_group_id",
                "conflict_receiver_id",
                "conflict_direction",
                "donor_kind",
                "class_name",
                "class_kind",
                "cf_class_name",
                "cf_class_kind",
                "class_label",
                "standard_label",
                "donor_label",
                "standard_cm",
                "cf_standard_cm",
                "value_cm",
                "log_ratio",
                "cf_log_ratio",
                "polarity",
                "margin_bin",
                "prior_congruence",
                "unit",
                "template_id",
            )

            case_specs: list[tuple[str, str, str]] = [(str(it.item_id), patch_span, "primary")]
            if bool(self.config.include_value_placebo) and family == "standard_swap":
                case_specs.append((f"{it.item_id}__value_span_placebo", "value_span", "placebo"))

            for case_id, active_patch_span, case_type in case_specs:
                span_obj = markers_obj.get(active_patch_span, None)
                if not isinstance(span_obj, Mapping):
                    skips.append(CaseSkip(case_id=str(case_id), reason=f"missing_marker:{active_patch_span}"))
                    continue

                try:
                    base_marker = _marker_from_obj(span_obj.get("base"))
                    cf_marker = _marker_from_obj(span_obj.get("cf"))
                    recv_text = str(it.base.prompt)
                    donor_text = str(it.cf.prompt)
                    recv_span, _recv_token_ids = token_span_for_marker(tokenizer, recv_text, base_marker)
                    donor_span, _donor_token_ids = token_span_for_marker(tokenizer, donor_text, cf_marker)
                    if len(recv_span) != len(donor_span):
                        skips.append(CaseSkip(case_id=str(case_id), reason="named_span_len_mismatch"))
                        continue
                    recv_ids = tokenizer(recv_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                    donor_ids = tokenizer(donor_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                except Exception as e:
                    skips.append(CaseSkip(case_id=str(case_id), reason=f"{type(e).__name__}:{e}"))
                    continue

                strata = {k: _stratum_value(md.get(k)) for k in strata_keys if k in md}
                strata["patch_span"] = str(active_patch_span)
                strata["case_type"] = str(case_type)
                cases.append(
                    PatchingCase(
                        case_id=str(case_id),
                        receiver_prompt=recv_text,
                        donor_prompt=donor_text,
                        receiver_ids=recv_ids,
                        donor_ids=donor_ids,
                        receiver_span=tuple(int(i) for i in recv_span),
                        donor_span=tuple(int(i) for i in donor_span),
                        choices=it.choices,
                        expected_label=str(it.cf.expected_label),
                        strata=strata,
                        label_aggregation="logmeanexp",
                        effect_sign=1.0,
                    )
                )

        return cases, skips
