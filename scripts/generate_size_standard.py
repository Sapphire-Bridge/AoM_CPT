#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformers import AutoTokenizer, PreTrainedTokenizerBase


ROOT = Path(__file__).resolve().parents[1]
READOUT_NAME = "bigness_logscore_margin"
CHOICES = {"small": [" tiny", " small"], "large": [" large", " huge"]}


@dataclass(frozen=True)
class ClassSpec:
    name: str
    kind: str
    real_standard_cm: float | None


@dataclass(frozen=True)
class SpanMarker:
    text: str
    left: str
    right: str


@dataclass(frozen=True)
class TemplateSpec:
    template_id: int
    text: str
    class_left: str
    class_right: str
    standard_left: str
    standard_right: str
    value_left: str
    value_right: str


@dataclass(frozen=True)
class ClassTemplateSpec:
    template_id: int
    text: str
    class_left: str
    class_right: str
    value_left: str
    value_right: str


STANDARD_TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        template_id=1,
        text="The reference for this object is {standard} cm. The measured {class_name} is {value} cm. Therefore it is",
        class_left="The measured ",
        class_right=" is",
        standard_left="The reference for this object is ",
        standard_right=" cm",
        value_left=" is ",
        value_right=" cm. Therefore",
    ),
    TemplateSpec(
        template_id=2,
        text="{Article} {class_name} has a trial standard of {standard} cm. Its measured length is {value} cm, so it is",
        class_left="{Article} ",
        class_right=" has",
        standard_left="trial standard of ",
        standard_right=" cm",
        value_left="measured length is ",
        value_right=" cm, so",
    ),
    TemplateSpec(
        template_id=3,
        text="This {class_name} is measured at {value} cm. Relative to a {standard} cm standard, it is",
        class_left="This ",
        class_right=" is measured",
        standard_left="Relative to a ",
        standard_right=" cm standard",
        value_left="is measured at ",
        value_right=" cm.",
    ),
)


CLASS_SWAP_TEMPLATES: tuple[ClassTemplateSpec, ...] = (
    ClassTemplateSpec(
        template_id=1,
        text="For class {class_name}, {value} cm is",
        class_left="For class ",
        class_right=",",
        value_left=", ",
        value_right=" cm",
    ),
    ClassTemplateSpec(
        template_id=2,
        text="As {class_name} sizes go, {value} cm is",
        class_left="As ",
        class_right=" sizes",
        value_left="go, ",
        value_right=" cm",
    ),
    ClassTemplateSpec(
        template_id=3,
        text="Compared with a typical {class_name}, {value} cm is",
        class_left="typical ",
        class_right=",",
        value_left=", ",
        value_right=" cm",
    ),
)


CONFLICT_TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        template_id=1,
        text="For this object, the class is {class_name}. The stated reference is {standard} cm. Its measured length is {value} cm, so it is",
        class_left="the class is ",
        class_right=".",
        standard_left="The stated reference is ",
        standard_right=" cm",
        value_left="measured length is ",
        value_right=" cm, so",
    ),
    TemplateSpec(
        template_id=2,
        text="This item's class is {class_name}. The comparison standard is {standard} cm. The item measures {value} cm and is",
        class_left="class is ",
        class_right=".",
        standard_left="comparison standard is ",
        standard_right=" cm",
        value_left="item measures ",
        value_right=" cm",
    ),
    TemplateSpec(
        template_id=3,
        text="Category: {class_name}. Reference length: {standard} cm. Observed length: {value} cm. Verdict:",
        class_left="Category: ",
        class_right=".",
        standard_left="Reference length: ",
        standard_right=" cm",
        value_left="Observed length: ",
        value_right=" cm",
    ),
)


CLASS_SPECS: tuple[ClassSpec, ...] = (
    ClassSpec("ant", "real", 1.0),
    ClassSpec("human", "real", 170.0),
    ClassSpec("elephant", "real", 350.0),
    ClassSpec("glorb", "nonce", None),
    ClassSpec("dax", "nonce", None),
    ClassSpec("fenzel", "nonce", None),
)


# Same digit-count standard swaps, selected so one shared value flips the label.
# The exact log-ratios are recorded; bins are only coarse strata.
STANDARD_SWAP_SPECS: tuple[tuple[int, int, int, str], ...] = (
    (100, 102, 101, "borderline"),
    (100, 120, 110, "borderline"),
    (50, 75, 61, "borderline"),
    (100, 150, 125, "borderline"),
    (100, 200, 150, "clear"),
    (10, 90, 30, "clear"),
    (100, 900, 300, "clear"),
    (100, 999, 150, "extreme"),
)


# Class-prior swaps are number-free: labels are assigned from real-world class
# priors, so prompts must not state an explicit standard.
CLASS_SWAP_SPECS: tuple[tuple[str, str, int], ...] = (
    ("ant", "human", 100),
    ("ant", "elephant", 100),
    ("human", "elephant", 250),
    ("ant", "elephant", 10),
)


# Conflict rows are generated from low-prior/high-prior class pairs. The value
# is between the two class priors, and the two stated standards straddle the
# value while preserving equal digit counts for named-span patching.
CONFLICT_SWAP_SPECS: tuple[tuple[str, str, int, int, int], ...] = (
    ("ant", "human", 20, 10, 90),
    ("ant", "elephant", 300, 100, 900),
    ("human", "elephant", 250, 100, 900),
)


CONFLICT_STANDARD_PAIRS: tuple[tuple[int, int], ...] = (
    (10, 90),
    (50, 75),
    (100, 200),
    (100, 900),
    (100, 999),
)


# Dense standard-relative penumbra probes.  Centers are chosen away from
# digit-count boundaries so standard-span swaps remain tokenizer-stable.
PENUMBRA_STANDARD_CENTERS_CM: tuple[int, ...] = (50, 200, 500)
PENUMBRA_ABS_LOG_RATIOS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.15, 0.3, 0.5)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _try_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return str(out).strip()
    except Exception:
        return ""


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "x"


def _article_for(class_name: str, *, capitalize: bool = False) -> str:
    first = str(class_name).strip().lower()[:1]
    article = "an" if first in {"a", "e", "i", "o", "u"} else "a"
    return article.capitalize() if bool(capitalize) else article


def _format_context(
    text: str,
    *,
    class_name: str,
    standard: int | None = None,
    value: int | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "class_name": str(class_name),
        "article": _article_for(str(class_name)),
        "Article": _article_for(str(class_name), capitalize=True),
    }
    if standard is not None:
        kwargs["standard"] = int(standard)
    if value is not None:
        kwargs["value"] = int(value)
    return str(text).format(**kwargs)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _load_tokenizers(
    model_names: list[str],
    *,
    local_files_only: bool,
    trust_remote_code: bool,
) -> dict[str, PreTrainedTokenizerBase]:
    tokenizers: dict[str, PreTrainedTokenizerBase] = {}
    for model_name in model_names:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
        )
        if not bool(getattr(tok, "is_fast", False)):
            raise ValueError(f"Tokenizer for {model_name!r} is not fast; offset_mapping is required.")
        tokenizers[str(model_name)] = tok
    return tokenizers


def _find_anchored_char_span(text: str, marker: SpanMarker) -> tuple[int, int]:
    anchor = f"{marker.left}{marker.text}{marker.right}"
    start = text.find(anchor)
    if start < 0:
        raise ValueError(f"Anchor not found: {anchor!r} in {text!r}")
    second = text.find(anchor, start + 1)
    if second >= 0:
        raise ValueError(f"Anchor not unique: {anchor!r} in {text!r}")
    span_start = start + len(marker.left)
    span_end = span_start + len(marker.text)
    return int(span_start), int(span_end)


def token_span_for_marker(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    marker: SpanMarker,
) -> tuple[list[int], list[int]]:
    char_start, char_end = _find_anchored_char_span(text, marker)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", None)
    if offsets is None:
        raise ValueError("Tokenizer must support return_offsets_mapping.")
    offsets_list = offsets[0].tolist()
    input_ids = enc["input_ids"][0].tolist()
    span = [i for i, (s, e) in enumerate(offsets_list) if (s < char_end and e > char_start)]
    if not span:
        raise ValueError(f"Could not map marker to tokens: {marker!r}")
    if span != list(range(span[0], span[-1] + 1)):
        raise ValueError(f"Marker maps to non-contiguous token span: {marker!r}")
    for i in span:
        s, e = offsets_list[i]
        left_extra = text[int(s) : char_start] if int(s) < char_start else ""
        if (left_extra and left_extra.strip()) or int(e) > char_end:
            raise ValueError(
                f"Token span bleeds outside marker text={marker.text!r}: token_offset={(s, e)} char_span={(char_start, char_end)}"
            )
    return [int(i) for i in span], [int(input_ids[i]) for i in span]


def _span_lengths_by_tokenizer(
    tokenizers: Mapping[str, PreTrainedTokenizerBase],
    *,
    base_prompt: str,
    cf_prompt: str,
    base_marker: SpanMarker,
    cf_marker: SpanMarker,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for model_name, tok in tokenizers.items():
        base_span, _base_ids = token_span_for_marker(tok, base_prompt, base_marker)
        cf_span, _cf_ids = token_span_for_marker(tok, cf_prompt, cf_marker)
        out[str(model_name)] = {"base": int(len(base_span)), "cf": int(len(cf_span))}
    return out


def _validate_equal_span_lengths(lengths: Mapping[str, Mapping[str, int]]) -> tuple[bool, str]:
    bad = {m: dict(v) for m, v in lengths.items() if int(v.get("base", -1)) != int(v.get("cf", -2))}
    if bad:
        return False, "span_len_mismatch:" + json.dumps(bad, sort_keys=True)
    return True, "ok"


def _render_prompt(template: TemplateSpec, *, class_name: str, standard: int, value: int) -> str:
    return _format_context(template.text, class_name=class_name, standard=int(standard), value=int(value))


def _render_class_prompt(template: ClassTemplateSpec, *, class_name: str, value: int) -> str:
    return _format_context(template.text, class_name=class_name, value=int(value))


def _markers_for(
    template: TemplateSpec,
    *,
    class_name: str,
    standard: int,
    value: int,
) -> dict[str, SpanMarker]:
    return {
        "class_span": SpanMarker(
            str(class_name),
            _format_context(template.class_left, class_name=class_name, standard=int(standard), value=int(value)),
            _format_context(template.class_right, class_name=class_name, standard=int(standard), value=int(value)),
        ),
        "standard_span": SpanMarker(
            str(int(standard)),
            _format_context(template.standard_left, class_name=class_name, standard=int(standard), value=int(value)),
            _format_context(template.standard_right, class_name=class_name, standard=int(standard), value=int(value)),
        ),
        "value_span": SpanMarker(
            str(int(value)),
            _format_context(template.value_left, class_name=class_name, standard=int(standard), value=int(value)),
            _format_context(template.value_right, class_name=class_name, standard=int(standard), value=int(value)),
        ),
    }


def _class_markers_for(
    template: ClassTemplateSpec,
    *,
    class_name: str,
    value: int,
) -> dict[str, SpanMarker]:
    return {
        "class_span": SpanMarker(
            str(class_name),
            _format_context(template.class_left, class_name=class_name, value=int(value)),
            _format_context(template.class_right, class_name=class_name, value=int(value)),
        ),
        "value_span": SpanMarker(
            str(int(value)),
            _format_context(template.value_left, class_name=class_name, value=int(value)),
            _format_context(template.value_right, class_name=class_name, value=int(value)),
        ),
    }


def _expected_from_standard(*, value: int, standard: int) -> str:
    if value == standard:
        raise ValueError("value must not equal standard")
    return "large" if int(value) > int(standard) else "small"


def _class_prior_cm(cls: ClassSpec, model_priors_cm: Mapping[str, float] | None = None) -> float | None:
    if model_priors_cm is not None and cls.name in model_priors_cm:
        return float(model_priors_cm[cls.name])
    return None if cls.real_standard_cm is None else float(cls.real_standard_cm)


def _expected_from_class_prior(
    *,
    value: int,
    cls: ClassSpec,
    model_priors_cm: Mapping[str, float] | None = None,
) -> str:
    prior = _class_prior_cm(cls, model_priors_cm=model_priors_cm)
    if prior is None:
        raise ValueError(f"Class {cls.name!r} has no real prior")
    if float(value) == float(prior):
        raise ValueError("value must not equal class prior")
    return "large" if float(value) > float(prior) else "small"


def _margin_bin(log_ratio: float) -> str:
    x = abs(float(log_ratio))
    if x <= 0.35:
        return "borderline"
    if x <= 1.5:
        return "clear"
    return "extreme"


def _penumbra_margin_bin(abs_log_ratio: float) -> str:
    x = abs(float(abs_log_ratio))
    if x <= 0.07:
        return "borderline"
    if x <= 0.15:
        return "near_boundary"
    if x <= 0.35:
        return "shoulder"
    return "clear"


def _penumbra_standard_specs() -> tuple[tuple[int, int, int, float], ...]:
    specs: set[tuple[int, int, int, float]] = set()
    for center in PENUMBRA_STANDARD_CENTERS_CM:
        for abs_delta in PENUMBRA_ABS_LOG_RATIOS:
            low_s = int(round(float(center) * math.exp(-float(abs_delta))))
            high_s = int(round(float(center) * math.exp(float(abs_delta))))
            if not (0 < low_s < int(center) < high_s):
                continue
            if len(str(low_s)) != len(str(high_s)):
                continue
            specs.add((int(low_s), int(high_s), int(center), float(abs_delta)))
    return tuple(sorted(specs, key=lambda x: (x[2], x[3], x[0], x[1])))


def _prior_congruence(standard: int, cls: ClassSpec) -> float | None:
    if cls.real_standard_cm is None:
        return None
    return float(abs(math.log(float(standard) / float(cls.real_standard_cm))))


def _row(
    *,
    item_id: str,
    base_prompt: str,
    base_label: str,
    cf_prompt: str,
    cf_label: str,
    metadata: dict[str, Any],
    contrast_labels: tuple[str, str] = ("small", "large"),
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "base": {"prompt": base_prompt, "expected_label": base_label},
        "cf": {"prompt": cf_prompt, "expected_label": cf_label},
        "choices": CHOICES,
        "intervention_type": "size_standard",
        "contrast_labels": [str(contrast_labels[0]), str(contrast_labels[1])],
        "expected_effect": "shift",
        "metadata": metadata,
    }


def _load_model_priors_cm(path: str | Path | None) -> dict[str, float]:
    if path is None or not str(path).strip():
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, Mapping):
        raise ValueError(f"Prior file must contain a JSON object: {path}")
    raw = obj.get("model_priors", obj.get("priors", obj))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Prior file has no model_priors mapping: {path}")
    out: dict[str, float] = {}
    for name, val in raw.items():
        if isinstance(val, Mapping):
            if "model_prior_cm_argmax" in val:
                out[str(name)] = float(val["model_prior_cm_argmax"])
            elif "prior_cm" in val:
                out[str(name)] = float(val["prior_cm"])
        else:
            out[str(name)] = float(val)
    return out


def _single_edit_error(base_prompt: str, cf_prompt: str, old: str, new: str) -> str | None:
    rebuilt = base_prompt.replace(old, new, 1)
    if rebuilt != cf_prompt:
        return f"prompt_pair_not_single_edit:{old!r}->{new!r}"
    if base_prompt.count(old) != 1:
        return f"base_edit_target_not_unique:{old!r}"
    if cf_prompt.count(new) != 1:
        return f"cf_edit_target_not_unique:{new!r}"
    return None


def _single_edit_assertion(base_prompt: str, cf_prompt: str, old: str, new: str) -> None:
    err = _single_edit_error(base_prompt, cf_prompt, old, new)
    if err is not None:
        raise ValueError(err)


def _class_by_name() -> dict[str, ClassSpec]:
    return {c.name: c for c in CLASS_SPECS}


def _dynamic_class_swap_specs(
    classes_by_name: Mapping[str, ClassSpec],
    *,
    model_priors_cm: Mapping[str, float] | None,
) -> tuple[tuple[str, str, int], ...]:
    if not model_priors_cm:
        return CLASS_SWAP_SPECS
    real = sorted(
        [
        (name, float(model_priors_cm[name]))
        for name in sorted(model_priors_cm)
        if name in classes_by_name and classes_by_name[name].kind == "real"
        ],
        key=lambda kv: (float(kv[1]), str(kv[0])),
    )
    out: set[tuple[str, str, int]] = set()
    for i, (low_name, low_prior) in enumerate(real):
        for high_name, high_prior in real[i + 1 :]:
            if not (math.isfinite(low_prior) and math.isfinite(high_prior) and 0 < low_prior < high_prior):
                continue
            if high_prior / low_prior < 1.4:
                continue
            lo = math.log(low_prior)
            hi = math.log(high_prior)
            for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
                value = int(round(math.exp(lo + frac * (hi - lo))))
                if float(low_prior) < float(value) < float(high_prior):
                    out.add((low_name, high_name, int(value)))
    return tuple(sorted(out, key=lambda x: (x[0], x[1], x[2])))


def _dynamic_conflict_swap_specs(
    classes_by_name: Mapping[str, ClassSpec],
    *,
    model_priors_cm: Mapping[str, float] | None,
) -> tuple[tuple[str, str, int, int, int], ...]:
    if not model_priors_cm:
        return CONFLICT_SWAP_SPECS
    class_specs = _dynamic_class_swap_specs(classes_by_name, model_priors_cm=model_priors_cm)
    out: set[tuple[str, str, int, int, int]] = set()
    for low_name, high_name, value in class_specs:
        for low_s, high_s in CONFLICT_STANDARD_PAIRS:
            if int(low_s) < int(value) < int(high_s):
                out.add((low_name, high_name, int(value), int(low_s), int(high_s)))
                break
    return tuple(sorted(out, key=lambda x: (x[0], x[1], x[2], x[3], x[4])))


def generate_size_standard_rows(
    *,
    tokenizers: Mapping[str, PreTrainedTokenizerBase],
    strict_generation: bool = False,
    model_priors_cm: Mapping[str, float] | None = None,
    require_model_priors: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    generation_error_counts: Counter[str] = Counter()
    span_length_examples: dict[str, Any] = {}
    classes_by_name = _class_by_name()

    if bool(require_model_priors):
        needed = {
            name
            for spec in list(CLASS_SWAP_SPECS) + [(a, b, v) for a, b, v, _lo, _hi in CONFLICT_SWAP_SPECS]
            for name in (spec[0], spec[1])
        }
        missing = sorted(name for name in needed if name not in (model_priors_cm or {}))
        nonpositive = sorted(
            name
            for name in needed
            if name in (model_priors_cm or {}) and (not math.isfinite(float((model_priors_cm or {})[name])) or float((model_priors_cm or {})[name]) <= 0)
        )
        if missing or nonpositive:
            raise ValueError(
                "Model-prior mode requires finite positive priors for all class/conflict classes; "
                f"missing={missing} nonpositive={nonpositive}"
            )

    class_swap_specs = _dynamic_class_swap_specs(classes_by_name, model_priors_cm=model_priors_cm)
    conflict_swap_specs = _dynamic_conflict_swap_specs(classes_by_name, model_priors_cm=model_priors_cm)

    def record_single_edit_skip(reason: str) -> bool:
        if bool(strict_generation):
            raise ValueError(str(reason))
        skip_reasons[str(reason)] += 1
        return False

    def record_generation_error(exc: Exception) -> None:
        if bool(strict_generation):
            raise exc
        generation_error_counts[f"{type(exc).__name__}:{exc}"] += 1

    def try_add(row: dict[str, Any], *, base_marker: SpanMarker, cf_marker: SpanMarker) -> None:
        md = row["metadata"]
        try:
            lengths = _span_lengths_by_tokenizer(
                tokenizers,
                base_prompt=row["base"]["prompt"],
                cf_prompt=row["cf"]["prompt"],
                base_marker=base_marker,
                cf_marker=cf_marker,
            )
            ok, reason = _validate_equal_span_lengths(lengths)
            if not ok:
                skip_reasons[reason] += 1
                return
            md["tokenizer_span_lengths"] = lengths
            span_length_examples.setdefault(str(md["patch_span"]), lengths)
            rows.append(row)
        except Exception as e:
            record_generation_error(e)

    for template in STANDARD_TEMPLATES:
        for cls in CLASS_SPECS:
            for low_s, high_s, value, coarse_bin in STANDARD_SWAP_SPECS:
                # Counterfactual validation requires contrast_labels[0] -> contrast_labels[1].
                # With the fixed ["small", "large"] contrast, orient every row as small -> large.
                base_s, cf_s = high_s, low_s
                base_label = _expected_from_standard(value=value, standard=base_s)
                cf_label = _expected_from_standard(value=value, standard=cf_s)
                if (base_label, cf_label) != ("small", "large"):
                    skip_reasons["standard_swap_not_small_to_large"] += 1
                    continue
                base_prompt = _render_prompt(template, class_name=cls.name, standard=base_s, value=value)
                cf_prompt = _render_prompt(template, class_name=cls.name, standard=cf_s, value=value)
                edit_error = _single_edit_error(base_prompt, cf_prompt, str(base_s), str(cf_s))
                if edit_error is not None and not record_single_edit_skip(edit_error):
                    continue
                base_markers = _markers_for(template, class_name=cls.name, standard=base_s, value=value)
                cf_markers = _markers_for(template, class_name=cls.name, standard=cf_s, value=value)
                log_ratio = math.log(float(value) / float(base_s))
                item_id = (
                    f"size-standard_swap-{_slug(cls.name)}-S{base_s}-V{value}-"
                    f"{base_label}_to_{cf_label}-m{coarse_bin}-t{template.template_id}"
                )
                metadata = {
                    "family": "standard_swap",
                    "patch_span": "standard_span",
                    "label_source": "stated_standard",
                    "class_name": cls.name,
                    "class_kind": cls.kind,
                    "standard_cm": int(base_s),
                    "cf_standard_cm": int(cf_s),
                    "value_cm": int(value),
                    "stated_standard_label": base_label,
                    "cf_stated_standard_label": cf_label,
                    "log_ratio": float(log_ratio),
                    "cf_log_ratio": float(math.log(float(value) / float(cf_s))),
                    "polarity": f"{base_label}_to_{cf_label}",
                    "margin_bin": str(coarse_bin),
                    "prior_congruence": _prior_congruence(base_s, cls),
                    "cf_prior_congruence": _prior_congruence(cf_s, cls),
                    "unit": "cm",
                    "template_id": int(template.template_id),
                    "base_item_id": item_id,
                    "readout": READOUT_NAME,
                    "span_markers": {
                        "standard_span": {"base": base_markers["standard_span"].__dict__, "cf": cf_markers["standard_span"].__dict__},
                        "class_span": {"base": base_markers["class_span"].__dict__, "cf": cf_markers["class_span"].__dict__},
                        "value_span": {"base": base_markers["value_span"].__dict__, "cf": cf_markers["value_span"].__dict__},
                    },
                }
                try_add(
                    _row(
                        item_id=item_id,
                        base_prompt=base_prompt,
                        base_label=base_label,
                        cf_prompt=cf_prompt,
                        cf_label=cf_label,
                        metadata=metadata,
                    ),
                    base_marker=base_markers["standard_span"],
                    cf_marker=cf_markers["standard_span"],
                )

    for template in STANDARD_TEMPLATES:
        for cls in CLASS_SPECS:
            for low_s, high_s, value, target_abs_delta in _penumbra_standard_specs():
                for base_s, cf_s, base_label, cf_label, polarity in (
                    (high_s, low_s, "small", "large", "small_to_large"),
                    (low_s, high_s, "large", "small", "large_to_small"),
                ):
                    observed_base_label = _expected_from_standard(value=value, standard=base_s)
                    observed_cf_label = _expected_from_standard(value=value, standard=cf_s)
                    if (observed_base_label, observed_cf_label) != (base_label, cf_label):
                        skip_reasons["penumbra_standard_label_mismatch"] += 1
                        continue
                    base_prompt = _render_prompt(template, class_name=cls.name, standard=base_s, value=value)
                    cf_prompt = _render_prompt(template, class_name=cls.name, standard=cf_s, value=value)
                    edit_error = _single_edit_error(base_prompt, cf_prompt, str(base_s), str(cf_s))
                    if edit_error is not None and not record_single_edit_skip(edit_error):
                        continue
                    base_markers = _markers_for(template, class_name=cls.name, standard=base_s, value=value)
                    cf_markers = _markers_for(template, class_name=cls.name, standard=cf_s, value=value)
                    log_ratio = float(math.log(float(value) / float(base_s)))
                    cf_log_ratio = float(math.log(float(value) / float(cf_s)))
                    abs_log_ratio = float(abs(log_ratio))
                    value_pair_id = (
                        f"penumbra-standard-C{value}-D{str(target_abs_delta).replace('.', 'p')}-"
                        f"{_slug(cls.name)}-t{template.template_id}"
                    )
                    coarse_bin = _penumbra_margin_bin(abs_log_ratio)
                    item_id = (
                        f"size-penumbra_standard-{_slug(cls.name)}-C{value}-S{base_s}-to{cf_s}-"
                        f"{polarity}-m{coarse_bin}-d{str(target_abs_delta).replace('.', 'p')}-t{template.template_id}"
                    )
                    metadata = {
                        "family": "penumbra_standard",
                        "patch_span": "standard_span",
                        "label_source": "stated_standard_penumbra",
                        "class_name": cls.name,
                        "class_kind": cls.kind,
                        "standard_cm": int(base_s),
                        "cf_standard_cm": int(cf_s),
                        "value_cm": int(value),
                        "center_value_cm": int(value),
                        "stated_standard_label": base_label,
                        "cf_stated_standard_label": cf_label,
                        "log_ratio": float(log_ratio),
                        "cf_log_ratio": float(cf_log_ratio),
                        "abs_log_ratio": float(abs_log_ratio),
                        "cf_abs_log_ratio": float(abs(cf_log_ratio)),
                        "target_abs_log_ratio": float(target_abs_delta),
                        "distance_from_standard": float(log_ratio),
                        "cf_distance_from_standard": float(cf_log_ratio),
                        "polarity": str(polarity),
                        "margin_bin": str(coarse_bin),
                        "penumbra_axis": "stated_standard",
                        "penumbral_relation": "standard_straddle",
                        "value_pair_id": value_pair_id,
                        "rank_in_chain": -1 if base_label == "small" else 1,
                        "monotonicity_direction": "lower_standard_increases_bigness",
                        "prior_congruence": _prior_congruence(base_s, cls),
                        "cf_prior_congruence": _prior_congruence(cf_s, cls),
                        "unit": "cm",
                        "template_id": int(template.template_id),
                        "base_item_id": item_id,
                        "readout": READOUT_NAME,
                        "span_markers": {
                            "standard_span": {"base": base_markers["standard_span"].__dict__, "cf": cf_markers["standard_span"].__dict__},
                            "class_span": {"base": base_markers["class_span"].__dict__, "cf": cf_markers["class_span"].__dict__},
                            "value_span": {"base": base_markers["value_span"].__dict__, "cf": cf_markers["value_span"].__dict__},
                        },
                    }
                    try_add(
                        _row(
                            item_id=item_id,
                            base_prompt=base_prompt,
                            base_label=base_label,
                            cf_prompt=cf_prompt,
                            cf_label=cf_label,
                            metadata=metadata,
                            contrast_labels=(base_label, cf_label),
                        ),
                        base_marker=base_markers["standard_span"],
                        cf_marker=cf_markers["standard_span"],
                    )

    for template in CLASS_SWAP_TEMPLATES:
        for base_class, cf_class, value in class_swap_specs:
            base_cls = classes_by_name[base_class]
            cf_cls = classes_by_name[cf_class]
            try:
                labels = (
                    _expected_from_class_prior(value=value, cls=base_cls, model_priors_cm=model_priors_cm),
                    _expected_from_class_prior(value=value, cls=cf_cls, model_priors_cm=model_priors_cm),
                )
            except ValueError as e:
                skip_reasons[f"class_swap_prior_incompatible:{e}"] += 1
                continue
            if labels == ("small", "large"):
                cls_a, cls_b = base_cls, cf_cls
            elif labels == ("large", "small"):
                cls_a, cls_b = cf_cls, base_cls
            else:
                skip_reasons["class_swap_no_label_flip"] += 1
                continue
            base_label = "small"
            cf_label = "large"
            base_prompt = _render_class_prompt(template, class_name=cls_a.name, value=value)
            cf_prompt = _render_class_prompt(template, class_name=cls_b.name, value=value)
            edit_error = _single_edit_error(base_prompt, cf_prompt, cls_a.name, cls_b.name)
            if edit_error is not None and not record_single_edit_skip(edit_error):
                continue
            base_markers = _class_markers_for(template, class_name=cls_a.name, value=value)
            cf_markers = _class_markers_for(template, class_name=cls_b.name, value=value)
            coarse_bin = "prior_only"
            item_id = (
                f"size-class_swap-{_slug(cls_a.name)}_to_{_slug(cls_b.name)}-V{value}-"
                f"{base_label}_to_{cf_label}-m{coarse_bin}-t{template.template_id}"
            )
            metadata = {
                "family": "class_swap",
                "patch_span": "class_span",
                "label_source": "class_prior",
                "class_name": cls_a.name,
                "class_kind": cls_a.kind,
                "cf_class_name": cls_b.name,
                "cf_class_kind": cls_b.kind,
                "class_pair": f"{cls_a.name}_to_{cls_b.name}",
                "value_cm": int(value),
                "model_prior_cm": _class_prior_cm(cls_a, model_priors_cm=model_priors_cm),
                "cf_model_prior_cm": _class_prior_cm(cls_b, model_priors_cm=model_priors_cm),
                "class_prior_label": base_label,
                "cf_class_prior_label": cf_label,
                "polarity": f"{base_label}_to_{cf_label}",
                "margin_bin": coarse_bin,
                "unit": "cm",
                "template_id": int(template.template_id),
                "base_item_id": item_id,
                "readout": READOUT_NAME,
                "span_markers": {
                    "class_span": {"base": base_markers["class_span"].__dict__, "cf": cf_markers["class_span"].__dict__},
                    "value_span": {"base": base_markers["value_span"].__dict__, "cf": cf_markers["value_span"].__dict__},
                },
            }
            try_add(
                _row(
                    item_id=item_id,
                    base_prompt=base_prompt,
                    base_label=base_label,
                    cf_prompt=cf_prompt,
                    cf_label=cf_label,
                    metadata=metadata,
                ),
                base_marker=base_markers["class_span"],
                cf_marker=cf_markers["class_span"],
            )

    for template in CONFLICT_TEMPLATES:
        for low_class, high_class, value, low_s, high_s in conflict_swap_specs:
            low_cls = classes_by_name[low_class]
            high_cls = classes_by_name[high_class]
            try:
                low_label = _expected_from_class_prior(value=value, cls=low_cls, model_priors_cm=model_priors_cm)
                high_label = _expected_from_class_prior(value=value, cls=high_cls, model_priors_cm=model_priors_cm)
            except Exception as e:
                skip_reasons[f"conflict_prior_incompatible:{e}"] += 1
                continue
            if (low_label, high_label) != ("large", "small"):
                skip_reasons["conflict_class_pair_not_large_small"] += 1
                continue
            if (_expected_from_standard(value=value, standard=low_s), _expected_from_standard(value=value, standard=high_s)) != (
                "large",
                "small",
            ):
                skip_reasons["conflict_standard_pair_not_large_small"] += 1
                continue

            conflict_specs = (
                {
                    "direction": "class_small_standard_large",
                    "receiver_class": high_cls,
                    "receiver_standard": low_s,
                    "class_label": "small",
                    "standard_label": "large",
                    "class_sibling_class": high_cls,
                    "class_sibling_standard": high_s,
                    "standard_sibling_class": low_cls,
                    "standard_sibling_standard": low_s,
                },
                {
                    "direction": "class_large_standard_small",
                    "receiver_class": low_cls,
                    "receiver_standard": high_s,
                    "class_label": "large",
                    "standard_label": "small",
                    "class_sibling_class": low_cls,
                    "class_sibling_standard": low_s,
                    "standard_sibling_class": high_cls,
                    "standard_sibling_standard": high_s,
                },
            )

            for spec in conflict_specs:
                recv_cls: ClassSpec = spec["receiver_class"]  # type: ignore[assignment]
                recv_s = int(spec["receiver_standard"])
                base_prompt = _render_prompt(template, class_name=recv_cls.name, standard=recv_s, value=value)
                base_markers = _markers_for(template, class_name=recv_cls.name, standard=recv_s, value=value)
                sibling_class_cls: ClassSpec = spec["class_sibling_class"]  # type: ignore[assignment]
                sibling_class_s = int(spec["class_sibling_standard"])
                sibling_standard_cls: ClassSpec = spec["standard_sibling_class"]  # type: ignore[assignment]
                sibling_standard_s = int(spec["standard_sibling_standard"])
                class_sibling_prompt = _render_prompt(
                    template,
                    class_name=sibling_class_cls.name,
                    standard=sibling_class_s,
                    value=value,
                )
                standard_sibling_prompt = _render_prompt(
                    template,
                    class_name=sibling_standard_cls.name,
                    standard=sibling_standard_s,
                    value=value,
                )

                donor_specs = (
                    ("class_span", sibling_standard_cls, sibling_standard_s, str(spec["standard_label"])),
                    ("standard_span", sibling_class_cls, sibling_class_s, str(spec["class_label"])),
                )
                for patch_span, donor_cls, donor_s, donor_label in donor_specs:
                    cf_prompt = _render_prompt(template, class_name=donor_cls.name, standard=int(donor_s), value=value)
                    cf_markers = _markers_for(template, class_name=donor_cls.name, standard=int(donor_s), value=value)
                    base_label = "small" if donor_label == "large" else "large"
                    contrast = (base_label, donor_label)
                    edit_old = recv_cls.name if patch_span == "class_span" else str(recv_s)
                    edit_new = donor_cls.name if patch_span == "class_span" else str(donor_s)
                    edit_error = _single_edit_error(base_prompt, cf_prompt, edit_old, edit_new)
                    if edit_error is not None and not record_single_edit_skip(edit_error):
                        continue
                    item_id = (
                        f"size-conflict_swap-{_slug(str(spec['direction']))}-{_slug(recv_cls.name)}-"
                        f"S{recv_s}-V{value}-{patch_span}-to_{donor_label}-t{template.template_id}"
                    )
                    metadata = {
                    "family": "conflict_swap",
                    "patch_span": str(patch_span),
                    "label_source": "cue_conflict",
                    "conflict_receiver_id": (
                        f"conflict-{_slug(str(spec['direction']))}-{_slug(recv_cls.name)}-S{recv_s}-V{value}-t{template.template_id}"
                    ),
                    "receiver_group_id": (
                        f"conflict-{_slug(str(spec['direction']))}-{_slug(recv_cls.name)}-S{recv_s}-V{value}-t{template.template_id}"
                    ),
                    "conflict_direction": str(spec["direction"]),
                    "donor_kind": "class_donor" if patch_span == "class_span" else "standard_donor",
                    "class_name": recv_cls.name,
                        "class_kind": recv_cls.kind,
                        "cf_class_name": donor_cls.name,
                        "cf_class_kind": donor_cls.kind,
                        "standard_cm": int(recv_s),
                        "cf_standard_cm": int(donor_s),
                        "value_cm": int(value),
                        "class_label": str(spec["class_label"]),
                        "standard_label": str(spec["standard_label"]),
                        "donor_label": str(donor_label),
                        "polarity": f"{base_label}_to_{donor_label}",
                        "margin_bin": _margin_bin(math.log(float(value) / float(recv_s))),
                        "log_ratio": float(math.log(float(value) / float(recv_s))),
                        "cf_log_ratio": float(math.log(float(value) / float(donor_s))),
                        "model_prior_cm": _class_prior_cm(recv_cls, model_priors_cm=model_priors_cm),
                        "cf_model_prior_cm": _class_prior_cm(donor_cls, model_priors_cm=model_priors_cm),
                        "prior_congruence": _prior_congruence(recv_s, recv_cls),
                        "cf_prior_congruence": _prior_congruence(int(donor_s), donor_cls),
                        "unit": "cm",
                        "template_id": int(template.template_id),
                        "base_item_id": item_id,
                        "readout": READOUT_NAME,
                        "conflict_gate_siblings": {
                            "class": {"prompt": class_sibling_prompt, "expected_label": str(spec["class_label"])},
                            "standard": {"prompt": standard_sibling_prompt, "expected_label": str(spec["standard_label"])},
                        },
                        "span_markers": {
                            "standard_span": {"base": base_markers["standard_span"].__dict__, "cf": cf_markers["standard_span"].__dict__},
                            "class_span": {"base": base_markers["class_span"].__dict__, "cf": cf_markers["class_span"].__dict__},
                            "value_span": {"base": base_markers["value_span"].__dict__, "cf": cf_markers["value_span"].__dict__},
                        },
                    }
                    marker_key = "class_span" if patch_span == "class_span" else "standard_span"
                    try_add(
                        _row(
                            item_id=item_id,
                            base_prompt=base_prompt,
                            base_label=base_label,
                            cf_prompt=cf_prompt,
                            cf_label=donor_label,
                            metadata=metadata,
                            contrast_labels=contrast,
                        ),
                        base_marker=base_markers[marker_key],
                        cf_marker=cf_markers[marker_key],
                    )

    ids = [str(r["item_id"]) for r in rows]
    dupes = sorted(k for k, v in Counter(ids).items() if v > 1)
    if dupes:
        raise ValueError(f"Duplicate item_id values: {dupes[:8]}")

    summary = {
        "generated_at_utc": _utc_now_iso(),
        "generator": "scripts/generate_size_standard.py",
        "git_commit": _try_git_commit(),
        "readout": READOUT_NAME,
        "n_rows": int(len(rows)),
        "counts_by_family": dict(sorted(Counter(str(r["metadata"]["family"]) for r in rows).items())),
        "counts_by_patch_span": dict(sorted(Counter(str(r["metadata"]["patch_span"]) for r in rows).items())),
        "model_priors_used": {str(k): float(v) for k, v in sorted((model_priors_cm or {}).items())},
        "class_swap_specs_used": [list(x) for x in class_swap_specs],
        "conflict_swap_specs_used": [list(x) for x in conflict_swap_specs],
        "penumbra_standard_specs_used": [list(x) for x in _penumbra_standard_specs()],
        "skip_reason_counts": dict(sorted(skip_reasons.items())),
        "generation_error_counts": dict(sorted(generation_error_counts.items())),
        "tokenizer_span_length_examples": span_length_examples,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate size/standard dissociation CF rows.")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "data_size_standard"))
    p.add_argument("--out_name", type=str, default="size_standard")
    p.add_argument(
        "--tokenizer_models",
        nargs="+",
        default=["gpt2", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"],
    )
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--model_priors_path",
        type=str,
        default="",
        help="Optional JSON from scripts/elicite_size_priors.py; overrides real-world class priors for class/conflict rows.",
    )
    p.add_argument("--min_rows", type=int, default=1, help="Fail if fewer rows are generated.")
    p.add_argument("--strict_generation", action="store_true", help="Raise on candidate generation errors.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(str(args.out_dir))
    out_name = str(args.out_name)
    tokenizers = _load_tokenizers(
        [str(x) for x in args.tokenizer_models],
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    model_priors_cm = _load_model_priors_cm(str(args.model_priors_path))
    rows, summary = generate_size_standard_rows(
        tokenizers=tokenizers,
        strict_generation=bool(args.strict_generation),
        model_priors_cm=model_priors_cm,
        require_model_priors=bool(str(args.model_priors_path).strip()),
    )
    summary["tokenizer_models_checked"] = [str(x) for x in args.tokenizer_models]
    summary["strict_generation"] = bool(args.strict_generation)
    summary["min_rows"] = int(args.min_rows)
    if len(rows) < int(args.min_rows):
        raise ValueError(f"Generated {len(rows)} rows, below --min_rows {int(args.min_rows)}")
    out_jsonl = out_dir / f"{out_name}.jsonl"
    out_summary = out_dir / f"{out_name}.summary.json"
    _write_jsonl(rows, out_jsonl)
    _write_json(out_summary, summary)
    print(f"Wrote {len(rows)} size-standard rows to {out_jsonl}")
    print(f"Wrote summary to {out_summary}")


if __name__ == "__main__":
    main()
