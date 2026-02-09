from __future__ import annotations

from typing import Mapping, Sequence

from .schemas import CoherenceItem, CounterfactualPair, DisambPair


def _validate_prompt_cont_boundary(*, prompt: str, continuation: str, item_id: str, field: str) -> None:
    if not prompt:
        raise ValueError(f"{item_id}: empty prompt/context for {field}")
    if not continuation:
        raise ValueError(f"{item_id}: empty continuation in {field}")
    if (not prompt[-1].isspace()) and (not continuation[0].isspace()):
        raise ValueError(
            f"{item_id}: continuation boundary likely wrong in {field} "
            f"(prompt does not end with whitespace and continuation does not start with whitespace)"
        )


def _validate_choices(
    *,
    choices: Mapping[str, Sequence[str]],
    expected_labels: Sequence[str],
    prompts: Sequence[str],
    item_id: str,
    field: str,
    require_equal_counts: bool,
) -> None:
    if not choices:
        raise ValueError(f"{item_id}: empty choices for {field}")

    for lab in expected_labels:
        if lab not in choices:
            raise ValueError(f"{item_id}: expected_label={lab!r} missing from choices for {field}")

    counts = set()
    for label, conts in choices.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{item_id}: invalid label in choices for {field}: {label!r}")
        if not conts:
            raise ValueError(f"{item_id}: empty continuation list for label={label!r} in {field}")
        counts.add(len(conts))
        for cont in conts:
            if not isinstance(cont, str):
                raise ValueError(f"{item_id}: non-string continuation for label={label!r} in {field}")
            for prompt in prompts:
                _validate_prompt_cont_boundary(prompt=prompt, continuation=cont, item_id=item_id, field=field)

    if require_equal_counts and len(counts) > 1:
        raise ValueError(f"{item_id}: unequal continuation counts per label in {field}: {sorted(counts)}")


def validate_disamb_pairs(items: Sequence[DisambPair], *, require_equal_choice_counts: bool = True) -> None:
    for it in items:
        _validate_choices(
            choices=it.choices,
            expected_labels=(it.a.expected_label, it.b.expected_label),
            prompts=(it.a.prompt, it.b.prompt),
            item_id=str(it.pair_id),
            field="disamb.choices",
            require_equal_counts=require_equal_choice_counts,
        )


def validate_counterfactual_pairs(items: Sequence[CounterfactualPair], *, require_equal_choice_counts: bool = True) -> None:
    for it in items:
        if it.expected_effect not in {"shift", "invariant", "graded"}:
            raise ValueError(
                f"{it.item_id}: invalid expected_effect={it.expected_effect!r} (expected 'shift'|'invariant'|'graded')"
            )
        if it.expected_effect == "shift" and it.base.expected_label == it.cf.expected_label:
            raise ValueError(
                f"{it.item_id}: shift item must satisfy base.expected_label!=cf.expected_label "
                f"(got {it.base.expected_label!r})"
            )

        if it.contrast_labels is None:
            if it.expected_effect in {"invariant", "graded"}:
                raise ValueError(f"{it.item_id}: {it.expected_effect} CF item requires contrast_labels")
        else:
            L0, L1 = it.contrast_labels
            if L0 == L1:
                raise ValueError(f"{it.item_id}: contrast_labels must differ (got {it.contrast_labels!r})")
            if L0 not in it.choices or L1 not in it.choices:
                raise ValueError(f"{it.item_id}: contrast_labels not found in choices: {it.contrast_labels!r}")
            if it.expected_effect == "shift":
                if it.base.expected_label != L0 or it.cf.expected_label != L1:
                    raise ValueError(
                        f"{it.item_id}: shift item must satisfy base.expected_label==contrast_labels[0] "
                        f"and cf.expected_label==contrast_labels[1] "
                        f"(got base={it.base.expected_label!r} cf={it.cf.expected_label!r} contrast={it.contrast_labels!r})"
                    )
            elif it.expected_effect in {"invariant", "graded"}:
                if it.base.expected_label != it.cf.expected_label:
                    raise ValueError(
                        f"{it.item_id}: {it.expected_effect} item must satisfy base.expected_label==cf.expected_label "
                        f"(got base={it.base.expected_label!r} cf={it.cf.expected_label!r})"
                    )
                if it.base.expected_label != L0:
                    raise ValueError(
                        f"{it.item_id}: {it.expected_effect} item must satisfy base.expected_label==contrast_labels[0] "
                        f"(got base={it.base.expected_label!r} contrast={it.contrast_labels!r})"
                    )

        _validate_choices(
            choices=it.choices,
            expected_labels=(it.base.expected_label, it.cf.expected_label),
            prompts=(it.base.prompt, it.cf.prompt),
            item_id=str(it.item_id),
            field="cf.choices",
            require_equal_counts=require_equal_choice_counts,
        )


def validate_coherence_items(items: Sequence[CoherenceItem]) -> None:
    for it in items:
        if not it.valid_continuations:
            raise ValueError(f"{it.item_id}: empty valid_continuations")
        if not it.invalid_continuations:
            raise ValueError(f"{it.item_id}: empty invalid_continuations")
        for cont in list(it.valid_continuations) + list(it.invalid_continuations):
            _validate_prompt_cont_boundary(prompt=it.context, continuation=cont, item_id=str(it.item_id), field="coh")
