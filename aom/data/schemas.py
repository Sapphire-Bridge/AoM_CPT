from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class PromptSide:
    prompt: str
    expected_label: str


@dataclass(frozen=True)
class DisambPair:
    """
    Minimal pair for lexical/structural disambiguation.

    `choices` maps labels -> list of continuation strings to score as next tokens/phrases.
    """
    pair_id: str
    target: str
    target_occurrence: int
    a: PromptSide
    b: PromptSide
    choices: Mapping[str, List[str]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CounterfactualPair:
    """
    Minimal-pair intervention sensitivity item.

    `base` and `cf` share `choices` but (optionally) differ in expected label.
    """
    item_id: str
    base: PromptSide
    cf: PromptSide
    choices: Mapping[str, List[str]]
    intervention_type: str
    # Optional explicit contrast labels used for AoM-CF shift calculations.
    # Stored in JSONL as a 2-list; interpreted as (base_label, cf_label).
    contrast_labels: Optional[Tuple[str, str]] = None
    # Expected effect of the intervention on the contrast:
    #  - "shift": expected label flips between base and cf
    #  - "invariant": meaning-preserving sham control (no flip)
    #  - "graded": meaning-relevant partial intervention (no flip; expect intermediate shift magnitude)
    expected_effect: str = "shift"
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CoherenceItem:
    item_id: str
    context: str
    valid_continuations: List[str]
    invalid_continuations: List[str]
    constraint_type: str
    # Optional grouping for controls (e.g., "main", "ablate_relevant", "ablate_irrelevant").
    group: str = "main"
    metadata: Optional[Dict[str, Any]] = None
