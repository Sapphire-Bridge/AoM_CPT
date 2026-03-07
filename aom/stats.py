from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricValue:
    """
    Shared representation for metric values that may be missing/invalid.

    Policy: missing/invalid measurements must not silently become plausible numbers.
    Use `NaN` + `valid=False` (and an optional reason) instead.
    """

    value: float
    n: int
    valid: bool
    reason: str | None = None


def bootstrap_ci_metric(
    values: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[MetricValue, float, float]:
    """
    Percentile bootstrap CI for the mean with explicit validity semantics.

    Returns (metric, ci_low, ci_high). On empty input, returns NaNs and `valid=False`.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")

    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        nan = float("nan")
        return MetricValue(value=nan, n=0, valid=False, reason="empty sample"), nan, nan
    if not np.isfinite(arr).all():
        nan = float("nan")
        return MetricValue(value=nan, n=n, valid=False, reason="non-finite sample"), nan, nan

    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, n, size=(int(n_bootstrap), n))
    means = arr[idx].mean(axis=1)

    alpha = 1.0 - float(ci)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return MetricValue(value=float(arr.mean()), n=n, valid=True, reason=None), lo, hi


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI for the mean.

    Returns (mean, ci_low, ci_high). For empty input, returns NaNs.
    """
    mv, lo, hi = bootstrap_ci_metric(values, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    return float(mv.value), float(lo), float(hi)


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cohen's d for independent samples (difference in means / pooled std).

    Returns NaN when either sample has <2 finite values or when pooled variance is 0.
    """
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    xa = xa[np.isfinite(xa)]
    ya = ya[np.isfinite(ya)]
    if xa.size < 2 or ya.size < 2:
        return float("nan")
    mx = float(xa.mean())
    my = float(ya.mean())
    vx = float(xa.var(ddof=1))
    vy = float(ya.var(ddof=1))
    pooled = ((xa.size - 1) * vx + (ya.size - 1) * vy) / max(1.0, float(xa.size + ya.size - 2))
    if pooled <= 0.0 or not np.isfinite(pooled):
        return float("nan")
    return float((mx - my) / float(np.sqrt(pooled)))
