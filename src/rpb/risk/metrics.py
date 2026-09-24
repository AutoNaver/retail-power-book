"""Risk metrics on simulated gross margin.

Margins are in EUR, one sample per simulation path. Low margin is bad, so the
tail metrics look at the lower tail and are reported as margin levels in EUR,
without flipping the sign into a "loss".
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TailRisk:
    level: float
    var_eur: float  # (1 - level) quantile of margin
    es_eur: float  # mean margin at or below var_eur


@dataclass(frozen=True)
class MarginSummary:
    mean_eur: float
    std_eur: float  # sample standard deviation (ddof=1)
    tails: tuple[TailRisk, ...]

    def tail(self, level: float) -> TailRisk:
        for t in self.tails:
            if math.isclose(t.level, level):
                return t
        raise KeyError(f"no tail metrics at level {level}")


def lower_tail(margin_eur: np.ndarray, level: float) -> TailRisk:
    """Value-at-risk and expected shortfall of margin at confidence `level`.

    VaR is the empirical (1 - level) quantile: the k-th smallest sample with
    k = ceil(n * (1 - level)). ES is the mean of all samples at or below it.
    """
    x = _validated(margin_eur)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    ordered = np.sort(x)
    # The small tolerance stops float noise (1 - 0.95 = 0.0500...04) from moving k up by one.
    k = max(1, math.ceil(x.size * (1.0 - level) - 1e-9))
    var = ordered[k - 1]
    return TailRisk(level=level, var_eur=float(var), es_eur=float(x[x <= var].mean()))


def margin_summary(margin_eur: np.ndarray, levels: Sequence[float] = (0.95, 0.99)) -> MarginSummary:
    """Mean, standard deviation, and lower-tail VaR/ES of margin at each level."""
    x = _validated(margin_eur)
    return MarginSummary(
        mean_eur=float(x.mean()),
        std_eur=float(x.std(ddof=1)),
        tails=tuple(lower_tail(x, level) for level in levels),
    )


def _validated(margin_eur: np.ndarray) -> np.ndarray:
    x = np.asarray(margin_eur, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"margin must be 1-D (one value per path), got shape {x.shape}")
    if x.size < 2:
        raise ValueError("need at least two margin samples")
    if not np.isfinite(x).all():
        raise ValueError("margin contains NaN or infinite values")
    return x
