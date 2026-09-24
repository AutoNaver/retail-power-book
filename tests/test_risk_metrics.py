from statistics import NormalDist

import numpy as np
import pytest

from rpb.risk.metrics import lower_tail, margin_summary


@pytest.mark.parametrize("level", [0.95, 0.99])
def test_normal_margin_matches_closed_form(level: float) -> None:
    mu, sigma = 1_000_000.0, 200_000.0
    margin = np.random.default_rng(20260924).normal(mu, sigma, size=1_000_000)

    z = NormalDist().inv_cdf(level)
    expected_var = mu - sigma * z  # = mu + sigma * inv_cdf(1 - level)
    expected_es = mu - sigma * NormalDist().pdf(z) / (1.0 - level)

    tail = margin_summary(margin, levels=(level,)).tail(level)
    # Monte Carlo standard errors are below 0.005 sigma at n = 1e6.
    assert tail.var_eur == pytest.approx(expected_var, abs=0.02 * sigma)
    assert tail.es_eur == pytest.approx(expected_es, abs=0.02 * sigma)


def test_discrete_sample_exact_values() -> None:
    # 20 margins: -90, -80, ..., 100 EUR, shuffled.
    margin = np.random.default_rng(1).permutation(np.arange(1, 21) * 10.0 - 100.0)
    summary = margin_summary(margin, levels=(0.95, 0.90, 0.75))

    assert summary.mean_eur == pytest.approx(5.0)
    # Sample variance of 1..n is n(n+1)/12 = 35 for n = 20; the step is 10 EUR.
    assert summary.std_eur == pytest.approx(np.sqrt(3500.0))
    assert (summary.tail(0.95).var_eur, summary.tail(0.95).es_eur) == (-90.0, -90.0)
    assert (summary.tail(0.90).var_eur, summary.tail(0.90).es_eur) == (-80.0, -85.0)
    assert (summary.tail(0.75).var_eur, summary.tail(0.75).es_eur) == (-50.0, -70.0)


def test_es_includes_ties_at_var() -> None:
    tail = lower_tail(np.array([1.0, 1.0, 1.0, 5.0, 9.0]), level=0.8)
    assert (tail.var_eur, tail.es_eur) == (1.0, 1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="NaN"):
        margin_summary(np.array([1.0, bad, 3.0]))


def test_invalid_shape_and_level_raise() -> None:
    with pytest.raises(ValueError):
        margin_summary(np.ones((2, 3)))
    with pytest.raises(ValueError):
        lower_tail(np.arange(10.0), level=1.0)
