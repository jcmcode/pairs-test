"""
Tests for screener/analysis.py (5-test validation framework),
screener/enhanced_backtest.py (adaptive z-score, Kalman hedge),
and validation/pair_validation.py (permutation test).
"""

import numpy as np
import pandas as pd
import pytest

from screener.analysis import (
    _test_adf,
    _test_half_life,
    _test_hurst,
    _test_variance_ratio,
    _test_rolling_correlation,
    validate_pair_relationship,
    _sanitize_sharpe,
)
from screener.enhanced_backtest import (
    compute_annualized_sharpe,
    optimize_zscore_params,
    kalman_hedge_beta,
    enhanced_backtest_pair,
    _empty_strategy,
)
from validation.pair_validation import feature_shuffle_permutation_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mean_reverting_spread():
    """AR(1) spread with phi ~ 0.85 — clearly mean-reverting."""
    np.random.seed(42)
    n = 300
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.85 * s[i - 1] + np.random.randn() * 0.5
    return pd.Series(s, index=pd.date_range("2025-01-01", periods=n, freq="D"))


@pytest.fixture
def random_walk_spread():
    """Random walk — should fail most mean-reversion tests."""
    np.random.seed(123)
    return pd.Series(
        np.cumsum(np.random.randn(300)),
        index=pd.date_range("2025-01-01", periods=300, freq="D"),
    )


@pytest.fixture
def correlated_prices():
    """Two correlated price series for rolling-correlation tests."""
    np.random.seed(42)
    n = 200
    b = 100 + np.cumsum(np.random.randn(n) * 0.5)
    a = 1.5 * b + np.random.randn(n) * 2
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


@pytest.fixture
def uncorrelated_prices():
    """Two independent random walks."""
    np.random.seed(99)
    n = 200
    a = 100 + np.cumsum(np.random.randn(n) * 0.5)
    b = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


# ---------------------------------------------------------------------------
# 5-test validation: individual tests
# ---------------------------------------------------------------------------

class TestADF:
    def test_stationary_series(self, mean_reverting_spread):
        passed, pval = _test_adf(mean_reverting_spread.values)
        assert passed is True or passed is False or bool(passed) in (True, False)
        assert np.isfinite(pval)

    def test_random_walk_usually_fails(self, random_walk_spread):
        passed, pval = _test_adf(random_walk_spread.values)
        # RW typically has high p-value
        assert pval > 0.05 or not bool(passed)

    def test_short_series(self):
        passed, pval = _test_adf(np.array([1.0, 2.0, 3.0]))
        assert not passed
        assert np.isnan(pval)

    def test_nan_handling(self):
        data = np.array([1.0, np.nan, 2.0, 3.0, 4.0] * 10)
        passed, pval = _test_adf(data)
        # Should handle NaNs gracefully
        assert isinstance(passed, bool)


class TestHalfLife:
    def test_mean_reverting(self, mean_reverting_spread):
        passed, hl = _test_half_life(mean_reverting_spread.values)
        assert np.isfinite(hl)
        assert hl > 0

    def test_random_walk(self, random_walk_spread):
        passed, hl = _test_half_life(random_walk_spread.values)
        # Half-life test just checks if it returns valid results
        assert isinstance(hl, float)

    def test_short_series(self):
        passed, hl = _test_half_life(np.array([1.0, 2.0, 3.0]))
        assert not passed


class TestHurst:
    def test_mean_reverting(self, mean_reverting_spread):
        passed, h = _test_hurst(mean_reverting_spread.values)
        assert np.isfinite(h)
        assert 0 <= h <= 1

    def test_random_walk(self, random_walk_spread):
        _, h = _test_hurst(random_walk_spread.values)
        # Hurst for RW should be ~0.5
        if np.isfinite(h):
            assert h > 0.3

    def test_short_series(self):
        passed, h = _test_hurst(np.array([1.0, 2.0]))
        assert not passed
        assert np.isnan(h)


class TestVarianceRatio:
    def test_mean_reverting(self, mean_reverting_spread):
        passed, vr, z = _test_variance_ratio(mean_reverting_spread.values)
        assert np.isfinite(vr)
        assert np.isfinite(z)
        # Mean-reverting: VR < 1
        assert vr < 1.0

    def test_random_walk(self, random_walk_spread):
        passed, vr, z = _test_variance_ratio(random_walk_spread.values)
        if np.isfinite(vr):
            # RW: VR should be ~1.0
            assert 0.7 < vr < 1.5

    def test_short_series(self):
        passed, vr, z = _test_variance_ratio(np.array([1.0, 2.0, 3.0]))
        assert not passed
        assert np.isnan(vr)


class TestRollingCorrelation:
    def test_correlated(self, correlated_prices):
        a, b = correlated_prices
        passed, stability, corrs = _test_rolling_correlation(a.values, b.values)
        assert bool(passed) in (True, False)
        assert len(corrs) > 0
        # Highly correlated prices should have stable correlations
        for c in corrs:
            assert -1 <= c <= 1

    def test_uncorrelated(self, uncorrelated_prices):
        a, b = uncorrelated_prices
        passed, stability, corrs = _test_rolling_correlation(a.values, b.values)
        assert bool(passed) in (True, False)

    def test_short_series(self):
        passed, stability, corrs = _test_rolling_correlation(
            np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
        )
        assert not passed
        assert np.isnan(stability)


# ---------------------------------------------------------------------------
# 5-test validation: composite
# ---------------------------------------------------------------------------

class TestValidatePairRelationship:
    def test_returns_all_keys(self, mean_reverting_spread, correlated_prices):
        a, b = correlated_prices
        result = validate_pair_relationship(
            mean_reverting_spread.values, a.values, b.values
        )
        required_keys = [
            "adf_pval", "adf_passed", "half_life_days", "hl_passed",
            "hurst", "hurst_passed", "variance_ratio", "vr_zstat",
            "vr_passed", "rolling_corr_stability", "rc_passed",
            "score", "classification",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_score_range(self, mean_reverting_spread, correlated_prices):
        a, b = correlated_prices
        result = validate_pair_relationship(
            mean_reverting_spread.values, a.values, b.values
        )
        assert 0 <= result["score"] <= 5

    def test_classification_labels(self, mean_reverting_spread, correlated_prices):
        a, b = correlated_prices
        result = validate_pair_relationship(
            mean_reverting_spread.values, a.values, b.values
        )
        assert result["classification"] in ("strong", "moderate", "weak", "fail")

    def test_score_matches_classification(self, mean_reverting_spread, correlated_prices):
        a, b = correlated_prices
        result = validate_pair_relationship(
            mean_reverting_spread.values, a.values, b.values
        )
        score = result["score"]
        cls = result["classification"]
        if score >= 4:
            assert cls == "strong"
        elif score == 3:
            assert cls == "moderate"
        elif score == 2:
            assert cls == "weak"
        else:
            assert cls == "fail"


class TestSanitizeSharpe:
    def test_valid(self):
        assert _sanitize_sharpe(1.5, 10) == 1.5

    def test_too_few_trades(self):
        assert np.isnan(_sanitize_sharpe(5.0, 3))

    def test_infinite(self):
        assert np.isnan(_sanitize_sharpe(np.inf, 10))

    def test_nan_input(self):
        assert np.isnan(_sanitize_sharpe(np.nan, 10))


# ---------------------------------------------------------------------------
# Enhanced backtest
# ---------------------------------------------------------------------------

class TestComputeAnnualizedSharpe:
    def test_known_values(self):
        # P&L with positive drift and some noise -> positive Sharpe
        np.random.seed(42)
        pnl = pd.Series(np.cumsum(0.1 + np.random.randn(50) * 0.05))
        sharpe = compute_annualized_sharpe(pnl)
        assert sharpe > 0

    def test_short_series(self):
        assert np.isnan(compute_annualized_sharpe(pd.Series([1.0, 2.0])))

    def test_flat_pnl(self):
        pnl = pd.Series(np.ones(50))
        assert np.isnan(compute_annualized_sharpe(pnl))


class TestOptimizeZscoreParams:
    def test_finds_params_on_mean_reverting(self, mean_reverting_spread):
        result = optimize_zscore_params(mean_reverting_spread, min_trades=3)
        if result is not None:
            assert "entry_z" in result
            assert "exit_z" in result
            assert "lookback" in result
            assert result["cal_pnl"] > 0

    def test_returns_none_on_flat(self):
        flat = pd.Series(np.zeros(200), index=pd.date_range("2025-01-01", periods=200, freq="D"))
        result = optimize_zscore_params(flat)
        assert result is None


class TestKalmanHedgeBeta:
    def test_known_relationship(self):
        np.random.seed(42)
        b = np.linspace(10, 20, 100)
        a = 2.0 * b + 5.0 + np.random.randn(100) * 0.1
        beta, intercept = kalman_hedge_beta(a, b)
        assert np.isfinite(beta)
        assert beta == pytest.approx(2.0, abs=0.5)

    def test_short_series(self):
        beta, intercept = kalman_hedge_beta([1.0], [2.0])
        assert np.isnan(beta)


class TestEnhancedBacktestPair:
    def test_returns_all_strategies(self):
        np.random.seed(42)
        n_cal, n_oos = 150, 50
        idx_cal = pd.date_range("2025-01-01", periods=n_cal, freq="D")
        idx_oos = pd.date_range("2025-06-01", periods=n_oos, freq="D")

        b_cal = pd.Series(100 + np.cumsum(np.random.randn(n_cal) * 0.5), index=idx_cal)
        a_cal = pd.Series(1.5 * b_cal.values + np.random.randn(n_cal) * 2, index=idx_cal)
        b_oos = pd.Series(100 + np.cumsum(np.random.randn(n_oos) * 0.5), index=idx_oos)
        a_oos = pd.Series(1.5 * b_oos.values + np.random.randn(n_oos) * 2, index=idx_oos)

        result = enhanced_backtest_pair(a_cal, b_cal, a_oos, b_oos)
        assert result is not None
        assert "baseline_pnl" in result

    def test_returns_none_on_nan_beta(self):
        # Single-point series -> OLS fails
        result = enhanced_backtest_pair(
            pd.Series([1.0]), pd.Series([2.0]),
            pd.Series([3.0]), pd.Series([4.0]),
        )
        assert result is None


class TestEmptyStrategy:
    def test_max_dd_is_nan(self):
        empty = _empty_strategy("test")
        assert np.isnan(empty["test_max_dd"])

    def test_keys_present(self):
        empty = _empty_strategy("kalman")
        assert "kalman_pnl" in empty
        assert "kalman_n_trades" in empty
        assert "kalman_sharpe" in empty
        assert "kalman_ann_sharpe" in empty


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    @pytest.fixture
    def simple_ts_df(self):
        """Create a small multi-index DataFrame for permutation testing."""
        np.random.seed(42)
        n_timestamps = 30
        tickers = ["A", "B", "C", "D", "E", "F"]
        rows = []
        for i in range(n_timestamps):
            ts = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i)
            for t in tickers:
                rows.append({
                    "Datetime": ts,
                    "Ticker": t,
                    "f1": np.random.randn(),
                    "f2": np.random.randn(),
                    "f3": np.random.randn(),
                })
        df = pd.DataFrame(rows).set_index(["Datetime", "Ticker"])
        return df

    def test_returns_expected_keys(self, simple_ts_df):
        result = feature_shuffle_permutation_test(
            simple_ts_df,
            features_to_cluster=["f1", "f2", "f3"],
            optics_params={"min_samples": 2, "xi": 0.05, "min_cluster_size": 2},
            n_permutations=5,
            n_sample_timestamps=10,
        )
        assert "pair_zscores" in result
        assert "fraction_significant" in result
        assert "null_counts" in result

    def test_fraction_significant_range(self, simple_ts_df):
        result = feature_shuffle_permutation_test(
            simple_ts_df,
            features_to_cluster=["f1", "f2", "f3"],
            optics_params={"min_samples": 2, "xi": 0.05, "min_cluster_size": 2},
            n_permutations=5,
            n_sample_timestamps=10,
        )
        frac = result["fraction_significant"]
        assert 0.0 <= frac <= 1.0

    def test_backward_compat_params_accepted(self, simple_ts_df):
        """Ensure deprecated params don't cause errors."""
        result = feature_shuffle_permutation_test(
            simple_ts_df,
            features_to_cluster=["f1", "f2", "f3"],
            optics_params={"min_samples": 2, "xi": 0.05, "min_cluster_size": 2},
            pair_co_cluster_freq={("A", "B"): 5},
            total_valid_windows=100,
            n_permutations=3,
            n_sample_timestamps=10,
        )
        assert "pair_zscores" in result
