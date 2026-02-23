"""
Unit tests for validation/pair_validation.py

Tests all core statistical functions with synthetic data where
expected results are analytically known.
"""

import numpy as np
import pandas as pd
import pytest

from validation.pair_validation import (
    make_clusters_wide,
    cluster_persistence,
    compute_spread,
    lag1_autocorr,
    ar1_phi,
    bounce_rate,
    validate_pair,
    compute_hedge_ratio,
    hedge_ratio_drift,
    spread_cv_normalized,
    half_life,
    zscore_signals,
    simulate_spread_pnl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cluster_history():
    """Simple cluster history with two tickers."""
    rows = []
    for i in range(10):
        ts = pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i)
        # A and B in same cluster for first 7 timestamps, different for last 3
        rows.append({'Datetime': ts, 'Ticker': 'A', 'Cluster_ID': 0})
        if i < 7:
            rows.append({'Datetime': ts, 'Ticker': 'B', 'Cluster_ID': 0})
        else:
            rows.append({'Datetime': ts, 'Ticker': 'B', 'Cluster_ID': 1})
    return pd.DataFrame(rows)


@pytest.fixture
def prices():
    """Correlated price series for two tickers."""
    np.random.seed(42)
    n = 100
    idx = pd.date_range('2025-01-01', periods=n, freq='h')
    b = 100 + np.cumsum(np.random.randn(n) * 0.5)
    # A = 1.5 * B + noise
    a = 1.5 * b + np.random.randn(n) * 2
    return pd.DataFrame({'A': a, 'B': b}, index=idx)


@pytest.fixture
def mean_reverting_spread():
    """AR(1) spread with phi ~ 0.8 (mean-reverting)."""
    np.random.seed(42)
    n = 200
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.8 * spread[i - 1] + np.random.randn() * 0.5
    idx = pd.date_range('2025-01-01', periods=n, freq='h')
    return pd.Series(spread, index=idx)


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

class TestMakeClustersWide:
    def test_basic_pivot(self, cluster_history):
        wide = make_clusters_wide(cluster_history)
        assert 'A' in wide.columns
        assert 'B' in wide.columns
        assert len(wide) == 10

    def test_values(self, cluster_history):
        wide = make_clusters_wide(cluster_history)
        # First timestamp: both in cluster 0
        assert wide.iloc[0]['A'] == 0
        assert wide.iloc[0]['B'] == 0
        # Last timestamp: A in 0, B in 1
        assert wide.iloc[-1]['A'] == 0
        assert wide.iloc[-1]['B'] == 1


class TestClusterPersistence:
    def test_full_persistence(self, cluster_history):
        wide = make_clusters_wide(cluster_history)
        # At t=4 (5th timestamp), last 5 timestamps all same cluster
        t = wide.index[4]
        assert cluster_persistence(wide, 'A', 'B', t, H=5) == 1.0

    def test_partial_persistence(self, cluster_history):
        wide = make_clusters_wide(cluster_history)
        # At t=9 (last), last 5 timestamps: 3 different, 2 same... let's check
        t = wide.index[9]
        persist = cluster_persistence(wide, 'A', 'B', t, H=5)
        # timestamps 5,6 same; 7,8,9 different => 2/5 = 0.4
        assert persist == pytest.approx(0.4, abs=0.01)

    def test_invalid_timestamp(self, cluster_history):
        wide = make_clusters_wide(cluster_history)
        bad_t = pd.Timestamp('2099-01-01')
        assert np.isnan(cluster_persistence(wide, 'A', 'B', bad_t))


# ---------------------------------------------------------------------------
# Spread + tests
# ---------------------------------------------------------------------------

class TestComputeSpread:
    def test_basic(self, prices):
        s = compute_spread(prices, 'A', 'B', beta=1.5)
        assert len(s) > 0
        # spread should be roughly: A - 1.5*B = noise ~ N(0, 2)
        assert abs(s.mean()) < 10

    def test_beta_one(self, prices):
        s = compute_spread(prices, 'A', 'B', beta=1.0)
        # A = 1.5*B + noise, so A - 1*B = 0.5*B + noise, should be large
        assert abs(s.mean()) > 10


class TestLag1Autocorr:
    def test_white_noise(self):
        np.random.seed(42)
        x = pd.Series(np.random.randn(500))
        ac = lag1_autocorr(x)
        assert abs(ac) < 0.15  # should be near zero

    def test_highly_autocorrelated(self):
        np.random.seed(42)
        x = np.cumsum(np.random.randn(500))
        ac = lag1_autocorr(pd.Series(x))
        assert ac > 0.9

    def test_short_series(self):
        assert np.isnan(lag1_autocorr(pd.Series([1.0, 2.0])))


class TestAr1Phi:
    def test_known_phi(self):
        np.random.seed(42)
        n = 1000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + np.random.randn() * 0.3
        phi = ar1_phi(pd.Series(x))
        assert phi == pytest.approx(0.7, abs=0.05)

    def test_random_walk(self):
        np.random.seed(42)
        x = np.cumsum(np.random.randn(500))
        phi = ar1_phi(pd.Series(x))
        assert phi > 0.95

    def test_short_series(self):
        assert np.isnan(ar1_phi(pd.Series([1.0, 2.0, 3.0])))


class TestBounceRate:
    def test_mean_reverting(self, mean_reverting_spread):
        br = bounce_rate(mean_reverting_spread, W=20, z_entry=1.5, horizon=3, improve=0.20)
        assert br is not np.nan
        assert 0.0 <= br <= 1.0

    def test_short_series(self):
        short = pd.Series(np.random.randn(5))
        assert np.isnan(bounce_rate(short, W=10, z_entry=2.0))

    def test_no_events(self):
        # Very tight entry threshold, should find no events
        x = pd.Series(np.zeros(100))
        assert np.isnan(bounce_rate(x, W=10, z_entry=100.0))


# ---------------------------------------------------------------------------
# Hedge ratio estimation
# ---------------------------------------------------------------------------

class TestComputeHedgeRatio:
    def test_ols_known_beta(self):
        np.random.seed(42)
        b = np.linspace(10, 20, 100)
        a = 2.0 * b + 5.0 + np.random.randn(100) * 0.1
        beta, intercept, r_sq = compute_hedge_ratio(a, b, method='ols')
        assert beta == pytest.approx(2.0, abs=0.05)
        assert intercept == pytest.approx(5.0, abs=0.5)
        assert r_sq > 0.99

    def test_tls_known_beta(self):
        np.random.seed(42)
        b = np.linspace(10, 20, 100)
        a = 2.0 * b + 5.0 + np.random.randn(100) * 0.1
        beta, intercept, r_sq = compute_hedge_ratio(a, b, method='tls')
        assert beta == pytest.approx(2.0, abs=0.1)

    def test_kalman_known_beta(self):
        np.random.seed(42)
        b = np.linspace(10, 20, 100)
        a = 2.0 * b + 5.0 + np.random.randn(100) * 0.1
        beta, intercept, r_sq = compute_hedge_ratio(a, b, method='kalman')
        assert beta == pytest.approx(2.0, abs=0.5)  # Kalman needs warmup

    def test_insufficient_data(self):
        beta, intercept, r_sq = compute_hedge_ratio([1.0], [2.0])
        assert np.isnan(beta)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            compute_hedge_ratio([1, 2, 3], [4, 5, 6], method='bad')


class TestHedgeRatioDrift:
    def test_no_drift(self):
        assert hedge_ratio_drift(1.5, 1.5) == 0.0

    def test_known_drift(self):
        drift = hedge_ratio_drift(1.0, 1.2)
        assert drift == pytest.approx(0.2, abs=0.01)

    def test_nan_input(self):
        assert np.isnan(hedge_ratio_drift(np.nan, 1.0))
        assert np.isnan(hedge_ratio_drift(1.0, np.nan))


class TestSpreadCVNormalized:
    def test_basic(self):
        spread = np.array([0.1, -0.1, 0.2, -0.2, 0.15])
        prices_a = np.array([100, 101, 102, 101, 100])
        prices_b = np.array([50, 51, 52, 51, 50])
        cv = spread_cv_normalized(spread, prices_a, prices_b)
        assert cv > 0
        assert np.isfinite(cv)

    def test_zero_prices(self):
        cv = spread_cv_normalized([1, 2], [0, 0], [0, 0])
        assert np.isnan(cv)


class TestHalfLife:
    def test_mean_reverting(self, mean_reverting_spread):
        hl = half_life(mean_reverting_spread)
        assert np.isfinite(hl)
        assert hl > 0

    def test_random_walk(self):
        np.random.seed(42)
        rw = pd.Series(np.cumsum(np.random.randn(200)))
        hl = half_life(rw)
        # Random walk may produce a finite half-life on a finite sample
        # but it should be relatively long compared to a mean-reverting process
        assert hl > 5 or hl == np.inf

    def test_short_series(self):
        assert np.isnan(half_life(pd.Series([1, 2, 3])))


# ---------------------------------------------------------------------------
# Signal generation & P&L
# ---------------------------------------------------------------------------

class TestZscoreSignals:
    def test_output_columns(self, mean_reverting_spread):
        signals = zscore_signals(mean_reverting_spread, lookback=20)
        assert 'z_score' in signals.columns
        assert 'long_entry' in signals.columns
        assert 'short_entry' in signals.columns
        assert 'exit' in signals.columns

    def test_signal_logic(self):
        # Construct a spread that clearly exceeds thresholds
        idx = pd.date_range('2025-01-01', periods=50, freq='h')
        spread = pd.Series(np.zeros(50), index=idx)
        spread.iloc[30:] = 10.0  # Big positive spike

        signals = zscore_signals(spread, lookback=20, entry_z=2.0)
        # After the spike, z-score should be very positive
        assert signals['short_entry'].any()


class TestSimulateSpreadPnl:
    def test_no_trades(self):
        idx = pd.date_range('2025-01-01', periods=50, freq='h')
        spread = pd.Series(np.zeros(50), index=idx)
        signals = zscore_signals(spread, lookback=10)
        result = simulate_spread_pnl(spread, signals)
        assert result['n_trades'] == 0
        assert result['total_pnl'] == 0.0

    def test_profitable_trade(self, mean_reverting_spread):
        signals = zscore_signals(mean_reverting_spread, lookback=20,
                                 entry_z=2.0, exit_z=0.5)
        result = simulate_spread_pnl(mean_reverting_spread, signals)
        assert result['n_trades'] >= 0
        assert 'pnl_series' in result
        assert len(result['pnl_series']) == len(signals)


# ---------------------------------------------------------------------------
# Validate pair (integration)
# ---------------------------------------------------------------------------

class TestValidatePair:
    def test_valid_pair(self, prices, cluster_history):
        # Build cluster history covering enough of the prices index
        idx = prices.index[:30]
        rows = []
        for t in idx:
            rows.append({'Datetime': t, 'Ticker': 'A', 'Cluster_ID': 0})
            rows.append({'Datetime': t, 'Ticker': 'B', 'Cluster_ID': 0})
        ch = pd.DataFrame(rows)

        # Use W_spread=10 so 30 prices is sufficient
        result = validate_pair(prices, ch, 'A', 'B', beta=1.5, t=idx[29],
                               W_spread=10)
        assert 'cluster_persist' in result
        assert 'ar1_phi' in result

    def test_missing_timestamp(self, prices, cluster_history):
        bad_t = pd.Timestamp('2099-01-01')
        result = validate_pair(prices, cluster_history, 'A', 'B', beta=1.5, t=bad_t)
        assert result == {}

    def test_noise_filtering(self, prices):
        idx = prices.index[:10]
        rows = []
        for t in idx:
            rows.append({'Datetime': t, 'Ticker': 'A', 'Cluster_ID': -1})
            rows.append({'Datetime': t, 'Ticker': 'B', 'Cluster_ID': 0})
        ch = pd.DataFrame(rows)

        result = validate_pair(prices, ch, 'A', 'B', beta=1.5, t=idx[5],
                               drop_noise_now=True)
        assert result == {}
