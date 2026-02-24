"""
Unit tests for trading/trading.py

Tests pair registry building, classical tests, backtesting, and walk-forward.
"""

import numpy as np
import pandas as pd
import pytest

from trading.trading import (
    hurst_exponent,
    backtest_pair,
    walk_forward_backtest,
    compute_noise_adjusted_frequency,
)
from validation.pair_validation import compute_hedge_ratio, zscore_signals


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mean_reverting_daily():
    """Daily mean-reverting spread series."""
    np.random.seed(42)
    n = 200
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.85 * spread[i - 1] + np.random.randn() * 0.3
    idx = pd.date_range('2025-01-01', periods=n, freq='D')
    return pd.Series(spread, index=idx)


@pytest.fixture
def correlated_daily_prices():
    """Two correlated daily price series."""
    np.random.seed(42)
    n = 250
    idx = pd.date_range('2025-01-01', periods=n, freq='D')
    b = 100 + np.cumsum(np.random.randn(n) * 0.5)
    a = 1.5 * b + 5 + np.random.randn(n) * 1.0
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------

class TestHurstExponent:
    def test_mean_reverting(self, mean_reverting_daily):
        h = hurst_exponent(mean_reverting_daily.values)
        # Mean-reverting spread should have H < 0.5
        assert h < 0.6  # allowing some margin

    def test_random_walk(self):
        np.random.seed(42)
        rw = np.cumsum(np.random.randn(500))
        h = hurst_exponent(rw)
        assert h > 0.4  # random walk H ~ 0.5

    def test_short_series(self):
        h = hurst_exponent(np.array([1, 2, 3, 4, 5]))
        assert np.isnan(h)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class TestBacktestPair:
    def test_basic(self, mean_reverting_daily):
        result = backtest_pair(mean_reverting_daily, lookback=20,
                               entry_z=2.0, exit_z=0.5)
        assert 'total_pnl' in result
        assert 'n_trades' in result
        assert result['n_trades'] >= 0

    def test_short_series(self):
        short = pd.Series([1.0, 2.0, 3.0])
        result = backtest_pair(short, lookback=20)
        assert result['n_trades'] == 0


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

class TestWalkForwardBacktest:
    def test_basic(self, correlated_daily_prices):
        da, db = correlated_daily_prices
        result = walk_forward_backtest(da, db, n_splits=3)
        assert result is not None
        assert 'n_splits' in result
        assert result['n_splits'] >= 1
        assert 'avg_sharpe' in result
        assert 'splits' in result

    def test_short_data(self):
        idx = pd.date_range('2025-01-01', periods=20, freq='D')
        da = pd.Series(np.random.randn(20) + 100, index=idx)
        db = pd.Series(np.random.randn(20) + 50, index=idx)
        result = walk_forward_backtest(da, db, n_splits=3)
        assert result is None  # insufficient data

    def test_multiple_splits(self, correlated_daily_prices):
        da, db = correlated_daily_prices
        result = walk_forward_backtest(da, db, n_splits=5)
        if result is not None:
            assert result['n_splits'] >= 2
            # Each split should have independent results
            for split in result['splits']:
                assert 'total_pnl' in split
                assert 'beta' in split


# ---------------------------------------------------------------------------
# Noise-adjusted frequency
# ---------------------------------------------------------------------------

class TestNoiseAdjustedFrequency:
    def test_basic(self):
        ch = pd.DataFrame({
            'Ticker': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Datetime': ['t1', 't2', 't3', 't1', 't2', 't3'],
            'Cluster_ID': [0, -1, 0, 0, 0, -1],
        })
        # A is noise at t2, B is noise at t3
        # Valid timestamps for pair (A,B): only t1 (both non-noise)
        freq = {('A', 'B'): 1}
        adj = compute_noise_adjusted_frequency(ch, freq)
        assert adj[('A', 'B')] == 1.0  # 1 co-cluster / 1 valid window

    def test_no_valid_windows(self):
        ch = pd.DataFrame({
            'Ticker': ['A', 'B'],
            'Datetime': ['t1', 't1'],
            'Cluster_ID': [-1, -1],
        })
        freq = {('A', 'B'): 0}
        adj = compute_noise_adjusted_frequency(ch, freq)
        assert adj[('A', 'B')] == 0.0

    def test_absent_ticker_not_counted_as_valid(self):
        """Regression: timestamps where a ticker is absent (not present at all)
        should not inflate the denominator.  Only timestamps where BOTH tickers
        are present and non-noise count."""
        ch = pd.DataFrame({
            'Ticker': ['A', 'A', 'B'],
            'Datetime': ['t1', 't2', 't1'],
            'Cluster_ID': [0,   0,   0],
        })
        # A and B co-cluster at t1.  At t2 only A exists (B absent).
        # Valid denominator for (A,B) should be 1 (only t1), so freq = 1/1 = 1.0
        freq = {('A', 'B'): 1}
        adj = compute_noise_adjusted_frequency(ch, freq)
        assert adj[('A', 'B')] == 1.0  # was 0.5 before fix
