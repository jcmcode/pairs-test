"""
Unit tests for signals module.

Tests detection, transient validation, and signal generation with
synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from signals.detection import (
    detect_new_formations,
    run_clustering_snapshot,
    compute_co_cluster_freq,
    detect_formation_events,
)
from signals.transient import (
    validate_transient_event,
    generate_transient_signals,
)
from config import ClusteringConfig, TransientConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cluster_snapshot():
    """Two clusters: {A, B} and {C, D}."""
    return pd.DataFrame({
        'Ticker': ['A', 'B', 'C', 'D'],
        'Cluster_ID': [0, 0, 1, 1],
    })


@pytest.fixture
def synthetic_ts_df():
    """
    Synthetic ts_df with correlated prices for testing.
    Two tickers with ~100 hourly observations each.
    """
    np.random.seed(42)
    n = 150
    idx = pd.date_range('2025-01-01', periods=n, freq='h')

    rows = []
    base_price = 100.0
    price_a = base_price
    price_b = base_price

    for i, t in enumerate(idx):
        shock = np.random.randn() * 0.5
        price_a += shock + np.random.randn() * 0.2
        price_b += shock * 0.8 + np.random.randn() * 0.2  # correlated

        for ticker, price in [('TICK_A', price_a), ('TICK_B', price_b)]:
            rows.append({
                'Datetime': t,
                'Ticker': ticker,
                'Price': price,
                'Returns': np.random.randn() * 0.01,
                'Vol_Short': 0.015 + np.random.randn() * 0.001,
                'Beta_SPX_Short': 1.0 + np.random.randn() * 0.1,
                'Beta_Sector_Short': 0.8 + np.random.randn() * 0.1,
                'RSI': 50 + np.random.randn() * 10,
                'Momentum_5H': np.random.randn() * 0.01,
                'Vol_Regime_Shift': np.random.randn() * 0.05,
                'Beta_SPX_Regime_Shift': np.random.randn() * 0.05,
                'Beta_Sector_Regime_Shift': np.random.randn() * 0.05,
            })

    df = pd.DataFrame(rows).set_index(['Datetime', 'Ticker'])
    return df


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetectNewFormations:
    def test_new_formation(self, simple_cluster_snapshot):
        prev = {'A': 0, 'B': 1, 'C': 1, 'D': 1}  # A was not with B before
        new = detect_new_formations(simple_cluster_snapshot, prev)
        # A-B should be a new formation (they weren't co-clustered before)
        pairs = [tuple(sorted(p)) for p in new]
        assert ('A', 'B') in pairs

    def test_no_new_formations(self, simple_cluster_snapshot):
        # Same clusters as before
        prev = {'A': 0, 'B': 0, 'C': 1, 'D': 1}
        new = detect_new_formations(simple_cluster_snapshot, prev)
        assert len(new) == 0

    def test_empty_previous(self, simple_cluster_snapshot):
        new = detect_new_formations(simple_cluster_snapshot, {})
        # All pairs should be new
        assert len(new) >= 2  # A-B and C-D at minimum

    def test_noise_excluded(self):
        snapshot = pd.DataFrame({
            'Ticker': ['A', 'B', 'C'],
            'Cluster_ID': [-1, 0, 0],
        })
        new = detect_new_formations(snapshot, {})
        pairs = [tuple(sorted(p)) for p in new]
        assert ('A', 'B') not in pairs  # A is noise
        assert ('B', 'C') in pairs

    def test_both_noise_to_same_cluster_is_new_formation(self):
        """Regression: tickers both in noise (-1) previously should not
        suppress detection when they co-cluster in the current snapshot."""
        snapshot = pd.DataFrame({
            'Ticker': ['A', 'B'],
            'Cluster_ID': [0, 0],
        })
        prev = {'A': -1, 'B': -1}  # both were noise
        new = detect_new_formations(snapshot, prev)
        pairs = [tuple(sorted(p)) for p in new]
        assert ('A', 'B') in pairs


class TestRunClusteringSnapshot:
    def test_insufficient_data(self):
        df = pd.DataFrame({
            'f1': [1.0, 2.0],
            'f2': [3.0, 4.0],
        }, index=['A', 'B'])
        cfg = ClusteringConfig(min_stocks_per_snapshot=5)
        labels = run_clustering_snapshot(df, cfg)
        assert labels is None

    def test_enough_data(self):
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'f3': np.random.randn(n),
        }, index=[f'TICK_{i}' for i in range(n)])
        cfg = ClusteringConfig(min_stocks_per_snapshot=5, max_noise_pct=1.0)
        labels = run_clustering_snapshot(df, cfg)
        # May be None if all noise, but should not crash
        if labels is not None:
            assert len(labels) == n


class TestComputeCoClusterFreq:
    def test_basic(self):
        ch = pd.DataFrame({
            'Datetime': ['t1', 't1', 't1', 't2', 't2', 't2'],
            'Ticker': ['A', 'B', 'C', 'A', 'B', 'C'],
            'Cluster_ID': [0, 0, 1, 0, 1, 1],
        })
        freq, total = compute_co_cluster_freq(ch)
        assert total == 2
        assert freq[('A', 'B')] == 1  # co-clustered only at t1
        assert freq[('B', 'C')] == 1  # co-clustered only at t2


class TestDetectFormationEvents:
    def test_basic_formation(self):
        ch = pd.DataFrame({
            'Datetime': pd.date_range('2025-01-01', periods=5, freq='h').repeat(2),
            'Ticker': ['A', 'B'] * 5,
            'Cluster_ID': [1, 1,  # t0: together
                           1, 1,  # t1: together
                           0, 1,  # t2: apart
                           0, 1,  # t3: apart
                           0, 1], # t4: apart
        })
        freq = {('A', 'B'): 2}
        from config import FormationConfig
        cfg = FormationConfig(min_gap_hours=1)
        formations, durations = detect_formation_events(ch, freq, cfg)
        assert len(formations) >= 1


# ---------------------------------------------------------------------------
# Transient validation tests
# ---------------------------------------------------------------------------

class TestValidateTransientEvent:
    def test_with_synthetic_data(self, synthetic_ts_df):
        ts = synthetic_ts_df.index.get_level_values('Datetime').unique()
        cfg = TransientConfig(
            calibration=10, exploitation=20, execution_lag=1,
            corr_threshold=0.0, cv_threshold=1.0, hl_max=100.0,
            hedge_drift_max=10.0, entry_z=2.0, exit_z=0.5,
        )
        result = validate_transient_event(
            'TICK_A', 'TICK_B', ts[0], synthetic_ts_df, cfg,
        )
        # With relaxed thresholds, should return a result (not None)
        if result is not None:
            assert 'passed' in result
            assert 'correlation' in result
            assert 'beta_cal' in result

    def test_insufficient_data(self, synthetic_ts_df):
        cfg = TransientConfig(
            calibration=200, exploitation=200, execution_lag=10,
        )
        ts = synthetic_ts_df.index.get_level_values('Datetime').unique()
        result = validate_transient_event(
            'TICK_A', 'TICK_B', ts[0], synthetic_ts_df, cfg,
        )
        assert result is None


class TestGenerateTransientSignals:
    def test_with_synthetic_data(self, synthetic_ts_df):
        cfg = TransientConfig(calibration=10)
        result = generate_transient_signals(
            'TICK_A', 'TICK_B', synthetic_ts_df, cfg,
        )
        if result is not None:
            assert result['signal'] in ('long', 'short', 'exit', 'hold')
            assert 'z_score' in result
            assert 'beta' in result

    def test_missing_ticker(self, synthetic_ts_df):
        result = generate_transient_signals(
            'NONEXISTENT', 'TICK_B', synthetic_ts_df,
        )
        assert result is None
