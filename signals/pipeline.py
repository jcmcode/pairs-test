"""
Real-time transient correlation detection pipeline.

Wires together: fetch data -> compute features -> cluster -> detect formations
-> validate -> signal. Can be run as a one-shot scan or iterated for
continuous monitoring.
"""

import numpy as np
import pandas as pd

from config import PipelineConfig, DEFAULT_CONFIG
from signals.features import fetch_data, compute_features
from signals.detection import (
    run_clustering_snapshot, detect_new_formations, compute_co_cluster_freq,
)
from signals.transient import validate_transient_event, generate_transient_signals
from signals.stable import track_stable_pairs


class TransientPipeline:
    """
    End-to-end pipeline for transient correlation detection.

    Usage
    -----
    >>> pipe = TransientPipeline()
    >>> pipe.initialize()          # fetch data, compute features, cluster
    >>> signals = pipe.scan()      # detect formations and generate signals
    >>> pipe.update()              # fetch new data and re-cluster latest timestamp
    """

    def __init__(self, tickers=None, cfg: PipelineConfig = None):
        self.cfg = cfg or DEFAULT_CONFIG
        self.tickers = tickers
        self.ts_df = None
        self.cluster_history = None
        self.pair_co_cluster_freq = None
        self._previous_clusters = {}

    def initialize(self, df_prices=None):
        """
        Full initialization: fetch data, compute features, run clustering
        across all timestamps.

        Parameters
        ----------
        df_prices : DataFrame, optional
            Pre-fetched price data. If None, fetches via yfinance.
        """
        if df_prices is None:
            df_prices = fetch_data(tickers=self.tickers)

        self.ts_df = compute_features(df_prices, cfg=self.cfg.features)

        from signals.detection import build_cluster_history
        self.cluster_history = build_cluster_history(
            self.ts_df, cfg=self.cfg.clustering, feat_cfg=self.cfg.features,
        )

        self.pair_co_cluster_freq, _ = compute_co_cluster_freq(self.cluster_history)

        # Set previous clusters to second-to-last timestamp
        timestamps = sorted(self.cluster_history['Datetime'].unique())
        if len(timestamps) >= 2:
            prev_snap = self.cluster_history[
                self.cluster_history['Datetime'] == timestamps[-2]
            ]
            self._previous_clusters = dict(
                zip(prev_snap['Ticker'], prev_snap['Cluster_ID'])
            )

    def scan(self):
        """
        Detect new formations at the latest timestamp and generate signals.

        Returns
        -------
        list of dict, each with keys: ticker_a, ticker_b, signal_type ('formation'),
        and optionally transient validation metrics.
        """
        if self.cluster_history is None or len(self.cluster_history) == 0:
            return []

        latest_ts = self.cluster_history['Datetime'].max()
        latest_snap = self.cluster_history[
            self.cluster_history['Datetime'] == latest_ts
        ][['Ticker', 'Cluster_ID']]

        new_formations = detect_new_formations(latest_snap, self._previous_clusters)

        results = []
        for ticker_a, ticker_b in new_formations:
            signal = generate_transient_signals(
                ticker_a, ticker_b, self.ts_df, cfg=self.cfg.transient,
            )
            if signal is not None:
                signal['signal_type'] = 'formation'
                signal['formation_time'] = latest_ts
                results.append(signal)

        return results

    def validate_formations(self, formations_df):
        """
        Run transient validation on a DataFrame of formation events.

        Parameters
        ----------
        formations_df : DataFrame
            Must have columns: Ticker_1, Ticker_2, Formation_Time.

        Returns
        -------
        DataFrame with validation results.
        """
        results = []
        for _, row in formations_df.iterrows():
            result = validate_transient_event(
                row['Ticker_1'], row['Ticker_2'], row['Formation_Time'],
                self.ts_df, cfg=self.cfg.transient,
            )
            if result is not None:
                results.append(result)

        return pd.DataFrame(results)

    def get_stable_signals(self, pair_list):
        """
        Get signals for a list of stable pairs.

        Parameters
        ----------
        pair_list : list of (ticker_a, ticker_b)

        Returns
        -------
        list of signal dicts.
        """
        return track_stable_pairs(pair_list, self.ts_df, cfg=self.cfg.backtest)

    def update(self, new_prices_row=None):
        """
        Incremental update: cluster the latest timestamp and detect new formations.

        Parameters
        ----------
        new_prices_row : Series, optional
            Latest prices keyed by ticker. If None, uses last row of ts_df.

        Returns
        -------
        list of new formation signals (same format as scan()).
        """
        if self.cluster_history is None:
            return []

        # Store current latest as previous
        latest_ts = self.cluster_history['Datetime'].max()
        latest_snap = self.cluster_history[
            self.cluster_history['Datetime'] == latest_ts
        ]
        self._previous_clusters = dict(
            zip(latest_snap['Ticker'], latest_snap['Cluster_ID'])
        )

        # If new_prices_row provided, we'd need to recompute features
        # for that timestamp. For now, use the latest available in ts_df.
        return self.scan()

    @property
    def top_pairs(self):
        """Return pairs sorted by co-clustering frequency."""
        if self.pair_co_cluster_freq is None:
            return []
        total = self.cluster_history['Datetime'].nunique()
        pairs = [
            {'pair': f'{a}-{b}', 'ticker_a': a, 'ticker_b': b,
             'count': cnt, 'frequency': cnt / total}
            for (a, b), cnt in self.pair_co_cluster_freq.items()
        ]
        return sorted(pairs, key=lambda x: x['frequency'], reverse=True)
