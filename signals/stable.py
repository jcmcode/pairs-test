"""
Stable pair tracking and signal generation.

Extracted from optics-signals.ipynb. Monitors stable pairs with daily
cointegration checks and Bollinger band positions.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from validation.pair_validation import compute_hedge_ratio
from config import BacktestConfig, DEFAULT_CONFIG


def track_stable_pairs(pair_list, ts_df, cfg: BacktestConfig = None):
    """
    Daily cointegration check and Bollinger position for a list of stable pairs.

    Parameters
    ----------
    pair_list : list of (ticker_a, ticker_b)
    ts_df : DataFrame with MultiIndex (Datetime, Ticker)
    cfg : BacktestConfig

    Returns
    -------
    list of dicts with pair, coint_pval, z_score, signal, beta.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.backtest

    results = []
    for ticker_a, ticker_b in pair_list:
        try:
            da = ts_df.xs(ticker_a, level='Ticker').sort_index()['Price'].resample('1D').last().dropna()
            db = ts_df.xs(ticker_b, level='Ticker').sort_index()['Price'].resample('1D').last().dropna()
        except Exception:
            continue

        common = da.index.intersection(db.index)
        if len(common) < 50:
            continue

        da = da.loc[common]
        db = db.loc[common]

        try:
            _, pval, _ = coint(da, db)
        except Exception:
            pval = 1.0

        beta, _, _ = compute_hedge_ratio(da, db)
        if np.isnan(beta):
            continue

        spread = da - beta * db
        lb = cfg.lookback
        mu = spread.rolling(lb).mean()
        sd = spread.rolling(lb).std()
        z_latest = float((spread.iloc[-1] - mu.iloc[-1]) / sd.iloc[-1]) if sd.iloc[-1] > 0 else 0.0

        if z_latest >= cfg.entry_z:
            signal = 'short'
        elif z_latest <= -cfg.entry_z:
            signal = 'long'
        elif abs(z_latest) <= cfg.exit_z:
            signal = 'exit'
        else:
            signal = 'hold'

        results.append({
            'pair': f'{ticker_a}-{ticker_b}',
            'coint_pval': float(pval),
            'z_score': z_latest,
            'signal': signal,
            'beta': float(beta),
        })

    return results
