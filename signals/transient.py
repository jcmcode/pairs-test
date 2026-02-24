"""
Transient pair validation and signal generation.

Extracted from optics-signals.ipynb. Validates transient formation events
using calibration/exploitation windows and generates real-time trading signals.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from validation.pair_validation import (
    compute_hedge_ratio, hedge_ratio_drift, spread_cv_normalized,
    half_life, zscore_signals, simulate_spread_pnl,
)
from config import TransientConfig, DEFAULT_CONFIG


def _get_price_window(ticker, timestamp, window_obs, ts_df, direction='forward'):
    """
    Extract price data for a ticker over a window of N observations.
    """
    try:
        ticker_data = ts_df.xs(ticker, level='Ticker').sort_index()
        if timestamp not in ticker_data.index:
            idx_pos = ticker_data.index.searchsorted(timestamp)
            if idx_pos >= len(ticker_data.index):
                return pd.Series(dtype=float)
            timestamp = ticker_data.index[idx_pos]
        loc = ticker_data.index.get_loc(timestamp)
        if direction == 'forward':
            end_loc = min(loc + window_obs, len(ticker_data))
            return ticker_data.iloc[loc:end_loc]['Price']
        else:
            start_loc = max(0, loc - window_obs + 1)
            return ticker_data.iloc[start_loc:loc + 1]['Price']
    except Exception:
        return pd.Series(dtype=float)


def validate_transient_event(ticker_a, ticker_b, timestamp, ts_df,
                             cfg: TransientConfig = None):
    """
    Validate a single transient pair event using three-window approach:
      - Execution lag: skip first N obs after formation
      - Calibration window: estimate hedge ratio
      - Exploitation window: test spread metrics + generate signals + simulate P&L

    Parameters
    ----------
    ticker_a, ticker_b : str
    timestamp : Timestamp
    ts_df : DataFrame with MultiIndex (Datetime, Ticker)
    cfg : TransientConfig

    Returns
    -------
    dict with all metrics + pass/fail, or None if insufficient data.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.transient

    total_need = cfg.execution_lag + cfg.calibration + cfg.exploitation

    prices_a = _get_price_window(ticker_a, timestamp, total_need, ts_df, direction='forward')
    prices_b = _get_price_window(ticker_b, timestamp, total_need, ts_df, direction='forward')

    if len(prices_a) < total_need or len(prices_b) < total_need:
        return None

    common_idx = prices_a.index.intersection(prices_b.index)
    pa = prices_a.loc[common_idx]
    pb = prices_b.loc[common_idx]
    if len(pa) < total_need:
        return None

    lag = cfg.execution_lag
    cal_len = cfg.calibration
    exploit_len = cfg.exploitation

    pa_cal = pa.iloc[lag:lag + cal_len]
    pb_cal = pb.iloc[lag:lag + cal_len]
    pa_exploit = pa.iloc[lag + cal_len:lag + cal_len + exploit_len]
    pb_exploit = pb.iloc[lag + cal_len:lag + cal_len + exploit_len]

    if len(pa_cal) < 5 or len(pa_exploit) < 10:
        return None

    beta_cal, intercept_cal, r_sq_cal = compute_hedge_ratio(pa_cal, pb_cal, method='ols')
    if np.isnan(beta_cal):
        return None

    spread_exploit = pa_exploit - beta_cal * pb_exploit
    cv = spread_cv_normalized(spread_exploit, pa_exploit, pb_exploit)
    hl = half_life(spread_exploit)

    ret_a = pa_exploit.pct_change().dropna()
    ret_b = pb_exploit.pct_change().dropna()
    common_ret = ret_a.index.intersection(ret_b.index)
    corr = pearsonr(ret_a.loc[common_ret], ret_b.loc[common_ret])[0] if len(common_ret) >= 10 else np.nan

    beta_exploit_val, _, _ = compute_hedge_ratio(pa_exploit, pb_exploit, method='ols')
    drift = hedge_ratio_drift(beta_cal, beta_exploit_val)

    lookback = max(5, min(20, len(spread_exploit) // 2))
    signals = zscore_signals(spread_exploit, lookback=lookback,
                             entry_z=cfg.entry_z, exit_z=cfg.exit_z)
    pnl_result = simulate_spread_pnl(spread_exploit, signals, cost_per_trade=0.0)

    has_signal = signals['long_entry'].any() or signals['short_entry'].any()

    passed = (
        (not np.isnan(corr)) and corr > cfg.corr_threshold
        and (not np.isnan(cv)) and cv < cfg.cv_threshold
        and (not np.isnan(hl)) and 0 < hl < cfg.hl_max
        and (not np.isnan(drift)) and drift < cfg.hedge_drift_max
        and has_signal
    )

    return {
        'ticker_a': ticker_a,
        'ticker_b': ticker_b,
        'timestamp': timestamp,
        'beta_cal': beta_cal,
        'r_squared_cal': r_sq_cal,
        'correlation': corr,
        'spread_cv': cv,
        'half_life': hl,
        'hedge_drift': drift,
        'has_signal': has_signal,
        'n_trades': pnl_result['n_trades'],
        'total_pnl': pnl_result['total_pnl'],
        'win_rate': pnl_result['win_rate'],
        'sharpe': pnl_result['sharpe'],
        'max_drawdown': pnl_result['max_drawdown'],
        'passed': passed,
    }


def generate_transient_signals(ticker_a, ticker_b, ts_df,
                               cfg: TransientConfig = None):
    """
    For a given pair, compute the current z-score and return a signal dict.

    Uses the last calibration-window observations to fit a hedge ratio,
    then scores the most recent observations against that baseline.

    Parameters
    ----------
    ticker_a, ticker_b : str
    ts_df : DataFrame with MultiIndex (Datetime, Ticker)
    cfg : TransientConfig

    Returns
    -------
    dict with z_score, signal ('long', 'short', 'exit', 'hold'), spread_cv, half_life.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.transient

    cal_len = cfg.calibration
    try:
        ticker_data_a = ts_df.xs(ticker_a, level='Ticker').sort_index()['Price']
        ticker_data_b = ts_df.xs(ticker_b, level='Ticker').sort_index()['Price']
    except KeyError:
        return None

    if len(ticker_data_a) < cal_len + 10 or len(ticker_data_b) < cal_len + 10:
        return None

    common = ticker_data_a.index.intersection(ticker_data_b.index)
    pa = ticker_data_a.loc[common]
    pb = ticker_data_b.loc[common]

    pa_cal = pa.iloc[-cal_len - 10:-10]
    pb_cal = pb.iloc[-cal_len - 10:-10]
    pa_recent = pa.iloc[-10:]
    pb_recent = pb.iloc[-10:]

    beta, _, _ = compute_hedge_ratio(pa_cal, pb_cal)
    if np.isnan(beta):
        return None

    spread_cal = pa_cal - beta * pb_cal
    mu = spread_cal.mean()
    sd = spread_cal.std()

    cv = spread_cv_normalized(spread_cal, pa_cal, pb_cal)
    hl = half_life(spread_cal)

    spread_recent = pa_recent - beta * pb_recent
    z = float((spread_recent.iloc[-1] - mu) / sd) if sd > 0 else 0.0

    if z >= cfg.entry_z:
        signal = 'short'
    elif z <= -cfg.entry_z:
        signal = 'long'
    elif abs(z) <= cfg.exit_z:
        signal = 'exit'
    else:
        signal = 'hold'

    return {
        'ticker_a': ticker_a,
        'ticker_b': ticker_b,
        'z_score': z,
        'signal': signal,
        'beta': float(beta),
        'spread_cv': float(cv) if np.isfinite(cv) else None,
        'half_life': float(hl) if np.isfinite(hl) else None,
    }
