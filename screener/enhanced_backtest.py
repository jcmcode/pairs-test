"""
Enhanced backtesting with adaptive z-score optimization and Kalman filtering.

Compares baseline (static z=2.0, OLS) against enhanced (optimized z, Kalman)
strategies with transaction costs.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from validation.pair_validation import (
    compute_hedge_ratio, zscore_signals, simulate_spread_pnl,
)
from trading.trading import get_daily_prices


# ---------------------------------------------------------------------------
# Annualized Sharpe
# ---------------------------------------------------------------------------

def compute_annualized_sharpe(pnl_series, trading_days=252):
    """
    Annualized Sharpe from a cumulative P&L series.

    Computes daily returns, then mean/std * sqrt(trading_days).
    """
    if len(pnl_series) < 5:
        return np.nan
    daily_returns = pnl_series.diff().dropna()
    if len(daily_returns) < 3 or daily_returns.std() == 0:
        return np.nan
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(trading_days))


# ---------------------------------------------------------------------------
# Z-score parameter optimization
# ---------------------------------------------------------------------------

def optimize_zscore_params(spread_cal, min_trades=5):
    """
    Grid search for best z-score entry/exit/lookback on calibration data.

    Parameters
    ----------
    spread_cal : Series
        Calibration spread.
    min_trades : int
        Minimum trades required for a valid parameter combo.

    Returns
    -------
    dict with best params and cal performance, or None if nothing works.
    """
    entry_zs = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    exit_zs = [0.0, 0.25, 0.5, 0.75]
    lookbacks = [10, 15, 20]

    best = None
    best_sharpe = -np.inf

    for lb in lookbacks:
        if lb >= len(spread_cal) // 3:
            continue
        for ez in entry_zs:
            for xz in exit_zs:
                if xz >= ez:
                    continue
                signals = zscore_signals(spread_cal, lb, entry_z=ez, exit_z=xz)
                result = simulate_spread_pnl(spread_cal, signals)

                if result['n_trades'] < min_trades:
                    continue

                sharpe = result['sharpe']
                if not np.isfinite(sharpe):
                    continue

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best = {
                        'entry_z': ez,
                        'exit_z': xz,
                        'lookback': lb,
                        'cal_sharpe': sharpe,
                        'cal_n_trades': result['n_trades'],
                        'cal_pnl': result['total_pnl'],
                        'cal_win_rate': result['win_rate'],
                    }

    # Only return if profitable
    if best is not None and best['cal_pnl'] > 0:
        return best
    return None


# ---------------------------------------------------------------------------
# Kalman hedge ratio (cal-only)
# ---------------------------------------------------------------------------

def kalman_hedge_beta(prices_a_cal, prices_b_cal):
    """
    Run Kalman filter on calibration data and return the terminal beta.

    This gives a better hedge ratio estimate than OLS because the Kalman
    filter exponentially weights recent observations, adapting to drift.
    The terminal beta is then used as a FIXED hedge ratio for OOS —
    no adaptive spread, no look-ahead.

    Returns (beta, intercept) or (np.nan, np.nan) on failure.
    """
    beta, intercept, _ = compute_hedge_ratio(prices_a_cal, prices_b_cal, method='kalman')
    return beta, intercept


# ---------------------------------------------------------------------------
# Enhanced single-pair backtest
# ---------------------------------------------------------------------------

def enhanced_backtest_pair(
    prices_a_cal, prices_b_cal,
    prices_a_oos, prices_b_oos,
    cost_per_trade=0.001,
):
    """
    Run baseline and enhanced backtests for a single pair.

    Baseline: static z=2.0, OLS hedge, no costs.
    Enhanced: optimized z-score params from cal, with transaction costs.
    Kalman: Kalman-filtered spread with optimized params.

    Returns dict with all strategy results, or None if insufficient data.
    """
    # OLS hedge ratio from cal
    beta_ols, _, _ = compute_hedge_ratio(prices_a_cal, prices_b_cal, method='ols')
    if np.isnan(beta_ols):
        return None

    spread_cal_ols = pd.Series(
        np.asarray(prices_a_cal) - beta_ols * np.asarray(prices_b_cal),
        index=prices_a_cal.index if isinstance(prices_a_cal, pd.Series) else None,
    )
    spread_oos_ols = pd.Series(
        np.asarray(prices_a_oos) - beta_ols * np.asarray(prices_b_oos),
        index=prices_a_oos.index if isinstance(prices_a_oos, pd.Series) else None,
    )

    result = {}

    # --- Baseline: static z=2.0, OLS, no costs ---
    lb_baseline = min(20, len(spread_oos_ols) // 3)
    if lb_baseline >= 5:
        sig_base = zscore_signals(spread_oos_ols, lb_baseline, entry_z=2.0, exit_z=0.5)
        bt_base = simulate_spread_pnl(spread_oos_ols, sig_base, cost_per_trade=0.0)
        result['baseline_pnl'] = bt_base['total_pnl']
        result['baseline_n_trades'] = bt_base['n_trades']
        result['baseline_win_rate'] = bt_base['win_rate']
        result['baseline_sharpe'] = bt_base['sharpe']
        result['baseline_max_dd'] = bt_base['max_drawdown']
        result['baseline_ann_sharpe'] = compute_annualized_sharpe(bt_base['pnl_series'])
    else:
        result.update({
            'baseline_pnl': 0, 'baseline_n_trades': 0,
            'baseline_win_rate': np.nan, 'baseline_sharpe': np.nan,
            'baseline_max_dd': 0, 'baseline_ann_sharpe': np.nan,
        })

    # --- Enhanced: optimized z, OLS, with costs ---
    opt = optimize_zscore_params(spread_cal_ols)
    if opt is not None:
        result['opt_entry_z'] = opt['entry_z']
        result['opt_exit_z'] = opt['exit_z']
        result['opt_lookback'] = opt['lookback']
        result['opt_cal_sharpe'] = opt['cal_sharpe']

        lb_opt = min(opt['lookback'], len(spread_oos_ols) // 3)
        if lb_opt >= 5:
            sig_enh = zscore_signals(
                spread_oos_ols, lb_opt,
                entry_z=opt['entry_z'], exit_z=opt['exit_z'],
            )
            bt_enh = simulate_spread_pnl(spread_oos_ols, sig_enh, cost_per_trade=cost_per_trade)
            result['enhanced_pnl'] = bt_enh['total_pnl']
            result['enhanced_n_trades'] = bt_enh['n_trades']
            result['enhanced_win_rate'] = bt_enh['win_rate']
            result['enhanced_sharpe'] = bt_enh['sharpe']
            result['enhanced_max_dd'] = bt_enh['max_drawdown']
            result['enhanced_ann_sharpe'] = compute_annualized_sharpe(bt_enh['pnl_series'])
        else:
            result.update(_empty_strategy('enhanced'))
    else:
        result.update({
            'opt_entry_z': np.nan, 'opt_exit_z': np.nan,
            'opt_lookback': np.nan, 'opt_cal_sharpe': np.nan,
        })
        result.update(_empty_strategy('enhanced'))

    # --- Kalman: fixed Kalman beta from cal, applied to OOS ---
    # Run Kalman on cal only to get terminal beta (no look-ahead).
    # This gives a better hedge estimate than OLS since Kalman
    # exponentially weights recent observations, but the OOS spread
    # uses a FIXED beta — no adaptive spread that artificially
    # mean-reverts.
    beta_kal, _ = kalman_hedge_beta(prices_a_cal, prices_b_cal)

    if not np.isnan(beta_kal):
        spread_oos_kal = pd.Series(
            np.asarray(prices_a_oos) - beta_kal * np.asarray(prices_b_oos),
            index=prices_a_oos.index if isinstance(prices_a_oos, pd.Series) else None,
        )
        result['kalman_beta'] = beta_kal

        # Use optimized params if available, otherwise fall back to defaults
        kal_entry_z = opt['entry_z'] if opt is not None else 2.0
        kal_exit_z = opt['exit_z'] if opt is not None else 0.5
        kal_lookback = opt['lookback'] if opt is not None else 20

        if len(spread_oos_kal) > 10:
            lb_kal = min(kal_lookback, len(spread_oos_kal) // 3)
            if lb_kal >= 5:
                sig_kal = zscore_signals(
                    spread_oos_kal, lb_kal,
                    entry_z=kal_entry_z, exit_z=kal_exit_z,
                )
                bt_kal = simulate_spread_pnl(spread_oos_kal, sig_kal, cost_per_trade=cost_per_trade)
                result['kalman_pnl'] = bt_kal['total_pnl']
                result['kalman_n_trades'] = bt_kal['n_trades']
                result['kalman_win_rate'] = bt_kal['win_rate']
                result['kalman_sharpe'] = bt_kal['sharpe']
                result['kalman_max_dd'] = bt_kal['max_drawdown']
                result['kalman_ann_sharpe'] = compute_annualized_sharpe(bt_kal['pnl_series'])
            else:
                result.update(_empty_strategy('kalman'))
        else:
            result.update(_empty_strategy('kalman'))
    else:
        result.update(_empty_strategy('kalman'))

    return result


def _empty_strategy(prefix):
    """Return NaN dict for a strategy that couldn't run."""
    return {
        f'{prefix}_pnl': np.nan,
        f'{prefix}_n_trades': 0,
        f'{prefix}_win_rate': np.nan,
        f'{prefix}_sharpe': np.nan,
        f'{prefix}_max_dd': np.nan,
        f'{prefix}_ann_sharpe': np.nan,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_enhanced_analysis(pair_registry, ts_df, cost_per_trade=0.001):
    """
    Run enhanced backtests for all pairs in registry.

    Parameters
    ----------
    pair_registry : DataFrame
        Output of build_pair_registry() — needs Ticker_1, Ticker_2 columns.
    ts_df : DataFrame
        MultiIndex (Datetime, Ticker) with Price column.
    cost_per_trade : float
        Round-trip transaction cost (default 10bps).

    Returns
    -------
    DataFrame with baseline, enhanced, and Kalman results per pair.
    """
    results = []
    n = len(pair_registry)

    for i, (_, row) in enumerate(pair_registry.iterrows()):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Enhanced backtest {i + 1}/{n}: {row['Pair']}")

        daily_a = get_daily_prices(row["Ticker_1"], ts_df)
        daily_b = get_daily_prices(row["Ticker_2"], ts_df)
        common = daily_a.index.intersection(daily_b.index)

        if len(common) < 60:
            continue

        da = daily_a.loc[common]
        db = daily_b.loc[common]

        split_idx = int(len(common) * 0.67)
        if split_idx < 40 or (len(common) - split_idx) < 20:
            continue

        da_cal, da_oos = da.iloc[:split_idx], da.iloc[split_idx:]
        db_cal, db_oos = db.iloc[:split_idx], db.iloc[split_idx:]

        bt = enhanced_backtest_pair(
            da_cal, db_cal, da_oos, db_oos,
            cost_per_trade=cost_per_trade,
        )
        if bt is None:
            continue

        entry = {
            "pair": row["Pair"],
            "ticker_a": row["Ticker_1"],
            "ticker_b": row["Ticker_2"],
            "noise_adj_freq": row["noise_adj_freq"],
        }
        for col in ["sector_1", "sector_2", "pair_type"]:
            if col in row.index:
                entry[col] = row[col]

        entry.update(bt)
        results.append(entry)

    return pd.DataFrame(results)
