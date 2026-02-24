"""
Validate transient engine discoveries as traditional pairs trades.

Takes pairs surfaced by OPTICS clustering, tests them with classical pairs
trading metrics (cointegration, half-life, Hurst exponent) over the full
daily history, and backtests z-score mean reversion strategies on qualifying pairs.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from validation.pair_validation import (
    compute_hedge_ratio, zscore_signals, simulate_spread_pnl,
    half_life, spread_cv_normalized,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research', 'data')

# 8 pairs found in all 3 algorithms' top-20 (from algorithm-comparison)
CONSENSUS_PAIRS = {
    'ADI-NXPI', 'ADI-SWKS', 'ADI-TXN', 'AMAT-KLAC',
    'AMAT-LRCX', 'KLAC-LRCX', 'NXPI-TXN', 'QRVO-SWKS',
}

# 11 pairs with permutation Z > 1.96 (KMeans/DBSCAN permutation tests)
SIGNIFICANT_PAIRS = {
    'QRVO-SWKS', 'ADI-TXN', 'ADI-SWKS', 'ADI-NXPI',
    'QCOM-QRVO', 'QCOM-SWKS', 'MCHP-ON', 'CDNS-SNPS',
    'MCHP-NXPI', 'STX-WDC', 'GFS-NXPI',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_artifacts(data_dir=DATA_DIR):
    """Load pickle artifacts for OPTICS, KMeans, and DBSCAN."""
    artifact_names = [
        'ts_df', 'cluster_history', 'df_pair_stability',
        'pair_co_cluster_freq', 'pair_classification',
    ]
    prefixes = {'optics': '', 'kmeans': 'kmeans_', 'dbscan': 'dbscan_'}

    artifacts = {}
    for algo, prefix in prefixes.items():
        algo_data = {}
        for name in artifact_names:
            path = os.path.join(data_dir, f'{prefix}{name}.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    algo_data[name] = pickle.load(f)
        if algo_data:
            artifacts[algo] = algo_data
            print(f"  {algo}: loaded {len(algo_data)}/{len(artifact_names)} artifacts")
        else:
            print(f"  {algo}: no artifacts found")

    return artifacts


# ---------------------------------------------------------------------------
# Noise-adjusted frequency
# ---------------------------------------------------------------------------

def compute_noise_adjusted_frequency(cluster_history, pair_co_cluster_freq):
    """
    Compute noise-adjusted co-clustering frequency.

    OPTICS raw frequency uses ALL timestamps as denominator, but ~58% of the
    time a ticker is in noise (cluster -1). Noise-adjusted denominator = only
    timestamps where BOTH tickers are present AND non-noise.
    """
    # Per-ticker: timestamps where the ticker is present (any cluster ID)
    present_ts = {}
    noise_ts = {}
    for ticker, group in cluster_history.groupby('Ticker'):
        present_ts[ticker] = set(group['Datetime'].values)
        noise_ts[ticker] = set(
            group[group['Cluster_ID'] == -1]['Datetime'].values
        )

    adjusted = {}
    for (a, b), raw_count in pair_co_cluster_freq.items():
        # Timestamps where both tickers exist AND neither is noise
        both_present = present_ts.get(a, set()) & present_ts.get(b, set())
        valid_ts = both_present - noise_ts.get(a, set()) - noise_ts.get(b, set())
        if len(valid_ts) == 0:
            adjusted[(a, b)] = 0.0
        else:
            adjusted[(a, b)] = raw_count / len(valid_ts)

    return adjusted


# ---------------------------------------------------------------------------
# Pair registry
# ---------------------------------------------------------------------------

def build_pair_registry(artifacts):
    """
    Build registry of pairs from OPTICS classification with noise-adjusted
    frequency and cross-algorithm consensus flags.
    """
    optics = artifacts['optics']
    pc = optics['pair_classification'].copy()
    stab = optics['df_pair_stability']

    # Merge to get ticker names and raw frequency
    registry = pc.merge(
        stab[['Pair', 'Ticker_1', 'Ticker_2', 'Co_Cluster_Frequency']],
        on='Pair', how='left',
    )

    # Noise-adjusted frequency
    noise_adj = compute_noise_adjusted_frequency(
        optics['cluster_history'], optics['pair_co_cluster_freq'],
    )
    registry['noise_adj_freq'] = registry.apply(
        lambda r: noise_adj.get(
            (r['Ticker_1'], r['Ticker_2']),
            noise_adj.get((r['Ticker_2'], r['Ticker_1']), 0.0),
        ), axis=1,
    )

    # KMeans/DBSCAN flags
    for algo_key, col_prefix in [('kmeans', 'kmeans'), ('dbscan', 'dbscan')]:
        if algo_key in artifacts and 'df_pair_stability' in artifacts[algo_key]:
            algo_stab = artifacts[algo_key]['df_pair_stability']
            top20 = set(algo_stab.nlargest(20, 'Co_Cluster_Frequency')['Pair'])
            registry[f'in_{col_prefix}_top20'] = registry['Pair'].isin(top20)

            if 'pair_classification' in artifacts[algo_key]:
                algo_pc = artifacts[algo_key]['pair_classification']
                transient = set(algo_pc[algo_pc['Category'] == 'transient']['Pair'])
                registry[f'in_{col_prefix}_transient'] = registry['Pair'].isin(transient)
            else:
                registry[f'in_{col_prefix}_transient'] = False
        else:
            registry[f'in_{col_prefix}_top20'] = False
            registry[f'in_{col_prefix}_transient'] = False

    # Consensus and significance flags
    registry['is_consensus'] = registry['Pair'].isin(CONSENSUS_PAIRS)
    registry['is_significant'] = registry['Pair'].isin(SIGNIFICANT_PAIRS)

    # Filter: noise-adjusted freq > 0.15
    registry = registry[registry['noise_adj_freq'] > 0.15].copy()
    registry = registry.sort_values('noise_adj_freq', ascending=False).reset_index(drop=True)

    return registry


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def get_daily_prices(ticker, ts_df):
    """Resample hourly prices to daily (last observation per day)."""
    try:
        ticker_data = ts_df.xs(ticker, level='Ticker').sort_index()['Price']
        return ticker_data.resample('1D').last().dropna()
    except Exception:
        return pd.Series(dtype=float)


def hurst_exponent(spread, max_lag=20):
    """Variance ratio method for Hurst exponent. H < 0.5 = mean-reverting."""
    ts = np.asarray(spread, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < max_lag * 2:
        return np.nan
    lags = range(2, max_lag + 1)
    tau = []
    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        tau.append(np.nanstd(diffs))
    if any(t == 0 for t in tau):
        return np.nan
    coef = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return coef[0]


# ---------------------------------------------------------------------------
# Classical pairs trading tests
# ---------------------------------------------------------------------------

def test_pair_fundamentals(ticker_a, ticker_b, ts_df, cal_frac=0.67):
    """
    Test a pair with classical pairs trading metrics.
    Calibration/OOS split: hedge ratio and metrics computed on first cal_frac,
    backtest runs on the remaining OOS portion.
    """
    daily_a = get_daily_prices(ticker_a, ts_df)
    daily_b = get_daily_prices(ticker_b, ts_df)

    common = daily_a.index.intersection(daily_b.index)
    if len(common) < 60:
        return None

    da = daily_a.loc[common]
    db = daily_b.loc[common]

    # Cal/OOS split
    split_idx = int(len(common) * cal_frac)
    if split_idx < 40 or (len(common) - split_idx) < 20:
        return None

    da_cal, da_oos = da.iloc[:split_idx], da.iloc[split_idx:]
    db_cal, db_oos = db.iloc[:split_idx], db.iloc[split_idx:]

    # Hedge ratio on calibration ONLY
    beta, intercept, r_sq = compute_hedge_ratio(da_cal, db_cal, method='ols')
    if np.isnan(beta):
        return None

    spread_cal = da_cal - beta * db_cal

    # Cointegration: test both directions, take minimum p-value (Issue 7)
    try:
        _, pval_ab, _ = coint(da_cal, db_cal)
        _, pval_ba, _ = coint(db_cal, da_cal)
        coint_pval = min(pval_ab, pval_ba)
    except Exception:
        coint_pval = 1.0

    hl = half_life(spread_cal)
    hurst = hurst_exponent(spread_cal.values)
    cv = spread_cv_normalized(spread_cal, da_cal, db_cal)

    # OOS spread for backtesting
    spread_oos = da_oos - beta * db_oos

    crit_coint = coint_pval < 0.05
    crit_hl = (not np.isnan(hl)) and hl != np.inf and 5 <= hl <= 60
    crit_hurst = (not np.isnan(hurst)) and hurst < 0.5
    n_met = sum([crit_coint, crit_hl, crit_hurst])

    return {
        'n_daily_obs': len(common),
        'n_cal_obs': split_idx,
        'n_oos_obs': len(common) - split_idx,
        'beta': beta,
        'r_squared': r_sq,
        'coint_pval': coint_pval,
        'half_life_days': hl,
        'hurst': hurst,
        'spread_cv': cv,
        'crit_coint': crit_coint,
        'crit_hl': crit_hl,
        'crit_hurst': crit_hurst,
        'n_criteria_met': n_met,
        'passed': n_met == 3,
        'near_miss': n_met == 2,
        'spread_cal': spread_cal,
        'spread_oos': spread_oos,
    }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_backtest(daily_a, daily_b, n_splits=5, cal_frac=0.67,
                          lookback=20, entry_z=2.0, exit_z=0.5):
    """
    Rolling walk-forward backtest with multiple cal/OOS splits.

    Instead of a single 67/33 split, slides a window through the data
    to produce n_splits independent OOS evaluations.

    Parameters
    ----------
    daily_a, daily_b : Series
        Aligned daily price series for the two tickers.
    n_splits : int
        Number of rolling splits.
    cal_frac : float
        Fraction of each window used for calibration.
    lookback, entry_z, exit_z : float
        Backtest parameters.

    Returns
    -------
    dict with per-split results and aggregate metrics.
    """
    common = daily_a.index.intersection(daily_b.index)
    if len(common) < 80:
        return None

    da = daily_a.loc[common]
    db = daily_b.loc[common]
    n = len(da)

    # Each split uses a window of size n // (1 + n_splits * (1 - cal_frac))
    # sliding forward by step_size
    window_size = int(n * 0.8)
    step_size = max(1, (n - window_size) // max(1, n_splits - 1))

    split_results = []
    for i in range(n_splits):
        start = i * step_size
        end = start + window_size
        if end > n:
            break

        da_win = da.iloc[start:end]
        db_win = db.iloc[start:end]
        split_idx = int(len(da_win) * cal_frac)

        if split_idx < 30 or (len(da_win) - split_idx) < 15:
            continue

        da_cal = da_win.iloc[:split_idx]
        db_cal = db_win.iloc[:split_idx]
        da_oos = da_win.iloc[split_idx:]
        db_oos = db_win.iloc[split_idx:]

        beta, _, _ = compute_hedge_ratio(da_cal, db_cal, method='ols')
        if np.isnan(beta):
            continue

        spread_oos = da_oos - beta * db_oos
        bt = backtest_pair(spread_oos, lookback=lookback, entry_z=entry_z, exit_z=exit_z)
        bt['split'] = i
        bt['cal_start'] = da_cal.index[0]
        bt['oos_start'] = da_oos.index[0]
        bt['oos_end'] = da_oos.index[-1]
        bt['beta'] = beta
        split_results.append(bt)

    if not split_results:
        return None

    sharpes = [r['sharpe'] for r in split_results if np.isfinite(r['sharpe'])]
    pnls = [r['total_pnl'] for r in split_results]
    trades = [r['n_trades'] for r in split_results]

    return {
        'n_splits': len(split_results),
        'avg_sharpe': float(np.mean(sharpes)) if sharpes else np.nan,
        'std_sharpe': float(np.std(sharpes)) if sharpes else np.nan,
        'avg_pnl': float(np.mean(pnls)),
        'total_trades': int(sum(trades)),
        'splits': split_results,
    }


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest_pair(spread, lookback=20, entry_z=2.0, exit_z=0.5):
    """Run z-score mean reversion backtest on a spread series."""
    lb = min(lookback, len(spread) // 3)
    if lb < 5:
        return {'total_pnl': 0, 'n_trades': 0, 'win_rate': np.nan,
                'sharpe': np.nan, 'max_drawdown': 0}

    signals = zscore_signals(spread, lb, entry_z, exit_z)
    result = simulate_spread_pnl(spread, signals)
    return {
        'total_pnl': result['total_pnl'],
        'n_trades': result['n_trades'],
        'win_rate': result['win_rate'],
        'sharpe': result['sharpe'],
        'max_drawdown': result['max_drawdown'],
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def run_full_analysis(pair_registry, ts_df):
    """Run fundamental tests and backtests on all pairs in registry."""
    results = []
    n = len(pair_registry)

    for i, (_, row) in enumerate(pair_registry.iterrows()):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Testing pair {i+1}/{n}: {row['Pair']}")

        fundamentals = test_pair_fundamentals(row['Ticker_1'], row['Ticker_2'], ts_df)

        entry = {
            'pair': row['Pair'],
            'ticker_a': row['Ticker_1'],
            'ticker_b': row['Ticker_2'],
            'category': row.get('Category', ''),
            'raw_freq': row.get('Co_Cluster_Frequency', np.nan),
            'noise_adj_freq': row['noise_adj_freq'],
            'is_consensus': row['is_consensus'],
            'is_significant': row['is_significant'],
            'in_kmeans_top20': row.get('in_kmeans_top20', False),
            'in_dbscan_top20': row.get('in_dbscan_top20', False),
        }

        if fundamentals is None:
            entry.update({
                'n_daily_obs': 0, 'beta': np.nan, 'coint_pval': np.nan,
                'half_life_days': np.nan, 'hurst': np.nan, 'spread_cv': np.nan,
                'n_criteria_met': 0, 'passed': False, 'near_miss': False,
                'total_pnl': np.nan, 'n_trades': 0, 'win_rate': np.nan,
                'sharpe': np.nan, 'max_drawdown': np.nan,
            })
        else:
            entry.update({
                'n_daily_obs': fundamentals['n_daily_obs'],
                'beta': fundamentals['beta'],
                'coint_pval': fundamentals['coint_pval'],
                'half_life_days': fundamentals['half_life_days'],
                'hurst': fundamentals['hurst'],
                'spread_cv': fundamentals['spread_cv'],
                'n_criteria_met': fundamentals['n_criteria_met'],
                'passed': fundamentals['passed'],
                'near_miss': fundamentals['near_miss'],
            })

            if fundamentals['passed'] or fundamentals['near_miss']:
                bt = backtest_pair(fundamentals['spread_oos'])
                entry.update(bt)
            else:
                entry.update({
                    'total_pnl': np.nan, 'n_trades': 0, 'win_rate': np.nan,
                    'sharpe': np.nan, 'max_drawdown': np.nan,
                })

        results.append(entry)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(results):
    """Generate text report from analysis results."""
    lines = []
    lines.append('=' * 80)
    lines.append('TRANSIENT ENGINE -> TRADITIONAL PAIRS TRADE VALIDATION')
    lines.append('(All metrics computed on calibration window; backtests on out-of-sample data)')
    lines.append('=' * 80)

    # Section 1: Summary
    n_total = len(results)
    n_passed = int(results['passed'].sum())
    n_near = int(results['near_miss'].sum())
    n_failed = n_total - n_passed - n_near

    lines.append(f'\n--- SUMMARY ---')
    lines.append(f'Pairs tested:    {n_total}')
    lines.append(f'Passed (3/3):    {n_passed}')
    lines.append(f'Near-miss (2/3): {n_near}')
    lines.append(f'Failed:          {n_failed}')

    # Section 2: Results table
    tradeable = results[results['passed'] | results['near_miss']].copy()
    if len(tradeable) > 0:
        lines.append(f'\n--- TRADEABLE PAIRS (sorted by Sharpe) ---')
        tradeable = tradeable.sort_values('sharpe', ascending=False)
        cols = [
            'pair', 'noise_adj_freq', 'coint_pval', 'half_life_days',
            'hurst', 'n_trades', 'total_pnl', 'sharpe', 'win_rate',
            'is_consensus', 'is_significant',
        ]
        lines.append(tradeable[cols].to_string(index=False, float_format='%.4f'))
    else:
        lines.append('\n--- TRADEABLE PAIRS ---')
        lines.append('  No pairs passed or near-missed.')

    # Failed pairs breakdown
    failed = results[~results['passed'] & ~results['near_miss']].copy()
    if len(failed) > 0:
        lines.append(f'\n--- FAILED PAIRS (top 15 by noise-adjusted freq) ---')
        failed_top = failed.nlargest(15, 'noise_adj_freq')
        cols_f = [
            'pair', 'noise_adj_freq', 'coint_pval', 'half_life_days',
            'hurst', 'n_criteria_met',
        ]
        lines.append(failed_top[cols_f].to_string(index=False, float_format='%.4f'))

    # Section 3: Consensus comparison
    lines.append(f'\n--- CONSENSUS vs NON-CONSENSUS ---')
    for label, mask in [('Consensus', results['is_consensus']),
                        ('Non-consensus', ~results['is_consensus'])]:
        subset = results[mask]
        if len(subset) == 0:
            continue
        n_p = int(subset['passed'].sum())
        n_nm = int(subset['near_miss'].sum())
        bt_subset = subset[subset['passed'] | subset['near_miss']]
        avg_sharpe = bt_subset['sharpe'].mean() if len(bt_subset) > 0 else np.nan
        avg_pnl = bt_subset['total_pnl'].mean() if len(bt_subset) > 0 else np.nan
        lines.append(f'  {label} ({len(subset)} pairs):')
        lines.append(f'    Passed: {n_p}, Near-miss: {n_nm}')
        if not np.isnan(avg_sharpe):
            lines.append(f'    Avg Sharpe (tradeable): {avg_sharpe:.3f}')
            lines.append(f'    Avg P&L (tradeable):    {avg_pnl:.4f}')
        else:
            lines.append(f'    Avg Sharpe: N/A')
            lines.append(f'    Avg P&L: N/A')

    # Section 4: Verdict
    lines.append(f'\n--- VERDICT ---')
    if n_passed > 0:
        lines.append(
            f'The transient engine identified {n_passed} pair(s) that pass all classical '
            f'pairs trading criteria (cointegration + half-life 5-60d + Hurst < 0.5).'
        )
    if n_near > 0:
        lines.append(
            f'{n_near} additional pair(s) are near-misses (2/3 criteria met), consistent '
            f'with transient dynamics showing mean reversion without formal cointegration.'
        )
    if n_passed == 0 and n_near == 0:
        lines.append(
            'The transient relationships detected by the engine do not translate to '
            'long-term tradeable opportunities under classical pairs trading criteria.'
        )

    consensus_tradeable = results[
        results['is_consensus'] & (results['passed'] | results['near_miss'])
    ]
    if len(consensus_tradeable) > 0:
        lines.append(
            f'Cross-algorithm consensus confirmation: {len(consensus_tradeable)} consensus '
            f'pair(s) are tradeable, supporting the engine as a pair discovery tool.'
        )

    lines.append('=' * 80)

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir=DATA_DIR):
    """Run full analysis pipeline. Returns (results_df, report_text)."""
    print('Loading artifacts...')
    artifacts = load_artifacts(data_dir)

    if 'optics' not in artifacts:
        print('ERROR: OPTICS artifacts not found.')
        return None, None

    print('\nBuilding pair registry...')
    registry = build_pair_registry(artifacts)
    print(f'  {len(registry)} pairs with noise-adjusted freq > 0.15')

    ts_df = artifacts['optics']['ts_df']

    print('\nRunning fundamental tests & backtests...')
    results = run_full_analysis(registry, ts_df)

    print('\nGenerating report...')
    report = generate_report(results)
    print(report)

    return results, report


if __name__ == '__main__':
    main()
