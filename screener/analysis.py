"""
Generic pair registry builder and cross-sector analysis.

Clusters are built from a combined multi-sector universe. After clustering,
pairs are classified as intra-sector or cross-sector based on the sector map.

Validation uses a 5-test scored framework designed for transient correlations
(not long-run cointegration). Each test scores 0 or 1; total score 0-5.
"""

import os
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from trading.trading import (
    compute_noise_adjusted_frequency,
    test_pair_fundamentals,
    backtest_pair,
    walk_forward_backtest,
    get_daily_prices,
    hurst_exponent,
)
from validation.pair_validation import compute_hedge_ratio, half_life

# Sharpe ratios from <5 trades are unreliable (std ~ 0 -> Sharpe explodes).
MIN_TRADES_FOR_SHARPE = 5


def _sanitize_sharpe(sharpe, n_trades):
    """Return NaN if trade count is too low for a meaningful Sharpe."""
    if n_trades < MIN_TRADES_FOR_SHARPE:
        return np.nan
    if not np.isfinite(sharpe):
        return np.nan
    return sharpe


# ---------------------------------------------------------------------------
# 5-test validation framework
# ---------------------------------------------------------------------------

def _test_adf(spread, threshold=0.10):
    """ADF test on spread. Returns (passed, p-value)."""
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 20:
        return False, np.nan
    try:
        result = adfuller(s, maxlag=None, autolag='AIC')
        pval = result[1]
        return pval < threshold, float(pval)
    except Exception:
        return False, np.nan


def _test_half_life(spread, lo=5, hi=60):
    """Half-life in [lo, hi] days. Returns (passed, value)."""
    hl = half_life(spread)
    if np.isnan(hl) or hl == np.inf:
        return False, hl
    return lo <= hl <= hi, float(hl)


def _test_hurst(spread, threshold=0.5, max_lag=20):
    """Hurst exponent < threshold. Returns (passed, value)."""
    h = hurst_exponent(spread, max_lag=max_lag)
    if np.isnan(h):
        return False, np.nan
    return h < threshold, float(h)


def _test_variance_ratio(spread, lag=2, alpha=0.10):
    """
    Lo-MacKinlay variance ratio test.

    Under RW, VR(lag) = 1. For mean-reverting series, VR < 1.
    Returns (passed, vr_value, z_stat).
    """
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < lag * 4:
        return False, np.nan, np.nan

    # Returns at lag-1 and lag-q
    ret1 = np.diff(s)
    retq = s[lag:] - s[:-lag]

    var1 = np.var(ret1, ddof=1)
    varq = np.var(retq, ddof=1)

    if var1 == 0:
        return False, np.nan, np.nan

    vr = (varq / lag) / var1

    # Asymptotic z-statistic under homoskedasticity
    # Variance of VR under null ~ 2*(2q-1)*(q-1) / (3*q*n)
    nq = n - 1  # number of lag-1 returns
    stat_var = 2.0 * (2 * lag - 1) * (lag - 1) / (3.0 * lag * nq)
    if stat_var <= 0:
        return False, float(vr), np.nan

    z_stat = (vr - 1.0) / np.sqrt(stat_var)

    # One-sided test: reject RW in favor of mean-reversion (VR < 1, z < 0)
    from scipy.stats import norm
    p_value = norm.cdf(z_stat)  # left tail
    passed = p_value < alpha

    return passed, float(vr), float(z_stat)


def _test_rolling_correlation(prices_a, prices_b, n_windows=4, stability_threshold=0.5):
    """
    Rolling correlation stability test on returns (not price levels).

    Split calibration period into n_windows sub-windows, compute Pearson
    correlation of daily returns in each, then measure consistency. Using
    returns avoids spurious correlation from co-trending prices.

    Stability = 1 - (std of correlations / mean of abs correlations).
    Returns (passed, stability_value, correlations_list).
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    if len(a) < n_windows * 10 + 1:
        return False, np.nan, []

    # Use percentage returns to avoid spurious correlation from co-trending levels
    ret_a = np.diff(a) / a[:-1]
    ret_b = np.diff(b) / b[:-1]

    window_size = len(ret_a) // n_windows
    correlations = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        if end > len(ret_a):
            break
        chunk_a = ret_a[start:end]
        chunk_b = ret_b[start:end]
        if np.std(chunk_a) == 0 or np.std(chunk_b) == 0:
            continue
        r = np.corrcoef(chunk_a, chunk_b)[0, 1]
        if np.isfinite(r):
            correlations.append(r)

    if len(correlations) < 2:
        return False, np.nan, correlations

    corr_arr = np.array(correlations)
    mean_abs = np.mean(np.abs(corr_arr))
    if mean_abs == 0:
        return False, 0.0, correlations

    stability = 1.0 - np.std(corr_arr) / mean_abs
    return stability > stability_threshold, float(stability), [float(c) for c in correlations]


def validate_pair_relationship(spread_cal, prices_a_cal, prices_b_cal):
    """
    Run the 5-test validation framework on a pair's calibration data.

    Returns dict with individual test results and total score (0-5).
    """
    # Test 1: ADF on spread
    adf_passed, adf_pval = _test_adf(spread_cal)

    # Test 2: Half-life 5-60 days
    hl_passed, hl_value = _test_half_life(spread_cal)

    # Test 3: Hurst exponent < 0.5
    hurst_passed, hurst_value = _test_hurst(spread_cal)

    # Test 4: Variance ratio
    vr_passed, vr_value, vr_zstat = _test_variance_ratio(spread_cal)

    # Test 5: Rolling correlation stability
    rc_passed, rc_stability, rc_corrs = _test_rolling_correlation(
        prices_a_cal, prices_b_cal
    )

    score = sum([adf_passed, hl_passed, hurst_passed, vr_passed, rc_passed])

    # Classification
    if score >= 4:
        classification = "strong"
    elif score == 3:
        classification = "moderate"
    elif score == 2:
        classification = "weak"
    else:
        classification = "fail"

    return {
        "adf_pval": adf_pval,
        "adf_passed": adf_passed,
        "half_life_days": hl_value,
        "hl_passed": hl_passed,
        "hurst": hurst_value,
        "hurst_passed": hurst_passed,
        "variance_ratio": vr_value,
        "vr_zstat": vr_zstat,
        "vr_passed": vr_passed,
        "rolling_corr_stability": rc_stability,
        "rolling_corr_values": rc_corrs,
        "rc_passed": rc_passed,
        "score": score,
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# Pair registry
# ---------------------------------------------------------------------------

def build_pair_registry(
    cluster_history,
    pair_co_cluster_freq,
    sector_map=None,
    noise_adj_freq_threshold=0.08,
):
    """
    Build a pair registry from clustering artifacts.

    Parameters
    ----------
    cluster_history : DataFrame
        Columns: Ticker, Datetime, Cluster_ID.
    pair_co_cluster_freq : dict
        Mapping (ticker_a, ticker_b) -> co-cluster count.
    sector_map : dict, optional
        Mapping ticker -> sector name. If provided, pairs are tagged as
        intra-sector or cross-sector.
    noise_adj_freq_threshold : float
        Minimum noise-adjusted frequency to include a pair (default 0.08).

    Returns
    -------
    DataFrame with columns: Pair, Ticker_1, Ticker_2, raw_count,
    raw_freq, noise_adj_freq, sector_1, sector_2, pair_type.
    """
    total_windows = cluster_history["Datetime"].nunique()
    noise_adj = compute_noise_adjusted_frequency(cluster_history, pair_co_cluster_freq)

    rows = []
    for (a, b), raw_count in pair_co_cluster_freq.items():
        raw_freq = raw_count / total_windows if total_windows > 0 else 0.0
        adj_freq = noise_adj.get((a, b), 0.0)

        if adj_freq < noise_adj_freq_threshold:
            continue

        row = {
            "Pair": f"{a}-{b}",
            "Ticker_1": a,
            "Ticker_2": b,
            "raw_count": raw_count,
            "raw_freq": raw_freq,
            "noise_adj_freq": adj_freq,
        }

        if sector_map:
            s1 = sector_map.get(a, "Unknown")
            s2 = sector_map.get(b, "Unknown")
            row["sector_1"] = s1
            row["sector_2"] = s2
            row["pair_type"] = "intra-sector" if s1 == s2 else "cross-sector"

        rows.append(row)

    registry = pd.DataFrame(rows)
    if not registry.empty:
        registry = registry.sort_values("noise_adj_freq", ascending=False).reset_index(drop=True)

    return registry


def run_analysis(pair_registry, ts_df):
    """
    Run 5-test validation and backtests for all pairs.

    Uses the scored validation framework instead of the old 3-criteria
    cointegration-based system. Pairs scoring >= 3 are backtested.

    Parameters
    ----------
    pair_registry : DataFrame
        Output of build_pair_registry().
    ts_df : DataFrame
        MultiIndex (Datetime, Ticker) with Price column.

    Returns
    -------
    DataFrame with test results per pair.
    """
    results = []
    n = len(pair_registry)

    for i, (_, row) in enumerate(pair_registry.iterrows()):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Testing pair {i + 1}/{n}: {row['Pair']}")

        entry = {
            "pair": row["Pair"],
            "ticker_a": row["Ticker_1"],
            "ticker_b": row["Ticker_2"],
            "noise_adj_freq": row["noise_adj_freq"],
            "raw_freq": row["raw_freq"],
        }

        # Carry over sector info if present
        for col in ["sector_1", "sector_2", "pair_type"]:
            if col in row.index:
                entry[col] = row[col]

        # Get daily prices and compute spread
        daily_a = get_daily_prices(row["Ticker_1"], ts_df)
        daily_b = get_daily_prices(row["Ticker_2"], ts_df)
        common = daily_a.index.intersection(daily_b.index)

        if len(common) < 60:
            entry.update(_empty_validation_entry())
            results.append(entry)
            continue

        da = daily_a.loc[common]
        db = daily_b.loc[common]

        # Cal/OOS split (67/33)
        split_idx = int(len(common) * 0.67)
        if split_idx < 40 or (len(common) - split_idx) < 20:
            entry.update(_empty_validation_entry())
            results.append(entry)
            continue

        da_cal, da_oos = da.iloc[:split_idx], da.iloc[split_idx:]
        db_cal, db_oos = db.iloc[:split_idx], db.iloc[split_idx:]

        # Hedge ratio on calibration only
        beta, intercept, r_sq = compute_hedge_ratio(da_cal, db_cal, method='ols')
        if np.isnan(beta):
            entry.update(_empty_validation_entry())
            results.append(entry)
            continue

        spread_cal = da_cal - beta * db_cal
        spread_oos = da_oos - beta * db_oos

        # Run 5-test validation
        val = validate_pair_relationship(spread_cal.values, da_cal.values, db_cal.values)

        entry.update({
            "n_daily_obs": len(common),
            "beta": beta,
            "adf_pval": val["adf_pval"],
            "adf_passed": val["adf_passed"],
            "half_life_days": val["half_life_days"],
            "hl_passed": val["hl_passed"],
            "hurst": val["hurst"],
            "hurst_passed": val["hurst_passed"],
            "variance_ratio": val["variance_ratio"],
            "vr_zstat": val["vr_zstat"],
            "vr_passed": val["vr_passed"],
            "rolling_corr_stability": val["rolling_corr_stability"],
            "rc_passed": val["rc_passed"],
            "score": val["score"],
            "classification": val["classification"],
            # Backward compatibility
            "passed": val["score"] >= 4,
            "near_miss": val["score"] == 3,
            "n_criteria_met": val["score"],
        })

        # Backtest pairs scoring >= 3 (moderate or strong)
        if val["score"] >= 3:
            bt = backtest_pair(spread_oos)
            bt["sharpe"] = _sanitize_sharpe(bt["sharpe"], bt["n_trades"])
            entry.update(bt)
            # Store spread series for enhanced backtest
            entry["_spread_cal"] = spread_cal
            entry["_spread_oos"] = spread_oos
            entry["_da_cal"] = da_cal
            entry["_db_cal"] = db_cal
            entry["_da_oos"] = da_oos
            entry["_db_oos"] = db_oos
        else:
            entry.update({
                "total_pnl": np.nan,
                "n_trades": 0,
                "win_rate": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
            })

        results.append(entry)

    return pd.DataFrame(results)


def _empty_validation_entry():
    """Return dict of NaN/default values when pair can't be validated."""
    return {
        "n_daily_obs": 0,
        "beta": np.nan,
        "adf_pval": np.nan,
        "adf_passed": False,
        "half_life_days": np.nan,
        "hl_passed": False,
        "hurst": np.nan,
        "hurst_passed": False,
        "variance_ratio": np.nan,
        "vr_zstat": np.nan,
        "vr_passed": False,
        "rolling_corr_stability": np.nan,
        "rc_passed": False,
        "score": 0,
        "classification": "fail",
        "passed": False,
        "near_miss": False,
        "n_criteria_met": 0,
        "total_pnl": np.nan,
        "n_trades": 0,
        "win_rate": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan,
    }


def pair_type_summary(results_df):
    """
    Compare intra-sector vs cross-sector pair statistics.

    Returns DataFrame with one row per pair_type.
    """
    if "pair_type" not in results_df.columns:
        return pd.DataFrame()

    rows = []
    for ptype, group in results_df.groupby("pair_type"):
        tradeable = group[group["passed"] | group["near_miss"]]
        valid_sharpe = tradeable["sharpe"].dropna()
        rows.append({
            "pair_type": ptype,
            "n_pairs": len(group),
            "n_strong": int((group["classification"] == "strong").sum()),
            "n_moderate": int((group["classification"] == "moderate").sum()),
            "n_passed": int(group["passed"].sum()),
            "n_near_miss": int(group["near_miss"].sum()),
            "avg_score": group["score"].mean(),
            "pass_rate": group["passed"].mean(),
            "avg_sharpe": valid_sharpe.mean() if len(valid_sharpe) > 0 else np.nan,
            "best_sharpe": valid_sharpe.max() if len(valid_sharpe) > 0 else np.nan,
            "profitable_frac": (tradeable["total_pnl"] > 0).mean() if len(tradeable) > 0 else np.nan,
            "avg_noise_adj_freq": group["noise_adj_freq"].mean(),
        })

    return pd.DataFrame(rows)


def sector_pair_breakdown(results_df):
    """
    Count how many pairs exist within and between each sector.

    Returns DataFrame with sector_1, sector_2, count, avg_noise_adj_freq.
    """
    if "sector_1" not in results_df.columns:
        return pd.DataFrame()

    rows = []
    for (s1, s2), group in results_df.groupby(["sector_1", "sector_2"]):
        tradeable = group[group["passed"] | group["near_miss"]]
        valid_sharpe = tradeable["sharpe"].dropna()
        rows.append({
            "sector_1": s1,
            "sector_2": s2,
            "n_pairs": len(group),
            "n_tradeable": len(tradeable),
            "avg_score": group["score"].mean(),
            "avg_noise_adj_freq": group["noise_adj_freq"].mean(),
            "avg_sharpe": valid_sharpe.mean() if len(valid_sharpe) > 0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values("n_pairs", ascending=False).reset_index(drop=True)


def generate_report(results_df):
    """
    Generate text report from cross-sector analysis results.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-SECTOR TRANSIENT CORRELATION ANALYSIS")
    lines.append("5-Test Scored Validation: ADF | Half-Life | Hurst | Variance Ratio | Rolling Corr")
    lines.append(f"(Sharpe only reported for pairs with >= {MIN_TRADES_FOR_SHARPE} OOS trades)")
    lines.append("=" * 80)

    n_total = len(results_df)

    # Score distribution
    lines.append(f"\n--- SCORE DISTRIBUTION ---")
    lines.append(f"Pairs tested:        {n_total}")
    for cls in ["strong", "moderate", "weak", "fail"]:
        n_cls = int((results_df["classification"] == cls).sum())
        lines.append(f"  {cls.capitalize():10s}:      {n_cls}")

    # Test-by-test pass rates
    lines.append(f"\n--- TEST PASS RATES ---")
    for test_col, label in [
        ("adf_passed", "ADF (p<0.10)"),
        ("hl_passed", "Half-life (5-60d)"),
        ("hurst_passed", "Hurst (<0.5)"),
        ("vr_passed", "Variance ratio"),
        ("rc_passed", "Rolling corr stability"),
    ]:
        if test_col in results_df.columns:
            valid = results_df[results_df["n_daily_obs"] > 0]
            if len(valid) > 0:
                rate = valid[test_col].mean()
                lines.append(f"  {label:25s}: {rate:.1%} ({int(valid[test_col].sum())}/{len(valid)})")

    # Backward compat summary
    n_passed = int(results_df["passed"].sum())
    n_near = int(results_df["near_miss"].sum())
    lines.append(f"\n--- LEGACY COMPAT ---")
    lines.append(f"Strong (score>=4):   {n_passed}")
    lines.append(f"Moderate (score==3): {n_near}")

    # Intra vs cross sector
    ptype = pair_type_summary(results_df)
    if not ptype.empty:
        lines.append(f"\n--- INTRA-SECTOR vs CROSS-SECTOR ---")
        lines.append(ptype.to_string(index=False, float_format="%.3f"))

    # Sector pair breakdown
    spb = sector_pair_breakdown(results_df)
    if not spb.empty:
        lines.append(f"\n--- SECTOR PAIR MATRIX (top 15) ---")
        lines.append(spb.head(15).to_string(index=False, float_format="%.3f"))

    # Top pairs by score then Sharpe
    tradeable = results_df[results_df["passed"] | results_df["near_miss"]].copy()
    valid_sharpe = tradeable.dropna(subset=["sharpe"])
    if len(valid_sharpe) > 0:
        lines.append(f"\n--- TOP 10 PAIRS (by score then Sharpe, min {MIN_TRADES_FOR_SHARPE} trades) ---")
        top = valid_sharpe.nlargest(10, ["score", "sharpe"])
        cols = ["pair", "score", "classification", "noise_adj_freq", "adf_pval",
                "half_life_days", "hurst", "variance_ratio", "n_trades", "sharpe", "total_pnl"]
        if "pair_type" in top.columns:
            cols = ["pair_type"] + cols
        lines.append(top[cols].to_string(index=False, float_format="%.4f"))

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
