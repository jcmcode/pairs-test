import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS

# -------------------------
# Cluster helpers
# -------------------------

def make_clusters_wide(cluster_history: pd.DataFrame) -> pd.DataFrame:
    ch = cluster_history.copy()
    ch["Datetime"] = pd.to_datetime(ch["Datetime"])
    return ch.pivot_table(index="Datetime", columns="Ticker", values="Cluster_ID", aggfunc="last").sort_index()

def cluster_persistence(clusters_wide: pd.DataFrame, a: str, b: str, t: pd.Timestamp, H: int = 5) -> float:
    if t not in clusters_wide.index:
        return np.nan
    loc = clusters_wide.index.get_loc(t)
    start = max(0, loc - (H - 1))
    w = clusters_wide.iloc[start:loc+1][[a, b]].dropna()
    if len(w) == 0:
        return np.nan
    same_cluster = w[a].values == w[b].values
    not_noise = (w[a].values != -1) & (w[b].values != -1)
    return float((same_cluster & not_noise).mean())

# -------------------------
# Spread + tests
# -------------------------

def compute_spread(df_prices: pd.DataFrame, a: str, b: str, beta: float) -> pd.Series:
    # Matches your current hedge.py logic: spread = A - beta*B
    s = df_prices[a] - beta * df_prices[b]
    return s.dropna()

def lag1_autocorr(x: pd.Series) -> float:
    if len(x) < 3:
        return np.nan
    return float(x.autocorr(lag=1))

def ar1_phi(x: pd.Series) -> float:
    # x_t = c + phi*x_{t-1} + eps
    if len(x) < 4:
        return np.nan
    y = x.iloc[1:].values
    xlag = x.iloc[:-1].values
    X = np.column_stack([np.ones(len(xlag)), xlag])
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(coef[1])
    except Exception:
        return np.nan

def bounce_rate(spread: pd.Series, W: int, z_entry: float, horizon: int = 2, improve: float = 0.30) -> float:
    """
    Uses z-score computed on a rolling window W over the full spread.
    Event: |z| >= z_entry
    Bounce: |z| shrinks by >= improve within next horizon steps.
    """
    if len(spread) < W + horizon + 2:
        return np.nan

    mu = spread.rolling(W).mean()
    sd = spread.rolling(W).std(ddof=1)
    z = (spread - mu) / sd

    events = 0
    bounces = 0
    for i in range(len(z) - horizon):
        z0 = z.iloc[i]
        if not np.isfinite(z0) or abs(z0) < z_entry:
            continue
        events += 1
        target = (1.0 - improve) * abs(z0)
        future = z.iloc[i+1:i+1+horizon].abs().values
        if np.any(future <= target):
            bounces += 1

    if events == 0:
        return np.nan
    return float(bounces / events)

# -------------------------
# Main: validate a pair at time t
# -------------------------

def validate_pair(
    prices: pd.DataFrame,
    cluster_history: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    beta: float,
    t: pd.Timestamp,
    H_persist: int = 5,
    W_spread: int = 15,
    z_entry: float = 2.0,   # match your signals.py default entry
    bounce_horizon: int = 2,
    bounce_improve: float = 0.30,
    drop_noise_now: bool = True
) -> dict:
    """
    Returns a dict of validation metrics for a single pair at time t.
    """
    clusters_wide = make_clusters_wide(cluster_history)

    if t not in prices.index or t not in clusters_wide.index:
        return {}

    # Optional: drop if either is noise in current hour
    if drop_noise_now:
        c1 = clusters_wide.at[t, ticker1] if ticker1 in clusters_wide.columns else np.nan
        c2 = clusters_wide.at[t, ticker2] if ticker2 in clusters_wide.columns else np.nan
        if pd.isna(c1) or pd.isna(c2) or c1 == -1 or c2 == -1:
            return {}

    persist = cluster_persistence(clusters_wide, ticker1, ticker2, t, H=H_persist)

    spread = compute_spread(prices.loc[:t, [ticker1, ticker2]].dropna(), ticker1, ticker2, beta=beta)
    if len(spread) < W_spread:
        return {}

    w = spread.iloc[-W_spread:]
    ac1 = lag1_autocorr(w)
    phi = ar1_phi(w)
    br = bounce_rate(spread, W=W_spread, z_entry=z_entry, horizon=bounce_horizon, improve=bounce_improve)

    return {
        "time": t,
        "ticker1": ticker1,
        "ticker2": ticker2,
        "beta": beta,
        "cluster_persist": float(persist) if np.isfinite(persist) else np.nan,
        "spread_ac1": ac1,
        "ar1_phi": phi,
        "bounce_rate": br,
    }


# -------------------------
# Hedge ratio estimation
# -------------------------

def compute_hedge_ratio(prices_a, prices_b, method='ols'):
    """
    Estimate hedge ratio between two price series.

    Parameters
    ----------
    prices_a, prices_b : array-like
        Aligned price series (dependent and independent).
    method : str
        'ols' (default), 'tls' (total least squares), or 'kalman'.

    Returns
    -------
    (beta, intercept, r_squared) : tuple of float
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return (np.nan, np.nan, np.nan)

    if method == 'ols':
        X = np.column_stack([b, np.ones(len(b))])
        coef, _, _, _ = np.linalg.lstsq(X, a, rcond=None)
        beta, intercept = coef[0], coef[1]

    elif method == 'tls':
        ab = np.column_stack([b, a])
        ab_centered = ab - ab.mean(axis=0)
        _, _, Vt = np.linalg.svd(ab_centered, full_matrices=False)
        beta = -Vt[-1, 0] / Vt[-1, 1]
        intercept = a.mean() - beta * b.mean()

    elif method == 'kalman':
        from pykalman import KalmanFilter
        obs = a
        n = len(obs)
        state_means = np.zeros((n, 2))
        state_cov = np.eye(2)
        state = np.array([0.0, 0.0])
        for i in range(n):
            obs_mat = np.array([[b[i], 1.0]])
            kf_step = KalmanFilter(
                transition_matrices=np.eye(2),
                observation_matrices=obs_mat,
                initial_state_mean=state,
                initial_state_covariance=state_cov,
                observation_covariance=np.array([[1.0]]),
                transition_covariance=0.01 * np.eye(2),
            )
            state, state_cov = kf_step.filter_update(state, state_cov, obs[i], observation_matrix=obs_mat)
            state_means[i] = state
        beta = state_means[-1, 0]
        intercept = state_means[-1, 1]

    else:
        raise ValueError(f"Unknown method: {method}")

    predicted = beta * b + intercept
    ss_res = np.sum((a - predicted) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return (float(beta), float(intercept), float(r_squared))


def hedge_ratio_drift(beta_cal, beta_exploit):
    """
    Measure how much the hedge ratio drifted between calibration and
    exploitation windows.

    Returns abs(beta_exploit - beta_cal) / max(abs(beta_cal), 0.1).
    The denominator floor of 0.1 prevents inflated drift when beta_cal
    is small.
    """
    if not np.isfinite(beta_cal) or not np.isfinite(beta_exploit):
        return np.nan
    return abs(beta_exploit - beta_cal) / max(abs(beta_cal), 0.1)


def spread_cv_normalized(spread, prices_a, prices_b):
    """
    Spread coefficient of variation normalized by average price level
    instead of spread mean (avoids blow-up when spread ~ 0).

    Returns spread.std() / avg_price_level.
    """
    s = np.asarray(spread, dtype=float)
    avg_price = (np.nanmean(prices_a) + np.nanmean(prices_b)) / 2.0
    if avg_price == 0:
        return np.nan
    return float(np.nanstd(s, ddof=1) / avg_price)


def half_life(spread):
    """
    AR(1) half-life of mean reversion: -log(2) / beta
    where beta is the OLS coefficient from regressing
    delta_spread on lagged spread.

    Returns half-life in the same time units as the spread index,
    or np.inf if the spread is not mean-reverting.
    """
    s = pd.Series(spread).dropna()
    if len(s) < 10:
        return np.nan
    s_lag = s.shift(1).dropna()
    s_diff = s.diff().dropna()
    common = s_lag.index.intersection(s_diff.index)
    if len(common) < 5:
        return np.nan
    x = s_lag.loc[common].values.reshape(-1, 1)
    y = s_diff.loc[common].values
    coef = np.linalg.lstsq(
        np.column_stack([x, np.ones(len(x))]), y, rcond=None
    )[0]
    beta = coef[0]
    if beta >= 0:
        return np.inf
    return float(-np.log(2) / beta)


# -------------------------
# Signal generation & P&L
# -------------------------

def zscore_signals(spread, lookback, entry_z=2.0, exit_z=0.5):
    """
    Generate trading signals from rolling z-score of the spread.

    Returns DataFrame with columns: z_score, long_entry, short_entry, exit.
    """
    s = pd.Series(spread)
    mu = s.rolling(lookback).mean()
    sd = s.rolling(lookback).std(ddof=1)
    z = pd.Series(np.where(sd > 0, (s - mu) / sd, np.nan), index=s.index)

    df = pd.DataFrame({'z_score': z}, index=s.index)
    df['long_entry'] = z <= -entry_z
    df['short_entry'] = z >= entry_z
    df['exit'] = z.abs() <= exit_z
    return df


def simulate_spread_pnl(spread, signals_df, cost_per_trade=0.0):
    """
    Simulate P&L from entering at +/-entry_z and exiting at exit_z.

    Parameters
    ----------
    spread : Series
        The spread to trade.
    signals_df : DataFrame
        Output of zscore_signals (columns: long_entry, short_entry, exit).
    cost_per_trade : float
        Round-trip cost deducted per trade.

    Returns
    -------
    dict with total_pnl, n_trades, win_rate, sharpe, max_drawdown, pnl_series.
    """
    s = pd.Series(spread).reindex(signals_df.index)
    position = 0  # +1 long, -1 short, 0 flat
    entry_price = 0.0
    trades = []
    pnl_curve = []
    cumulative = 0.0

    for i in range(len(signals_df)):
        sig = signals_df.iloc[i]
        price = s.iloc[i]
        if np.isnan(price):
            pnl_curve.append(cumulative)
            continue

        if position == 0:
            if sig['long_entry']:
                position = 1
                entry_price = price
            elif sig['short_entry']:
                position = -1
                entry_price = price
        elif sig['exit'] and position != 0:
            pnl = position * (price - entry_price) - cost_per_trade
            trades.append(pnl)
            cumulative += pnl
            position = 0
            entry_price = 0.0

        pnl_curve.append(cumulative)

    # Close any remaining open position at end
    if position != 0:
        final_price = s.iloc[-1]
        if not np.isnan(final_price):
            pnl = position * (final_price - entry_price) - cost_per_trade
            trades.append(pnl)
            cumulative += pnl
            pnl_curve[-1] = cumulative

    pnl_series = pd.Series(pnl_curve, index=signals_df.index)
    n_trades = len(trades)
    if n_trades == 0:
        return {
            'total_pnl': 0.0,
            'n_trades': 0,
            'win_rate': np.nan,
            'sharpe': np.nan,
            'max_drawdown': 0.0,
            'pnl_series': pnl_series,
        }

    wins = sum(1 for t in trades if t > 0)
    trade_arr = np.array(trades)
    per_trade_sharpe = trade_arr.mean() / trade_arr.std() if trade_arr.std() > 0 else np.nan
    running_max = pnl_series.cummax()
    drawdown = pnl_series - running_max
    max_dd = float(drawdown.min())

    # Annualized Sharpe from daily returns of the cumulative P&L curve
    daily_returns = pnl_series.diff().dropna()
    if len(daily_returns) >= 3 and daily_returns.std() > 0:
        ann_sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
    else:
        ann_sharpe = np.nan

    return {
        'total_pnl': float(cumulative),
        'n_trades': n_trades,
        'win_rate': float(wins / n_trades),
        'sharpe': float(per_trade_sharpe),
        'sharpe_annualized': ann_sharpe,
        'max_drawdown': max_dd,
        'pnl_series': pnl_series,
    }


# -------------------------
# Permutation test (FIX 1)
# -------------------------

def feature_shuffle_permutation_test(
    ts_df,
    features_to_cluster,
    optics_params,
    pair_co_cluster_freq=None,
    total_valid_windows=None,
    n_permutations=30,
    n_sample_timestamps=80,
):
    """
    Feature-shuffling permutation test for co-clustering significance.

    For each permutation: sample timestamps, shuffle feature vectors across
    tickers (breaking the ticker-feature mapping while preserving cross-feature
    correlation structure), run StandardScaler -> PCA -> OPTICS on the shuffled
    features, and count co-clustering occurrences.

    Parameters
    ----------
    ts_df : DataFrame
        MultiIndex (Datetime, Ticker) with feature columns.
    features_to_cluster : list of str
        Feature column names used for clustering.
    optics_params : dict
        OPTICS parameters (min_samples, xi, min_cluster_size).
    pair_co_cluster_freq : dict, optional
        Kept for backward compatibility; not used internally.
        Observed counts are recomputed on the sampled timestamps.
    total_valid_windows : int, optional
        Kept for backward compatibility; not used internally.
    n_permutations : int
        Number of permutations to run.
    n_sample_timestamps : int
        Number of timestamps to sample per permutation.

    Returns
    -------
    dict with:
        - pair_zscores: dict mapping pair -> Z-score
        - fraction_significant: fraction of tested pairs significant at
          one-sided p < 0.05 (Z > 1.645)
        - null_counts: dict mapping pair -> list of permuted counts
    """
    all_timestamps = ts_df.index.get_level_values('Datetime').unique()
    rng = np.random.default_rng(42)
    if len(all_timestamps) > n_sample_timestamps:
        sample_ts = rng.choice(all_timestamps, size=n_sample_timestamps, replace=False)
    else:
        sample_ts = all_timestamps

    # Compute observed co-clustering on the SAME sampled timestamps
    obs_counts_sampled = {}
    for ts in sample_ts:
        try:
            snapshot = ts_df.xs(ts, level='Datetime')[features_to_cluster].dropna()
        except KeyError:
            continue
        if len(snapshot) < 5:
            continue
        tickers = snapshot.index.tolist()
        values_obs = snapshot.values.copy()
        scaler_obs = StandardScaler()
        X_scaled_obs = scaler_obs.fit_transform(values_obs)
        pca_obs = PCA(n_components=0.90)
        X_pca_obs = pca_obs.fit_transform(X_scaled_obs)
        optics_obs = OPTICS(**optics_params)
        optics_obs.fit(X_pca_obs)
        for cid in set(optics_obs.labels_):
            if cid == -1:
                continue
            members = sorted([tickers[j] for j in range(len(tickers)) if optics_obs.labels_[j] == cid])
            for s1, s2 in itertools.combinations(members, 2):
                key = (s1, s2)
                obs_counts_sampled[key] = obs_counts_sampled.get(key, 0) + 1

    null_counts = {}

    for perm in range(n_permutations):
        perm_counts = {}
        for ts in sample_ts:
            try:
                snapshot = ts_df.xs(ts, level='Datetime')[features_to_cluster].dropna()
            except KeyError:
                continue
            if len(snapshot) < 5:
                continue

            tickers = snapshot.index.tolist()
            values = snapshot.values.copy()
            # Shuffle rows (feature vectors across tickers)
            rng.shuffle(values)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(values)
            pca = PCA(n_components=0.90)
            X_pca = pca.fit_transform(X_scaled)

            optics = OPTICS(**optics_params)
            optics.fit(X_pca)

            for cid in set(optics.labels_):
                if cid == -1:
                    continue
                members = sorted([tickers[j] for j in range(len(tickers)) if optics.labels_[j] == cid])
                for s1, s2 in itertools.combinations(members, 2):
                    key = (s1, s2)
                    perm_counts[key] = perm_counts.get(key, 0) + 1

        for key, cnt in perm_counts.items():
            null_counts.setdefault(key, []).append(cnt)

    # Pad missing permutations with 0
    for key in null_counts:
        while len(null_counts[key]) < n_permutations:
            null_counts[key].append(0)

    # Compute Z-scores using observed counts on same sampled timestamps.
    # Only score pairs that were actually observed in the sampled data —
    # pairs from pair_co_cluster_freq that didn't appear in the sample
    # would get obs_val=0 and misleading z-scores.
    pair_zscores = {}
    for pair in obs_counts_sampled:
        obs_val = obs_counts_sampled[pair]
        null_vals = np.array(null_counts.get(pair, [0] * n_permutations), dtype=float)
        null_mean = null_vals.mean()
        null_std = null_vals.std()
        if null_std > 0:
            z = (obs_val - null_mean) / null_std
        else:
            z = 0.0 if obs_val <= null_mean else np.inf
        pair_zscores[pair] = float(z)

    # One-sided test: Z > 1.645 corresponds to p < 0.05
    n_significant = sum(1 for z in pair_zscores.values() if z > 1.645)
    frac_sig = n_significant / len(pair_zscores) if pair_zscores else 0.0

    return {
        'pair_zscores': pair_zscores,
        'fraction_significant': frac_sig,
        'null_counts': null_counts,
    }
