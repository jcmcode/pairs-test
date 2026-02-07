import numpy as np
import pandas as pd

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
    return float((w[a].values == w[b].values).mean())

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
    Uses z-score computed on the same window W (no optimization).
    Event: |z| >= z_entry
    Bounce: |z| shrinks by >= improve within next horizon steps.
    """
    if len(spread) < W + horizon + 2:
        return np.nan

    s = spread.iloc[-(W + horizon):].copy()
    mu = s.rolling(W).mean()
    sd = s.rolling(W).std(ddof=1)
    z = (s - mu) / sd

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
