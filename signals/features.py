"""
Feature engineering for transient regime detection.

Extracted from optics-clustering.ipynb. Computes 9 features per ticker
per timestamp: short/medium volatility, market/sector beta, RSI, momentum,
and regime shift indicators.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from config import FeatureConfig, DEFAULT_CONFIG


SEMICONDUCTOR_TICKERS = [
    "^GSPC",
    "NVDA", "TSM", "AVGO", "AMD", "INTC", "MU", "TXN", "QCOM", "ADI", "MCHP",
    "ASML", "AMAT", "LRCX", "KLAC", "TER", "ENTG", "NVMI", "TOELY",
    "ON", "NXPI", "STM", "LSCC", "MPWR", "QRVO", "SWKS", "ALAB", "CRDO",
    "ARM", "SNPS", "CDNS", "CEVA",
    "WDC", "STX",
    "GFS", "MRVL", "MTSI", "POWI", "SMTC", "VICR", "CAMT",
]


def fetch_data(tickers=None, period="252d", interval="1h"):
    """
    Fetch hourly price data via yfinance.

    Returns a DataFrame with tickers as columns, datetime index.
    Forward-fills within the same trading day only (avoids false overnight correlations).
    """
    if tickers is None:
        tickers = SEMICONDUCTOR_TICKERS

    data = yf.download(
        tickers=tickers, period=period, interval=interval,
        group_by='ticker', auto_adjust=True, threads=True,
    )

    price_series = []
    for s in tickers:
        try:
            if s in data:
                series = data[s]['Close']
                series.name = s
                price_series.append(series)
        except Exception:
            pass

    if not price_series:
        return pd.DataFrame()

    df = pd.concat(price_series, axis=1)
    df = df.groupby(df.index.date).apply(lambda g: g.ffill()).droplevel(0)
    return df


def _calculate_rsi(data, window=70):
    """RSI calculation using Wilder's smoothing (EMA with alpha=1/window)."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_features(df_prices, cfg: FeatureConfig = None):
    """
    Compute per-ticker, per-timestamp features from a price DataFrame.

    Parameters
    ----------
    df_prices : DataFrame
        Columns are tickers (including ^GSPC for market), datetime index.
    cfg : FeatureConfig
        Feature engineering parameters.

    Returns
    -------
    ts_df : DataFrame
        MultiIndex (Datetime, Ticker) with feature columns + Price + Returns.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.features

    returns_df = df_prices.pct_change().dropna()
    if '^GSPC' not in returns_df.columns:
        raise ValueError("^GSPC (S&P 500) required for market beta calculation")

    market_returns = returns_df['^GSPC']
    non_spx = returns_df.drop(columns=['^GSPC'], errors='ignore').dropna(axis=1, how='all')

    # Short-term features
    vol_short = returns_df.rolling(window=cfg.window_short).std()

    cov_mkt_short = returns_df.rolling(window=cfg.window_short).cov(market_returns)
    mkt_var_short = market_returns.rolling(window=cfg.window_short).var()
    beta_spx_short = cov_mkt_short.divide(mkt_var_short, axis=0)

    sector_sum = non_spx.sum(axis=1)
    n_stocks = non_spx.count(axis=1)

    beta_sector_short = pd.DataFrame(index=returns_df.index, columns=non_spx.columns)
    for ticker in non_spx.columns:
        loo_sector = (sector_sum - non_spx[ticker].fillna(0)) / (n_stocks - 1).clip(lower=1)
        cov = returns_df[ticker].rolling(window=cfg.window_short).cov(loo_sector)
        var = loo_sector.rolling(window=cfg.window_short).var()
        beta_sector_short[ticker] = cov / var

    # Medium-term features
    vol_medium = returns_df.rolling(window=cfg.window_medium).std()

    cov_mkt_medium = returns_df.rolling(window=cfg.window_medium).cov(market_returns)
    mkt_var_medium = market_returns.rolling(window=cfg.window_medium).var()
    beta_spx_medium = cov_mkt_medium.divide(mkt_var_medium, axis=0)

    beta_sector_medium = pd.DataFrame(index=returns_df.index, columns=non_spx.columns)
    for ticker in non_spx.columns:
        loo_sector = (sector_sum - non_spx[ticker].fillna(0)) / (n_stocks - 1).clip(lower=1)
        cov = returns_df[ticker].rolling(window=cfg.window_medium).cov(loo_sector)
        var = loo_sector.rolling(window=cfg.window_medium).var()
        beta_sector_medium[ticker] = cov / var

    # Instantaneous features
    rsi_df = df_prices.apply(_calculate_rsi, window=cfg.rsi_window)
    momentum_5h = df_prices.pct_change(periods=cfg.momentum_period)

    # Regime shift indicators
    vol_regime_shift = (vol_short - vol_medium) / vol_medium
    beta_spx_regime_shift = beta_spx_short - beta_spx_medium
    beta_sector_regime_shift = beta_sector_short - beta_sector_medium

    # Assemble MultiIndex DataFrame
    ts_data = []
    tickers = [t for t in df_prices.columns if t != '^GSPC']

    for ticker in tickers:
        if ticker not in non_spx.columns:
            continue
        temp = pd.DataFrame({
            'Price': df_prices[ticker],
            'Returns': returns_df[ticker],
            'Vol_Short': vol_short[ticker],
            'Beta_SPX_Short': beta_spx_short[ticker],
            'Beta_Sector_Short': beta_sector_short[ticker],
            'Vol_Medium': vol_medium[ticker],
            'Beta_SPX_Medium': beta_spx_medium[ticker],
            'Beta_Sector_Medium': beta_sector_medium[ticker],
            'RSI': rsi_df[ticker],
            'Momentum_5H': momentum_5h[ticker],
            'Vol_Regime_Shift': vol_regime_shift[ticker],
            'Beta_SPX_Regime_Shift': beta_spx_regime_shift[ticker],
            'Beta_Sector_Regime_Shift': beta_sector_regime_shift[ticker],
        }, index=df_prices.index)

        temp['Ticker'] = ticker
        ts_data.append(temp)

    if not ts_data:
        return pd.DataFrame()

    ts_df = pd.concat(ts_data).reset_index().set_index(['Datetime', 'Ticker'])
    ts_df = ts_df.dropna()
    ts_df = ts_df[~ts_df.index.duplicated(keep='first')]
    return ts_df
