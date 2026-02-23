"""
Data fetching, quality validation, sector mapping, and caching.
"""

import os
import pickle

import numpy as np
import pandas as pd

from screener.config import ScreenerConfig

# Import fetch_data from the main signals module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from signals.features import fetch_data


def fetch_and_validate_prices(tickers, cfg: ScreenerConfig = None):
    """
    Fetch price data and apply quality filters.

    Drops tickers with:
    - Less than cfg.min_data_coverage non-null fraction
    - More than 5% zero-volume hours (proxy for illiquidity)

    Returns (df_prices, dropped_report) where dropped_report is a list of
    (ticker, reason) tuples for tickers that were removed.
    """
    if cfg is None:
        cfg = ScreenerConfig()

    df_prices = fetch_data(tickers=tickers, period=cfg.data_period, interval=cfg.data_interval)

    if df_prices.empty:
        return df_prices, [("ALL", "fetch returned empty")]

    dropped = []
    keep_cols = []

    for col in df_prices.columns:
        if col == "^GSPC":
            keep_cols.append(col)
            continue

        series = df_prices[col]
        coverage = series.notna().mean()

        if coverage < cfg.min_data_coverage:
            dropped.append((col, f"low coverage ({coverage:.1%})"))
            continue

        # Check for excessive zero values (proxy for zero-volume)
        zero_frac = (series == 0).mean()
        if zero_frac > 0.05:
            dropped.append((col, f"excessive zeros ({zero_frac:.1%})"))
            continue

        keep_cols.append(col)

    df_prices = df_prices[keep_cols]
    return df_prices, dropped


def build_sector_map(screen_results: pd.DataFrame):
    """
    Extract ticker -> sector/industry mapping from screen results.

    Returns dict mapping ticker -> {"sector": ..., "industry": ...}.
    """
    sector_map = {}

    # Detect column names
    ticker_col = None
    for candidate in ["symbol", "Symbol", "ticker", "Ticker"]:
        if candidate in screen_results.columns:
            ticker_col = candidate
            break

    sector_col = None
    for candidate in ["sector_screen", "sector", "Sector"]:
        if candidate in screen_results.columns:
            sector_col = candidate
            break

    industry_col = None
    for candidate in ["industry", "Industry"]:
        if candidate in screen_results.columns:
            industry_col = candidate
            break

    if ticker_col is None:
        tickers = screen_results.index.tolist()
    else:
        tickers = screen_results[ticker_col].tolist()

    for i, ticker in enumerate(tickers):
        entry = {}
        if sector_col:
            entry["sector"] = screen_results.iloc[i].get(sector_col, "Unknown")
        if industry_col:
            entry["industry"] = screen_results.iloc[i].get(industry_col, "Unknown")
        sector_map[ticker] = entry

    return sector_map


def cache_universe(name, tickers, prices, sector_map=None, path=None):
    """
    Save universe data (tickers + prices + optional sector map) as pickle.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            name.lower().replace(" ", "_"),
        )

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "tickers.pkl"), "wb") as f:
        pickle.dump(tickers, f)

    with open(os.path.join(path, "prices.pkl"), "wb") as f:
        pickle.dump(prices, f)

    if sector_map is not None:
        with open(os.path.join(path, "sector_map.pkl"), "wb") as f:
            pickle.dump(sector_map, f)

    print(f"  Cached {name}: {len(tickers)} tickers, prices shape {prices.shape}")


def load_cached_universe(name, path=None):
    """
    Load cached universe data.

    Returns (tickers, prices, sector_map) or (None, None, None) if not found.
    sector_map may be None if it wasn't cached.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            name.lower().replace(" ", "_"),
        )

    tickers_path = os.path.join(path, "tickers.pkl")
    prices_path = os.path.join(path, "prices.pkl")
    sector_map_path = os.path.join(path, "sector_map.pkl")

    if not os.path.exists(tickers_path) or not os.path.exists(prices_path):
        return None, None, None

    with open(tickers_path, "rb") as f:
        tickers = pickle.load(f)

    with open(prices_path, "rb") as f:
        prices = pickle.load(f)

    sector_map = None
    if os.path.exists(sector_map_path):
        with open(sector_map_path, "rb") as f:
            sector_map = pickle.load(f)

    return tickers, prices, sector_map
