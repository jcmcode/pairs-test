"""
yfscreen wrapper: filter construction, screen execution, universe selection.

yfscreen API:
  filters = [["eq", ["region", "us"]], ["gt", ["intradaymarketcap", 2e9]], ...]
  query = yfs.create_query(filters)
  payload = yfs.create_payload("equity", query, size=250)
  data = yfs.get_data(payload)
"""

import yfscreen as yfs
import pandas as pd

from screener.config import ScreenerConfig


def explore_available_filters():
    """Print all available yfscreen equity filter fields for runtime discovery."""
    df = yfs.data_filters
    equity = df[df["sec_type"] == "equity"]
    print(f"Available equity filters ({len(equity)}):")
    for _, row in equity.iterrows():
        print(f"  {row['name']:45s}  field={row['field']:40s}  type={row['python']}")
    return equity


def build_screen_filters(cfg: ScreenerConfig, sector: str):
    """
    Convert config + sector into a yfscreen filter list.

    Returns a list of lists in yfscreen format:
      [["operator", ["field", value]], ...]
    """
    filters = [
        ["eq", ["region", cfg.region]],
        ["gt", ["intradaymarketcap", cfg.min_market_cap]],
        ["gt", ["avgdailyvol3m", cfg.min_daily_volume]],
        ["gt", ["intradayprice", cfg.min_price]],
        ["eq", ["sector", sector]],
    ]

    if cfg.require_positive_ebitda:
        filters.append(["gt", ["ebitda.lasttwelvemonths", 0]])

    return filters


def run_screen(cfg: ScreenerConfig, sector: str):
    """
    Execute yfscreen query for one sector.

    Returns DataFrame of screen results or empty DataFrame on failure.
    """
    filters = build_screen_filters(cfg, sector)
    try:
        query = yfs.create_query(filters)
        payload = yfs.create_payload("equity", query, size=cfg.max_universe_size)
        results = yfs.get_data(payload)
        if results is not None and not results.empty:
            results["sector_screen"] = sector
            return results
    except Exception as e:
        print(f"  Screen failed for {sector}: {e}")

    return pd.DataFrame()


def select_universe(screen_results: pd.DataFrame, cfg: ScreenerConfig):
    """
    Post-process screen results into a ticker list.

    Sorts by market cap (descending), caps at max_universe_size,
    and prepends ^GSPC (required for market beta computation).
    """
    if screen_results.empty:
        return []

    # Sort by market cap descending (column name may vary)
    cap_col = None
    for candidate in ["marketCap", "intradaymarketcap", "Market Cap"]:
        if candidate in screen_results.columns:
            cap_col = candidate
            break

    if cap_col:
        screen_results = screen_results.sort_values(cap_col, ascending=False)

    # Extract ticker symbols
    ticker_col = None
    for candidate in ["symbol", "Symbol", "ticker", "Ticker"]:
        if candidate in screen_results.columns:
            ticker_col = candidate
            break

    if ticker_col is None:
        tickers = screen_results.index.tolist()
    else:
        tickers = screen_results[ticker_col].tolist()

    # Cap universe size
    tickers = tickers[: cfg.max_universe_size]

    # Prepend ^GSPC for market beta
    if "^GSPC" not in tickers:
        tickers = ["^GSPC"] + tickers

    return tickers


def screen_all_sectors(cfg: ScreenerConfig = None):
    """
    Run screening for all configured sectors.

    Returns dict mapping sector name -> list of tickers (excluding ^GSPC).
    """
    if cfg is None:
        cfg = ScreenerConfig()

    universes = {}
    for sector in cfg.sectors:
        print(f"\nScreening {sector}...")
        results = run_screen(cfg, sector)

        if results.empty:
            print(f"  {sector}: no results")
            continue

        tickers = select_universe(results, cfg)
        n_stocks = len([t for t in tickers if t != "^GSPC"])

        if n_stocks < cfg.min_universe_size:
            print(f"  {sector}: only {n_stocks} stocks, skipping (min={cfg.min_universe_size})")
            continue

        universes[sector] = tickers
        print(f"  {sector}: {n_stocks} stocks selected")

    return universes


def screen_combined_universe(cfg: ScreenerConfig = None):
    """
    Screen all sectors and combine into a single universe for cross-sector clustering.

    Returns:
        combined_tickers: list of all tickers (with ^GSPC once)
        sector_map: dict mapping ticker -> sector name
        per_sector: dict mapping sector -> list of tickers (for reference)
    """
    if cfg is None:
        cfg = ScreenerConfig()

    per_sector = screen_all_sectors(cfg)

    sector_map = {}
    all_tickers = set()
    for sector, tickers in per_sector.items():
        for t in tickers:
            if t != "^GSPC":
                sector_map[t] = sector
                all_tickers.add(t)

    combined_tickers = ["^GSPC"] + sorted(all_tickers)

    print(f"\n{'='*50}")
    print(f"Combined universe: {len(all_tickers)} stocks from {len(per_sector)} sectors")
    for sector, tickers in per_sector.items():
        n = len([t for t in tickers if t != "^GSPC"])
        print(f"  {sector}: {n}")

    return combined_tickers, sector_map, per_sector
