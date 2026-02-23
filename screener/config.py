"""
Screening configuration for broad-universe testing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenerConfig:
    """Parameters for systematic universe construction."""
    region: str = "us"
    min_market_cap: float = 2_000_000_000
    min_daily_volume: int = 5_000_000
    min_price: float = 5.0
    require_positive_ebitda: bool = True
    max_universe_size: int = 80
    min_universe_size: int = 15
    min_data_coverage: float = 0.80
    data_period: str = "252d"
    data_interval: str = "1h"
    sectors: tuple = (
        "Technology",
        "Healthcare",
        "Energy",
        "Financial Services",
        "Industrials",
    )
