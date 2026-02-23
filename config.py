"""
Centralized configuration for all modules.

All hardcoded thresholds are defined here as frozen dataclasses.
Import the defaults or construct custom instances for experimentation.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClusteringConfig:
    """OPTICS clustering parameters."""
    min_samples: int = 3
    xi: float = 0.05
    min_cluster_size: int = 3
    metric: str = 'euclidean'
    max_noise_pct: float = 0.75
    pca_variance_threshold: float = 0.90
    min_stocks_per_snapshot: int = 5


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering parameters."""
    window_short: int = 50       # ~1 week hourly
    window_medium: int = 147     # ~3 weeks hourly
    rsi_window: int = 70         # ~2 weeks hourly
    momentum_period: int = 5
    features_to_cluster: tuple[str, ...] = (
        'Returns', 'Vol_Short', 'Beta_SPX_Short', 'Beta_Sector_Short',
        'RSI', 'Momentum_5H', 'Vol_Regime_Shift',
        'Beta_SPX_Regime_Shift', 'Beta_Sector_Regime_Shift',
    )


@dataclass(frozen=True)
class TransientConfig:
    """Transient pair validation thresholds."""
    calibration: int = 20        # hours for hedge ratio estimation
    exploitation: int = 40       # hours for OOS testing
    execution_lag: int = 2       # hours delay before entering
    corr_threshold: float = 0.70
    cv_threshold: float = 0.03
    hl_max: float = 8.0          # max half-life in hours
    hedge_drift_max: float = 0.20
    entry_z: float = 2.0
    exit_z: float = 0.5


@dataclass(frozen=True)
class ValidationConfig:
    """Pair validation parameters."""
    H_persist: int = 5           # cluster persistence rolling window
    W_spread: int = 15           # spread rolling window
    z_entry: float = 2.0
    bounce_horizon: int = 2
    bounce_improve: float = 0.30
    drop_noise_now: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest and trading parameters."""
    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    cost_per_trade: float = 0.0
    cal_frac: float = 0.67       # calibration fraction for cal/OOS split
    min_daily_obs: int = 60
    min_cal_obs: int = 40
    min_oos_obs: int = 20
    hl_min: float = 5.0
    hl_max: float = 60.0
    hurst_max: float = 0.5
    coint_pval_max: float = 0.05


@dataclass(frozen=True)
class FormationConfig:
    """Formation/dissolution detection parameters."""
    min_gap_hours: int = 5       # min gap before counting new formation
    min_episode_hours: int = 5   # min duration for actionable formation
    duration_target_hours: int = 10  # for duration prediction model


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregate config for the full real-time pipeline."""
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    transient: TransientConfig = field(default_factory=TransientConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    formation: FormationConfig = field(default_factory=FormationConfig)
    noise_adj_freq_threshold: float = 0.15
    max_pairs: int = 50
    max_events_per_pair: int = 20


# Default instances for convenience
DEFAULT_CONFIG = PipelineConfig()
