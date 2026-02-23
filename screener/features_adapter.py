"""
Thin wrapper around signals.features for extensibility.

Phase 1: direct passthrough to compute_features().
Exists as an extensibility point for Phase 2 sector-aware betas if needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from signals.features import compute_features
from config import FeatureConfig, DEFAULT_CONFIG


def compute_features_for_universe(df_prices, cfg: FeatureConfig = None):
    """
    Compute features for a screened universe.

    Currently a direct passthrough to signals.features.compute_features().
    Separated so that sector-specific feature adjustments (e.g., sector beta
    computed within the screened universe rather than leave-one-out of the
    full ticker list) can be added without modifying the core module.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.features

    return compute_features(df_prices, cfg=cfg)
