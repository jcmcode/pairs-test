"""
Unit tests for config.py

Tests that configuration dataclasses are frozen and have correct defaults.
"""

import pytest
from config import (
    ClusteringConfig, FeatureConfig, TransientConfig,
    ValidationConfig, BacktestConfig, FormationConfig,
    PipelineConfig, DEFAULT_CONFIG,
)


class TestConfigDefaults:
    def test_clustering_defaults(self):
        cfg = ClusteringConfig()
        assert cfg.min_samples == 3
        assert cfg.xi == 0.05
        assert cfg.min_cluster_size == 3

    def test_transient_defaults(self):
        cfg = TransientConfig()
        assert cfg.calibration == 20
        assert cfg.exploitation == 40
        assert cfg.entry_z == 2.0

    def test_backtest_defaults(self):
        cfg = BacktestConfig()
        assert cfg.cal_frac == 0.67
        assert cfg.hl_min == 5.0
        assert cfg.hl_max == 60.0

    def test_pipeline_aggregation(self):
        cfg = PipelineConfig()
        assert cfg.clustering.min_samples == 3
        assert cfg.transient.entry_z == 2.0
        assert cfg.backtest.cal_frac == 0.67

    def test_default_config(self):
        assert DEFAULT_CONFIG is not None
        assert DEFAULT_CONFIG.clustering.min_samples == 3


class TestConfigFrozen:
    def test_cannot_mutate(self):
        cfg = ClusteringConfig()
        with pytest.raises(AttributeError):
            cfg.min_samples = 10

    def test_custom_values(self):
        cfg = TransientConfig(calibration=30, entry_z=1.5)
        assert cfg.calibration == 30
        assert cfg.entry_z == 1.5
        # Other fields keep defaults
        assert cfg.exploitation == 40
