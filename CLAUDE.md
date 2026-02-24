# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a two-phase transient correlation detection research project. The goal is to identify short-term relationship formations between assets using clustering algorithms, validate those relationships with statistical tests, and backtest trading strategies on discovered pairs.

Unlike traditional pairs trading (which seeks cointegrated pairs with mean reversion), this project looks for **transient correlations** that form on shorter time frames. When two assets cluster together, it signals a relationship has formed; the validation module then tests the strength and persistence of that relationship.

### Two Phases

- **Phase 1 (Semiconductors):** Proof of concept on 40 hand-picked semiconductor tickers using three clustering algorithms (OPTICS, DBSCAN, KMeans). Results in `research/` notebooks.
- **Phase 2 (Cross-Sector):** Extension to 142 systematically screened tickers across 5 sectors. Uses a 5-test scored validation framework (replacing cointegration), enhanced backtesting with Kalman hedge ratios, and permutation testing. Results in `screener/notebooks/`.

### Architecture

1. **Validation Module** (`validation/pair_validation.py`): Core statistical testing library
   - Hedge ratio estimation (OLS, Kalman), half-life, Hurst exponent, spread computation
   - `validate_pair()`: computes all metrics for a single pair at a given timestamp
   - `feature_shuffle_permutation_test()`: statistical significance testing

2. **Signals Module** (`signals/`): Feature engineering and clustering
   - `features.py`: 9-feature computation per ticker (volatility, beta, RSI, regime shifts)
   - `detection.py`: Clustering pipeline, formation/dissolution event detection
   - `transient.py`: Transient event validation
   - `stable.py`: Stable pair tracking

3. **Trading Module** (`trading/trading.py`): Backtesting and pair analysis
   - Z-score mean-reversion strategy, walk-forward validation
   - Pair registry construction, noise-adjusted frequency computation

4. **Screener Module** (`screener/`): Phase 2 cross-sector analysis
   - `screening.py` / `universe.py`: 3-layer screening pipeline (liquidity, sector, quality)
   - `analysis.py`: 5-test scored validation (ADF, half-life, Hurst, variance ratio, rolling correlation)
   - `enhanced_backtest.py`: Adaptive z-score optimization, Kalman hedge, transaction costs
   - `notebooks/01-05`: Full analysis pipeline (run in order)

5. **Deliverables** (`deliverables/`): Final outputs
   - `results_summary.md`: Comprehensive results covering both phases
   - `build_presentation.py`: Generates 25-slide presentation
   - `report.tex`: LaTeX technical report

### Key Concepts

- **Transient Correlation**: Two assets that cluster together at the same timestamp. Unlike cointegration, these relationships form and dissolve over time.
- **Noise-Adjusted Frequency**: Co-clustering rate using only timestamps where both tickers are non-noise as denominator. Essential for OPTICS (58% noise rate).
- **5-Test Validation**: ADF, half-life, Hurst, variance ratio, rolling correlation stability. Score 0-5; tradeable at score >= 3.
- **Kalman Terminal Beta**: Kalman filter estimates hedge ratio on calibration data only; fixed beta applied to OOS. Avoids artificial mean-reversion from adaptive updates.

## Development Environment

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, yfinance, scikit-learn, statsmodels, pandas-ta-classic, pykalman, scipy, jupyter, matplotlib, seaborn, plotly, pytest, yfscreen

## Common Commands

- **Run tests**: `pytest tests/`
- **Run notebooks**: `jupyter notebook`
- **Build presentation**: `python3 deliverables/build_presentation.py`
- **Phase 2 pipeline**: Run `screener/notebooks/01` through `05` in order

## Important Notes

- `validate_pair()` expects aligned timestamps between `prices` and `cluster_history` DataFrames
- All validation metrics return `np.nan` if insufficient data
- The `drop_noise_now` parameter filters pairs where either asset has cluster_id == -1
- Cointegration fails on transient correlations by design (0% pass rate) — use the 5-test framework instead
- OPTICS is the primary algorithm: best at detecting variable-density clusters and transient formations
- Kalman spread must use fixed terminal beta, not adaptive updates (see `screener/enhanced_backtest.py`)
- Sharpe ratios require >= 5 OOS trades to be meaningful; pairs with fewer trades get NaN
