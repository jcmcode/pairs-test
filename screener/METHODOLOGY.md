# Screener Methodology

## Objective

The original transient correlation detection methodology was developed and validated on 40 hand-picked semiconductor tickers. This screener module extends the analysis to a systematically constructed multi-sector universe, testing whether the approach generalizes beyond semiconductors.

**Research question:** Do transient correlations detected by OPTICS clustering produce tradeable pairs across different market sectors? Are cross-sector pairs stronger than intra-sector pairs?

## Key Design Decision: Cross-Sector Clustering

All screened tickers are pooled into a **single combined universe** and clustered together. Pairs are classified as intra-sector or cross-sector **after** clustering, not before. This means:

- The clustering algorithm discovers relationships without sector boundaries
- If two stocks from different sectors (e.g., Energy + Technology) cluster together, that's a genuine cross-sector finding
- If intra-sector pairs dominate, that's an empirical result rather than a methodological constraint

This contrasts with clustering within each sector separately, which would prevent cross-sector discovery.

## Universe Construction

### Three-Layer Screening Pipeline

**Layer 1: Liquidity & Investability**
- Region: US
- Market cap > $2B (avoids micro-caps with unreliable price data)
- Average daily volume > 5M shares (ensures executable trade sizes)
- Price > $5 (avoids penny stocks and their noise-driven dynamics)

**Layer 2: Sector Grouping**
Screens are run per GICS sector to build the combined universe:
- Technology
- Healthcare
- Energy
- Financial Services
- Industrials

Each sector produces ~40-80 stocks after filtering. These are then pooled into one universe.

**Layer 3: Fundamental Quality**
- Positive EBITDA (removes speculative / pre-revenue companies whose price action is noise-driven)

### Data Quality Validation

After screening, price data is fetched and filtered:
- Minimum 80% non-null data coverage (drops tickers with too many gaps)
- Maximum 5% zero-price observations (proxy for illiquidity / data issues)
- ^GSPC (S&P 500) is always included for market beta computation

## Data Pipeline

1. **Screen** all sectors via `yfscreen`, combine into one ticker list with sector map
2. **Fetch prices** via `yfinance` (hourly, ~252 trading days)
3. **Compute 9 features** per ticker per timestamp using `signals.features.compute_features()`:
   - Returns, Vol_Short, Beta_SPX_Short, Beta_Sector_Short
   - RSI, Momentum_5H
   - Vol_Regime_Shift, Beta_SPX_Regime_Shift, Beta_Sector_Regime_Shift
4. **Cache** combined universe (tickers, prices, sector_map) to `screener/data/combined/`

## Clustering Methodology

Uses the same OPTICS pipeline as the semiconductor analysis, applied to the full cross-sector universe:

1. **Per-timestamp clustering** via `signals.detection.build_cluster_history()`:
   - StandardScaler → PCA (90% variance) → OPTICS
   - OPTICS parameters: min_samples=3, xi=0.05, min_cluster_size=3
   - Noise threshold: skip snapshots with >75% noise
2. **Formation/dissolution detection** via `signals.detection.detect_formation_events()`:
   - Minimum 5-hour gap before counting a new formation
   - Minimum 5-hour duration for an actionable formation
3. **Co-cluster frequency** via `signals.detection.compute_co_cluster_freq()`:
   - Noise-adjusted: denominator counts only timestamps where both tickers are non-noise

After clustering, each pair is tagged with its sector classification:
- **Intra-sector**: both tickers from the same GICS sector
- **Cross-sector**: tickers from different GICS sectors

## Pair Registry

Pairs are filtered by noise-adjusted co-cluster frequency (threshold: 0.08) and registered with:
- Raw and noise-adjusted frequency
- Sector labels and pair type (intra/cross)
- 5-test validation score and classification
- Backtest performance (Sharpe, P&L, win rate)

**Frequency threshold:** Lowered from 0.15 to 0.08 to include cross-sector pairs. At 0.15, zero cross-sector pairs survive — the best cross-sector pair has noise-adjusted frequency ~0.104. At 0.08, ~109 pairs pass including cross-sector discoveries.

**Sharpe ratio guard:** Pairs with fewer than 5 out-of-sample trades have their Sharpe set to NaN, since `mean/std` with 1-3 trades produces statistically meaningless ratios.

## Validation Framework

### Why Not Cointegration?

The original validation used Engle-Granger cointegration as a primary criterion. This produces a 0% pass rate on transient correlations by design: cointegration tests require a long-run equilibrium relationship, but transient correlations form and dissolve on shorter time frames. Full-period stationarity tests fail because the relationship isn't present for the entire sample.

Meanwhile, Hurst exponent passes 96% of pairs and half-life passes 81% — the spreads *do* mean-revert short-term. This mismatch between cointegration failure and mean-reversion success motivated the switch to a scored framework.

### 5-Test Scored Framework

Each test scores 0 or 1; total score ranges 0-5. Implemented in `screener.analysis.validate_pair_relationship()`.

| Test | Threshold | Rationale |
|------|-----------|-----------|
| **ADF on spread** | p < 0.10 | Tests spread stationarity directly. Uses 0.10 rather than 0.05 because with ~136 calendar days ADF has low power; 0.10 captures genuinely borderline pairs (e.g., DAL-UAL p=0.052, AAL-UAL p=0.058) |
| **Half-life** | 5-60 days | Speed of mean reversion. Too fast (<5d) is noise, too slow (>60d) won't produce enough OOS trades |
| **Hurst exponent** | H < 0.5 | Variance ratio method. H < 0.5 indicates mean-reverting behavior |
| **Variance ratio** | Reject RW at 10% | Lo-MacKinlay test. More powerful than ADF for detecting short-run mean-reversion in noisy data. Tests whether VR(2) deviates significantly from 1.0 |
| **Rolling correlation** | Stability > 0.5 | Pearson correlation of daily **returns** (not price levels) on 4 sub-windows. Using returns avoids spurious correlation from co-trending prices. High stability means the return co-movement persists across sub-periods |

**Classification:**
- **Strong (4-5/5):** High-confidence pair — multiple independent tests confirm mean-reversion
- **Moderate (3/5):** Likely valid relationship — most evidence supports trading
- **Weak (2/5):** Marginal evidence — some mean-reversion but inconsistent
- **Fail (<2/5):** Insufficient evidence

**Backward compatibility:** `passed` = score >= 4, `near_miss` = score == 3, preserving the interface for existing code.

### Transient Validation
- `signals.transient.validate_transient_event()`: 2h execution lag + 20h calibration + 40h exploitation
- Criteria: correlation > 0.70, spread CV < 0.03, half-life < 8h, hedge drift < 0.20, must generate signals

### Walk-Forward Backtest
- `trading.trading.walk_forward_backtest()`: 5-split rolling validation
- Reports average Sharpe, standard deviation across splits, total trades

### Multi-Algorithm Consensus
- KMeans and DBSCAN run alongside OPTICS on the same combined universe
- Consensus pairs: those appearing in the top-N of all three algorithms
- Feature shuffle permutation test validates that clustering structure is non-random

## Enhanced Backtesting

The enhanced backtest module (`screener.enhanced_backtest`) addresses the baseline strategy's limitations: static z=2.0 entry with 20-day lookback produces 1-4 OOS trades across ~68 OOS days, yielding no valid Sharpe ratios.

### Adaptive Z-Score Optimization

`optimize_zscore_params()` performs grid search on calibration data only (no look-ahead bias):

- **Entry z:** [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
- **Exit z:** [0.0, 0.25, 0.5, 0.75]
- **Lookback:** [10, 15, 20]

Scoring: maximize calibration Sharpe, requiring >= 3 calibration trades. Returns `None` if no profitable parameter combo found, preventing overfitting to noise.

### Kalman Hedge Ratio

Uses `compute_hedge_ratio(method='kalman')` from `validation/pair_validation.py` to estimate the hedge ratio on calibration data only. The Kalman filter exponentially weights recent observations, giving a better terminal beta estimate than OLS for non-stationary relationships.

**Important:** The Kalman beta is estimated on calibration data and then applied as a **fixed** hedge ratio to OOS data. An earlier version used an adaptive Kalman spread (beta updated at each OOS step), but this creates artificial mean-reversion — the filter's prediction error has near-zero mean and no autocorrelation by construction, making z-score trading trivially profitable but not representative of real alpha. The fixed-beta approach avoids this pitfall while still benefiting from Kalman's superior beta estimation.

### Transaction Costs

Default 10bps (0.001) round-trip cost per trade, deducted from each completed trade's P&L. This is conservative for liquid large-cap equities but accounts for slippage.

### Three-Strategy Comparison

For each pair, the module produces side-by-side results:

1. **Baseline:** Static z=2.0, OLS hedge, 20-day lookback, no transaction costs
2. **Enhanced:** Optimized z-score params from calibration, OLS hedge, with 10bps costs
3. **Kalman:** Optimized z-score params, Kalman-filtered spread, with 10bps costs

This enables direct measurement of the value added by parameter optimization and adaptive hedging, net of realistic costs.

## Cross-Sector Analysis

The key research output is the comparison of intra-sector vs cross-sector pairs:
- **Pair type summary**: pass rates, average Sharpe, profitability fraction for each type
- **Sector pair breakdown**: which sector combinations produce the most tradeable pairs
- **Baseline comparison**: results vs semiconductor baseline (54% profitable, top Sharpe 3.55)

## Notebook Pipeline

1. `01-screen-universe.ipynb` — Screen all sectors, build combined universe, cache
2. `02-clustering.ipynb` — Run OPTICS on the full combined universe, tag pairs by sector
3. `03-signals-validation.ipynb` — Transient + stable validation on discovered pairs
4. `04-algorithm-comparison.ipynb` — Multi-algorithm consensus + permutation test
5. `05-cross-sector-comparison.ipynb` — 5-test validation, enhanced backtests, baseline comparison

## Cached Artifacts (in `screener/data/combined/`)

- `tickers.pkl`: Combined screened ticker list
- `prices.pkl`: Fetched price DataFrame
- `sector_map.pkl`: Ticker → sector mapping
- `ts_df.pkl`: Computed features (MultiIndex: Datetime × Ticker)
- `cluster_history.pkl`: OPTICS clustering results
- `pair_co_cluster_freq.pkl`: Co-clustering frequency dict
- `df_formations.pkl`: Formation events
- `df_durations.pkl`: Formation durations
- `pair_registry.pkl`: Filtered pair registry with sector tags
- `analysis_results.pkl`: Full analysis results with 5-test scores and Sharpe guard
- `enhanced_results.pkl`: Enhanced backtest results (baseline/enhanced/Kalman comparison)
- `walk_forward_results.pkl`: Walk-forward validation on top pairs
- `consensus_pairs.pkl`: Multi-algorithm consensus pairs
- `permutation_results.pkl`: Permutation test results

## Reproducibility

1. Run `01-screen-universe.ipynb` to rebuild the combined universe (screens may return different results over time)
2. Run `02-clustering.ipynb` to cluster the full cross-sector universe
3. Run `03-signals-validation.ipynb` for transient + stable validation
4. Run `04-algorithm-comparison.ipynb` for multi-algorithm consensus
5. Run `05-cross-sector-comparison.ipynb` for final analysis and comparison
