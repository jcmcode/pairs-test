# Next Steps: Research Roadmap

This document captures the full context of where this project stands and what to do next. It's written to be self-contained -- you should be able to pick this up cold and know exactly what's been done, what the open questions are, and what to build.

---

## Where We Are Now

### What Exists

```
validation/pair_validation.py    - Statistical testing library (complete, recently bug-fixed)
research/optics-clustering.ipynb - Full OPTICS pipeline: data fetch -> features -> clustering -> events
research/optics-signals.ipynb    - Pair validation, baseline comparison, duration model, signal functions
research/data/*.pkl              - 9 pickled artifacts from the clustering notebook
trading/trading.py               - Empty placeholder
```

### What We Know

1. **OPTICS clustering works for transient detection.** It reliably identifies real subsector relationships (equipment, RF, analog, EDA) validated by permutation test (11 pairs with Z > 1.96) and random baseline comparison.

2. **All relationships are transient.** Max co-clustering frequency is 37.2% (AMAT-LRCX). Median episode duration is 10 hours. No pair clusters together more than ~1/3 of the time.

3. **Prediction is weak.** Duration prediction model AUC ~0.65-0.75. Price-derived features don't predict when relationships form or how long they last.

4. **No formal cointegration found.** Zero pairs pass all three stable criteria (cointegration p<0.05, half-life 5-60d, Hurst <0.5) on 11 months of hourly data.

5. **But the top pairs keep reforming.** OOS correlation of co-clustering frequency is r=0.723. The same pairs keep finding each other, which suggests durable structural connections beneath the transient surface behavior.

### Recent Bug Fixes (Already Implemented)

- `bounce_rate()`: Now uses full spread for z-score computation (was only evaluating ~1 event)
- `hedge_ratio_drift()`: Added denominator floor of 0.1 to prevent inflated drift on small betas
- Transient validation loop: Now filters to OOS-only formation events
- Calibration window: Increased from 10h to 20h for more stable hedge ratio estimates

### Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Tickers | 40 semiconductors |
| Period | 2025-03-11 to 2026-02-09 (~11 months hourly) |
| Valid clustering timestamps | 1,368 of 1,583 (86.4%) |
| Unique co-clustering pairs | 775 |
| Permutation-significant pairs (Z>1.96) | 11 |
| Formation events | 2,069 total, 1,665 actionable (>=5h) |
| Median episode duration | 10 hours |
| OOS co-clustering correlation | r = 0.723 |
| Avg noise rate | 58.2% |

### The 11 Statistically Significant Pairs

These are the pairs where co-clustering significantly exceeds what random feature assignment would produce:

| Pair | Z-Score | Frequency | Subsector |
|------|---------|-----------|-----------|
| AMAT-LRCX | 3.49 | 37.2% | Equipment |
| QRVO-SWKS | 3.45 | 37.1% | RF |
| KLAC-LRCX | 3.42 | 37.1% | Equipment |
| AMAT-KLAC | 3.15 | 33.8% | Equipment |
| ADI-NXPI | 2.87 | 28.6% | Analog |
| ADI-TXN | 2.73 | 27.2% | Analog |
| MCHP-NXPI | 2.38 | 23.6% | Analog |
| QCOM-QRVO | 2.27 | 23.0% | RF |
| NXPI-TXN | 2.14 | 22.4% | Analog |
| ADI-SWKS | 2.12 | 22.3% | Analog/RF |
| QCOM-SWKS | 2.09 | 22.0% | RF |

These are the primary candidates for long-term relationship investigation.

---

## Phase 1: Long-Term Relationship Investigation

**Goal**: Determine if the top co-clustering pairs have tradeable long-term properties that the hourly analysis couldn't capture.

**Why this matters**: The hourly OPTICS clustering found that these pairs repeatedly enter the same behavioral regime. That's a discovery signal. But cointegration, half-life, and Hurst are statistical properties that need longer observation windows and lower-frequency data to manifest. 11 months of hourly data may simply be the wrong lens for detecting a relationship that mean-reverts over weeks, not hours.

### Step 1.1: Extended Daily Data Collection

**What to do**: Fetch 5 years of daily OHLCV data for the 40 semiconductor tickers + S&P 500 using `yfinance`. Use daily close prices (not hourly).

**Why 5 years**: Cointegration tests need sufficient data to have statistical power. With daily data, 5 years gives ~1,260 trading days. Engle-Granger and Johansen tests need at minimum 100-200 observations to be reliable; 1,260 is comfortable.

**Implementation notes**:
- Use `yf.download(tickers, period="5y", interval="1d")`
- Some tickers may not have 5 years of history (ARM IPO'd Sep 2023, ALAB listed Nov 2024, CRDO listed Jan 2024). Handle missing data gracefully -- either use available history or exclude from long-term analysis.
- Align all tickers to common trading dates (inner join on index)
- Store as a new artifact (`daily_prices_5y.pkl`) separate from the hourly `ts_df`

### Step 1.2: Cointegration Testing on Daily Data

**What to do**: For each of the 11 significant pairs (and optionally all 775), run:

1. **Engle-Granger cointegration test** (`statsmodels.tsa.stattools.coint`)
   - Already implemented in the stable pair strategy cell of optics-signals.ipynb
   - Test at p < 0.05 significance level
   - Record the test statistic, p-value, and critical values

2. **Johansen cointegration test** (`statsmodels.tsa.vector_ar.vecm.coint_johansen`)
   - More robust than Engle-Granger for detecting cointegrating relationships
   - Tests multiple cointegrating vectors simultaneously
   - Use `det_order=0` (constant in cointegrating relation) or `det_order=1` (constant + trend)
   - Record trace statistic, max eigenvalue statistic, and critical values

3. **Rolling cointegration** (important for transient context)
   - Run Engle-Granger on rolling 6-month windows (126 trading days) with 1-month step
   - This captures whether cointegration is persistent or regime-dependent
   - If a pair is cointegrated in 60-80% of windows, it's a strong long-term candidate
   - If cointegrated in 20-40% of windows, it may only cointegrate during specific market conditions (which aligns with the transient clustering findings)

**Key question this answers**: Do the pairs that cluster transiently on hourly data actually have a formal cointegrating relationship on daily data? If yes, the OPTICS clustering is acting as an efficient pair discovery mechanism.

### Step 1.3: Half-Life and Hurst on Daily Data

**What to do**: For each of the 11 pairs:

1. **Half-life of mean reversion**
   - Already implemented in `validation/pair_validation.py` as `half_life()`
   - Compute on the daily spread (using OLS hedge ratio from daily data)
   - Target range: 5-60 trading days (1-12 weeks)
   - Half-lives < 5 days are too fast to trade at daily frequency
   - Half-lives > 60 days are too slow for practical mean-reversion strategies

2. **Hurst exponent** (variance ratio method)
   - Already implemented in optics-signals.ipynb stable pair cell
   - H < 0.5 = mean-reverting (what we want)
   - H = 0.5 = random walk
   - H > 0.5 = trending
   - Compute on daily spread with max_lag=20

3. **Rolling half-life**
   - Same rolling window approach as rolling cointegration (126-day windows)
   - Check if half-life is stable over time or varies wildly
   - Unstable half-life = harder to calibrate trading parameters

**Key question this answers**: Even if cointegrated, is the mean-reversion speed practical for trading?

### Step 1.4: Hedge Ratio Methods Comparison

**What to do**: For each pair, estimate the hedge ratio using all three methods already in `compute_hedge_ratio()`:

1. **OLS** -- simple, but biased when both series have measurement error
2. **TLS (Total Least Squares)** -- accounts for noise in both series, often more stable
3. **Kalman Filter** -- adaptive, tracks time-varying hedge ratios

**Comparison metrics**:
- Compute spread for each method
- Run cointegration test on each spread
- Compare half-lives
- Compare hedge ratio stability over time (rolling window)
- The Kalman filter is particularly interesting because it naturally handles the regime-switching behavior we see in the hourly clustering

**Key question this answers**: Does a better hedge ratio estimation method make the long-term relationships tradeable even if OLS doesn't find cointegration?

### Step 1.5: Cross-Frequency Validation

**What to do**: For the pairs that show promise on daily data, check if the hourly transient clustering aligns with daily cointegration regimes.

1. Identify daily date ranges where a pair is cointegrated (from rolling cointegration in Step 1.2)
2. Check hourly co-clustering frequency during those same date ranges
3. Compare to co-clustering frequency during non-cointegrated periods

**Hypothesis**: If hourly co-clustering frequency is higher during cointegrated daily regimes, the transient clustering is detecting the same underlying relationship at a different frequency. This would validate using OPTICS as a real-time entry signal for a longer-term pairs trade.

**Key question this answers**: Is hourly co-clustering an early indicator of (or concurrent with) daily cointegration regimes?

### Step 1.6: Output

Create a new notebook `research/long-term-investigation.ipynb` with:
- Summary table: pair, Engle-Granger p-value, Johansen result, half-life, Hurst, hedge method comparison
- Rolling cointegration heatmap (pair x time window)
- Cross-frequency validation results
- Final classification: which pairs (if any) are long-term tradeable

Save new artifacts:
- `daily_prices_5y.pkl`
- `long_term_results.pkl` (cointegration + half-life + Hurst for all tested pairs)

---

## Phase 2: Algorithm Comparison (DBSCAN + KMeans)

**Goal**: Run the same pipeline with DBSCAN and KMeans to compare which algorithm best captures the relationships we care about.

**See `dbscan+kmeans.md` for the full implementation plan.**

The key comparison metrics will be:
- Do they find the same top pairs as OPTICS?
- How do noise rates compare?
- How do formation/dissolution patterns differ?
- Do any algorithms find pairs that pass long-term cointegration that OPTICS missed?

---

## Phase 3: External Data Integration (Duration Prediction)

**Goal**: Improve the duration prediction model (currently AUC ~0.65-0.75) by adding non-price features.

**Why this matters**: If we can predict which formation events will last >= 10-20 hours, we can be much more selective about which trades to enter. The current model using only price-derived features is barely above random.

### Step 3.1: Earnings Calendar Features

**What to do**: For each formation event, check whether either ticker in the pair has an earnings announcement within the next 1-5 trading days.

**Data source**: `yfinance` provides earnings dates via `Ticker.calendar` or `Ticker.earnings_dates`. Alternatively, scrape from a free calendar API.

**Features to add**:
- `days_to_next_earnings_a`: Days until ticker A reports
- `days_to_next_earnings_b`: Days until ticker B reports
- `either_reporting_within_5d`: Binary flag
- `both_reporting_within_5d`: Binary flag (same-day/week earnings for sector pairs)

**Hypothesis**: Formation events near earnings may dissolve faster (event-driven spike then reversion to independent behavior) or last longer (earnings confirm shared narrative).

### Step 3.2: Implied Volatility / Options Data

**What to do**: Capture the options-implied volatility for each ticker at formation time.

**Data source**: `yfinance` provides some options data. For higher quality, consider CBOE or a paid provider. At minimum, get the 30-day ATM implied vol.

**Features to add**:
- `iv_a`, `iv_b`: Implied vol for each ticker
- `iv_ratio`: Ratio of the two (divergent IV may predict relationship breakdown)
- `iv_vs_realized`: IV / realized vol (high ratio = market expects a move)

**Hypothesis**: High implied volatility at formation time may predict shorter episodes (the market is pricing in an event that will break the correlation).

### Step 3.3: Macro Event Calendar

**What to do**: Flag formation events that occur near FOMC meetings, CPI releases, jobs reports, or other macro events.

**Data source**: FRED calendar, or hardcode the known dates for the observation period.

**Features to add**:
- `hours_to_next_fomc`: Hours until next Fed decision
- `hours_since_last_macro`: Hours since last major release
- `macro_event_today`: Binary flag

**Hypothesis**: Macro events may cause temporary sector-wide correlation (everything moves together) that dissolves once the event passes. These would be short-duration episodes.

### Step 3.4: Retrain Duration Model

**What to do**: Add the new features to the existing logistic regression pipeline in optics-signals.ipynb and retrain.

- Keep the same TimeSeriesSplit 5-fold CV
- Compare AUC/precision/recall before and after external features
- Test feature importance to identify which external features matter most
- Also test Random Forest and XGBoost as alternative models (logistic regression may underfit the interaction between technical and external features)

**Target**: AUC > 0.80 would make the model useful for trade filtering.

---

## Phase 4: Cluster Transition Analysis

**Goal**: Instead of just detecting when pairs form/dissolve, analyze what happens when individual tickers switch between clusters.

**Why this matters**: A ticker moving from the "analog" cluster to the "RF" cluster might signal a regime change that's tradeable. Currently we only track pair-level events; ticker-level transitions are a richer signal.

### Step 4.1: Ticker Transition Matrix

**What to do**: For each ticker, track its cluster assignment over time and build a transition matrix.

- At each timestamp, record which cluster each ticker belongs to (or -1 for noise)
- Compute transition probabilities: P(cluster_j at t+1 | cluster_i at t)
- Identify tickers with high transition rates (regime-switchers) vs low (stable members)

**Output**: Per-ticker transition matrices and a "stability score" (fraction of consecutive timestamps in the same cluster).

### Step 4.2: Transition-Based Signals

**What to do**: When a ticker transitions into a new cluster, it becomes a candidate pair with every existing member of that cluster.

- Track which transitions lead to formation events that pass validation
- Compare: are transitions more predictive of good trades than just observing co-clustering?
- A ticker joining an established cluster (where other members have been there for several hours) may be a stronger signal than two tickers simultaneously entering a new cluster

### Step 4.3: Pre-Formation Feature Analysis

**What to do**: For formation events that pass validation, look at what the features looked like 5-10 hours before formation.

- Extract feature vectors for both tickers at t-5, t-4, ..., t-1 before formation at time t
- Compare to feature vectors at non-formation timestamps
- Use this to build a "formation likelihood" model -- not predicting duration, but predicting whether a formation will happen at all

**This is the prediction angle the project originally aimed for.** The current work proved that price-derived features at formation time don't predict duration. But they might predict formation itself -- the question is whether the feature trajectory leading up to formation is distinctive.

---

## Phase 5: Universe Expansion

**Goal**: Test the methodology on other sectors to see if the findings generalize.

### Candidate Sectors

| Sector | Why It Might Work | Tickers to Consider |
|--------|-------------------|-------------------|
| **Biotech** | Subsector clusters (oncology, immunology, gene therapy), event-driven (FDA approvals, trial data) | XBI holdings, top 30-40 by market cap |
| **Energy** | Clear subsectors (E&P, midstream, services, refining), macro-driven (oil price, OPEC) | XLE holdings |
| **REITs** | Property type clusters (office, residential, industrial, data center), interest rate sensitive | VNQ holdings |
| **Regional Banks** | Geographic/size clusters, rate-sensitive, M&A activity | KRE holdings |

### Implementation

For each new sector:
1. Select 30-50 tickers with sufficient liquidity
2. Fetch hourly data (same period as semiconductors if possible)
3. Engineer same 13 features (Vol, Beta_SPX, Beta_Sector, RSI, Momentum, Regime Shift)
4. Run the full OPTICS pipeline
5. Compare: co-clustering frequency distribution, noise rate, episode durations, permutation test results
6. Run long-term investigation (Phase 1) on the top pairs

**Key comparison**: Do other sectors produce more stable pairs (higher co-clustering frequency, longer episodes)? If REITs or energy show 50%+ co-clustering with cointegration, those may be better sectors for classical pairs trading while semiconductors are better for transient strategies.

---

## Phase 6: Real-Time Implementation

**Goal**: Build a live scanner that runs the pipeline on new data as it arrives.

### Step 6.1: Data Pipeline

- Fetch latest hourly candle for all 40 tickers
- Append to rolling feature DataFrame (maintain last 150 hours for medium-term window calculations)
- Compute all 13 features for the new timestamp
- Run StandardScaler -> PCA -> OPTICS on the new snapshot

### Step 6.2: Event Detection

- Compare new clustering to previous timestamp using `detect_new_formations()` (already implemented in optics-signals.ipynb)
- For each new formation, run `validate_transient_event()` using the most recent calibration window
- If the duration prediction model (Phase 3) has been improved, apply it as a filter

### Step 6.3: Signal Generation

- For active pairs (currently co-clustering and validated), run `generate_transient_signals()` (already implemented)
- Output: ticker_a, ticker_b, z_score, signal (long/short/exit/wait), beta, spread_cv, half_life

### Step 6.4: Execution Interface

**What to build in `trading/trading.py`**:
- Position tracker: current positions, entry prices, entry timestamps
- Risk limits: max position size, max concurrent pairs, max sector exposure
- Order generation: translate signals into buy/sell orders (paper trading first)
- P&L tracking: real-time cumulative P&L per pair and aggregate

### Step 6.5: Monitoring

- Dashboard showing: active clusters, active pair positions, z-scores, P&L
- Alerts for: new formation events, validation passes, z-score threshold crossings, dissolution events
- Historical log: all events, signals, and trades for post-hoc analysis

---

## Priority Order

| Priority | Phase | Effort | Expected Value |
|----------|-------|--------|---------------|
| **1** | Phase 1: Long-term investigation | Medium | High -- determines if there's a tradeable long-term strategy |
| **2** | Phase 2: DBSCAN + KMeans comparison | Medium | Medium -- validates algorithm choice, may find pairs OPTICS missed |
| **3** | Phase 3: External data for duration prediction | High | Medium-High -- makes transient strategy more selective |
| **4** | Phase 4: Cluster transition analysis | Medium | Medium -- novel signal, may unlock the prediction angle |
| **5** | Phase 5: Universe expansion | Low-Medium per sector | Medium -- tests generalizability |
| **6** | Phase 6: Real-time implementation | High | High (but only after strategy is validated) |

**Recommended starting point**: Phase 1 (long-term investigation) because it directly answers the most important open question -- whether the top clustering pairs have tradeable long-term properties. Phase 2 (algorithm comparison) can run in parallel since it uses the same data and features.

---

## Files to Create

| File | Purpose |
|------|---------|
| `research/long-term-investigation.ipynb` | Phase 1: daily cointegration, half-life, Hurst, rolling analysis |
| `research/dbscan-clustering.ipynb` | Phase 2: DBSCAN version of optics-clustering.ipynb |
| `research/kmeans-clustering.ipynb` | Phase 2: KMeans version of optics-clustering.ipynb |
| `research/algorithm-comparison.ipynb` | Phase 2: side-by-side comparison of all three algorithms |
| `research/duration-prediction-v2.ipynb` | Phase 3: retrained model with external features |
| `research/cluster-transitions.ipynb` | Phase 4: ticker-level transition analysis |
| `trading/trading.py` | Phase 6: execution logic |

---

## Context for Future Sessions

### Artifacts Available in `research/data/`

All pickle files are produced by `optics-clustering.ipynb` and consumed by `optics-signals.ipynb`:

| Artifact | Type | Key Fields |
|----------|------|------------|
| `ts_df.pkl` | DataFrame (MultiIndex: Datetime, Ticker) | Price, Returns, 13 feature columns |
| `cluster_history.pkl` | DataFrame | Datetime, Ticker, Cluster_ID, Num_Clusters, Noise_Pct |
| `df_formations.pkl` | DataFrame | Ticker_1, Ticker_2, Pair, Formation_Time, Timestamp_Index |
| `df_formations_actionable.pkl` | DataFrame | Same + Duration_Hours (filtered >= 5h) |
| `df_durations.pkl` | DataFrame | Pair, Formation_Time, Dissolution_Time, Duration_Hours |
| `df_pair_stability.pkl` | DataFrame | Ticker_1, Ticker_2, Pair, Co_Cluster_Count, Co_Cluster_Frequency, Is_Stable |
| `pair_classification.pkl` | DataFrame | Pair, Formation_Count, Avg_Duration, Co_Cluster_Frequency, Category |
| `pair_co_cluster_freq.pkl` | dict | {(ticker_a, ticker_b): count} |
| `oos_split_timestamp.pkl` | Timestamp | 2025-11-18 11:30:00+00:00 |

### Validation Module Functions

All in `validation/pair_validation.py`:

| Function | Purpose | Used By |
|----------|---------|---------|
| `compute_hedge_ratio(a, b, method)` | OLS/TLS/Kalman hedge ratio | Both notebooks |
| `hedge_ratio_drift(beta_cal, beta_exploit)` | Drift measurement (floor=0.1) | optics-signals |
| `spread_cv_normalized(spread, a, b)` | Spread CV normalized by price | optics-signals |
| `half_life(spread)` | AR(1) mean-reversion half-life | Both notebooks |
| `zscore_signals(spread, lookback, entry_z, exit_z)` | Z-score trading signals | optics-signals |
| `simulate_spread_pnl(spread, signals, cost)` | P&L backtest | optics-signals |
| `bounce_rate(spread, W, z_entry, horizon, improve)` | Mean-reversion bounce rate (fixed) | validate_pair() |
| `cluster_persistence(clusters_wide, a, b, t, H)` | Rolling co-cluster persistence | validate_pair() |
| `feature_shuffle_permutation_test(...)` | Statistical significance test | optics-clustering |

### OPTICS Configuration Used

```python
# Clustering
OPTICS(min_samples=3, metric='euclidean', xi=0.05, min_cluster_size=3)

# Features used for clustering (9 of 13)
features_to_cluster = [
    'Returns', 'Vol_Short', 'Beta_SPX_Short', 'Beta_Sector_Short',
    'RSI', 'Momentum_5H', 'Vol_Regime_Shift',
    'Beta_SPX_Regime_Shift', 'Beta_Sector_Regime_Shift'
]

# Validation config
TRANSIENT_CONFIG = {
    'CALIBRATION': 20,
    'EXPLOITATION': 40,
    'EXECUTION_LAG': 2,
    'CORR_THRESHOLD': 0.70,
    'CV_THRESHOLD': 0.03,
    'HL_MAX': 8,
    'HEDGE_DRIFT_MAX': 0.20,
    'ENTRY_Z': 2.0,
    'EXIT_Z': 0.5,
}

# Formation detection
MIN_GAP_HOURS = 5
MIN_EPISODE_HOURS = 5
```

### Tickers in Universe

```python
stocks = [
    "^GSPC",  # S&P 500 (for beta)
    "NVDA", "TSM", "AVGO", "AMD", "INTC", "MU", "TXN", "QCOM", "ADI", "MCHP",
    "ASML", "AMAT", "LRCX", "KLAC", "TER", "ENTG", "NVMI", "TOELY",
    "ON", "NXPI", "STM", "LSCC", "MPWR", "QRVO", "SWKS", "ALAB", "CRDO",
    "ARM", "SNPS", "CDNS", "CEVA",
    "WDC", "STX",
    "GFS", "MRVL", "MTSI", "POWI", "SMTC", "VICR", "CAMT"
]
```
