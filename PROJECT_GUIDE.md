# Project Guide: Clustering-Driven Transient Correlation Detection

This document is the deep-dive companion to `README.md`. It explains what this project does, how every piece fits together, all results from both phases, and why certain design decisions were made.

---

## 1. What This Project Does

Traditional pairs trading looks for **cointegrated** assets: two stocks with a long-run equilibrium relationship where the spread mean-reverts indefinitely. This project takes a fundamentally different approach.

Instead of assuming permanent relationships, we use **unsupervised clustering** on real-time features (volatility, beta, RSI, momentum, regime shifts) to discover assets that *temporarily* behave alike. When two stocks land in the same cluster, it signals a **transient correlation** has formed. We validate that relationship with statistical tests, then trade the spread while the relationship holds.

The key insight: correlations between assets form and dissolve constantly. Rather than requiring a permanent relationship (cointegration), we detect when relationships *form*, validate they're tradeable, and exploit the mean-reversion in the spread during the relationship window.

**Why cointegration fails here (0% pass rate):** Cointegration requires I(1) variables with an I(0) linear combination, i.e., a permanent equilibrium. Transient correlations are short-lived by definition. Over the full sample, the spread between two transiently correlated assets looks like a random walk. The 5-test framework we developed focuses on short-term mean-reversion properties instead.

---

## 2. Repository Organization

```
pairs-test/
│
├── validation/                     Core statistical testing library
│   └── pair_validation.py          validate_pair(), hedge ratios, permutation tests
│
├── signals/                        Signal detection and feature engineering
│   ├── features.py                 9-feature computation (vol, beta, RSI, regime shifts)
│   ├── detection.py                Clustering pipeline, formation/dissolution events
│   ├── transient.py                Transient event validation (3-window approach)
│   └── stable.py                   Stable pair tracking
│
├── trading/                        Backtesting and trading logic
│   └── trading.py                  Z-score strategy, walk-forward validation, pair analysis
│
├── screener/                       Phase 2: Cross-sector screening module
│   ├── screening.py                Ticker screening via yfscreen
│   ├── universe.py                 Universe construction (3-layer pipeline)
│   ├── analysis.py                 5-test scored validation framework
│   ├── enhanced_backtest.py        Adaptive z-score, Kalman hedge, transaction costs
│   ├── features_adapter.py         Feature computation adapter for screener
│   ├── config.py                   Screener configuration
│   └── notebooks/                  Analysis pipeline (run 01 through 05 in order)
│       ├── 01-screen-universe.ipynb
│       ├── 02-clustering.ipynb
│       ├── 03-signals-validation.ipynb
│       ├── 04-algorithm-comparison.ipynb
│       └── 05-cross-sector-comparison.ipynb
│
├── research/                       Phase 1: Semiconductor exploratory notebooks
│   ├── optics-clustering.ipynb     OPTICS pipeline (features → clustering → events)
│   ├── optics-signals.ipynb        Transient validation and signal generation
│   ├── KMeans.ipynb                KMeans pipeline
│   ├── DBScan.ipynb                DBSCAN pipeline
│   └── algorithm-comparison.ipynb  Algorithm comparison and consensus pairs
│
├── tests/                          Test suite (pytest tests/)
│   ├── test_validation.py
│   ├── test_signals.py
│   ├── test_trading.py
│   └── test_config.py
│
├── config.py                       Global configuration (all thresholds as frozen dataclasses)
├── requirements.txt                Python dependencies
├── CLAUDE.md                       AI assistant context
└── README.md                       Quick-start guide
```

### Why Phase 2 has separate .py modules

Phase 1 (`research/`) is notebook-centric: the analysis lives directly in the notebooks, which was appropriate for exploratory proof-of-concept work on 40 tickers.

Phase 2 (`screener/`) factors all logic into Python modules (`analysis.py`, `enhanced_backtest.py`, `screening.py`, `universe.py`) that notebooks import. This was necessary because:

- **142 tickers across 5 sectors** meant the analysis code needed to be reusable across sectors
- **Testability**: functions in `.py` files can be unit tested; notebook cells cannot
- **Clean notebooks**: the Phase 2 notebooks read as high-level orchestration, not implementation detail
- **Iteration speed**: changing a threshold in `analysis.py` propagates to all notebooks without copy-pasting

---

## 3. How It Works (Pipeline)

```
Raw Prices (yfinance hourly data)
    │
    ▼
Feature Engineering (signals/features.py)
    9 features per ticker per timestamp:
    returns, vol_short, vol_medium, beta_SPX_short, beta_SPX_medium,
    beta_sector_short, beta_sector_medium, RSI, momentum_5H
    + 3 regime shift indicators (vol, beta_SPX, beta_sector)
    │
    ▼
Clustering (signals/detection.py)
    StandardScaler → PCA (90% variance) → OPTICS
    Per-timestamp: each ticker gets a cluster label (-1 = noise)
    │
    ▼
Pair Discovery
    Count co-cluster frequency across all timestamps
    Compute noise-adjusted frequency (see Key Concepts)
    Filter: frequency > threshold (15% Phase 1, 8% Phase 2)
    │
    ▼
Validation
    Phase 1: 3-window transient validation (signals/transient.py)
    Phase 2: 5-test scored framework (screener/analysis.py)
    │
    ▼
Backtesting (trading/trading.py, screener/enhanced_backtest.py)
    Z-score mean-reversion on the spread
    Walk-forward validation with calibration/OOS splits
    │
    ▼
Results
    Pair rankings, Sharpe ratios, P&L, permutation significance
```

---

## 4. Key Concepts

### Transient Correlation vs Cointegration

| Property | Cointegration | Transient Correlation |
|----------|--------------|----------------------|
| Duration | Permanent (years) | Short-lived (hours to weeks) |
| Test | Engle-Granger / Johansen | Cluster membership |
| Spread behavior | Always mean-reverts | Mean-reverts during cluster overlap |
| Discovery | Statistical test on historical data | Real-time clustering on features |
| Failure mode | Relationship breaks unexpectedly | Relationship dissolves (expected) |

### Noise-Adjusted Frequency

OPTICS labels ~58% of observations as noise (cluster_id = -1). Using total timestamps as the denominator drastically underestimates how often two assets genuinely co-cluster.

**Concrete example:** Consider AAPL and NVDA over 100 timestamps.

- AAPL is non-noise at 50 timestamps
- NVDA is non-noise at 40 timestamps
- Both are non-noise at the same time at 30 timestamps
- They co-cluster at 15 of those 30 timestamps

**Naive frequency:** 15 / 100 = **15%** (misleading)
**Noise-adjusted frequency:** 15 / 30 = **50%** (accurate)

The noise-adjusted metric answers: "When both tickers are actual cluster members, how often are they in the *same* cluster?"

Formula from `signals/detection.py`:
```
valid_ts = (timestamps where A is non-noise) ∩ (timestamps where B is non-noise)
noise_adj_freq = co_cluster_count / len(valid_ts)
```

### Why Cointegration Fails Here

Cointegration tests require a long-run stable equilibrium. Transient correlations form and dissolve over short timeframes by design. Over the full sample period, the spread between two transiently correlated assets looks like a random walk (I(1)), so cointegration tests correctly reject it. In Phase 1, 0 out of 6 stable pair candidates passed the cointegration test. This is expected, not a failure, which is why the 5-test framework was developed for Phase 2.

---

## 5. Validation Methodology

### Phase 1: Transient Event Validation (3-Window Approach)

When a formation event is detected (two assets enter the same cluster), the validation pipeline from `signals/transient.py` applies three consecutive windows:

1. **Execution lag** (2 observations): Delay before entering, simulating real-world reaction time
2. **Calibration window** (20 observations): Estimate hedge ratio via OLS on this data
3. **Exploitation window** (40 observations): Apply the hedge ratio, generate z-score signals, simulate P&L

Five checks determine pass/fail:

| Check | Threshold | What It Measures |
|-------|-----------|------------------|
| Return correlation | > 0.70 | Are the assets still moving together? |
| Spread CV | < 0.03 | Is the spread tight relative to price? |
| Half-life | 0 < hl < 8.0 hours | Does the spread mean-revert fast enough? |
| Hedge drift | < 0.20 | Has the hedge ratio stayed stable? |
| Has signal | At least one entry | Does the z-score actually trigger a trade? |

All five must pass. This is deliberately strict: only 4.0% of 657 tested events passed, but those that did averaged 76.9% win rate and positive P&L. Random (non-clustered) pairs passed at only 0.8%, giving a **5.0x lift** over random.

### Phase 2: 5-Test Scored Framework

Phase 2 (`screener/analysis.py`) replaced the binary pass/fail with a 0-5 score:

| Test | Question It Answers | Threshold | Pass Rate |
|------|-------------------|-----------|-----------|
| ADF (stationarity) | Is the spread stationary? | p < 0.10 | 38.3% |
| Half-life | Does mean-reversion happen in a tradeable timeframe? | 5-60 days | 82.3% |
| Hurst exponent | Does the spread trend or mean-revert? | H < 0.5 | 90.5% |
| Variance ratio | Does the spread reject the random walk hypothesis? | VR < 1, p < 0.10 | 21.8% |
| Rolling correlation stability | Is the relationship stable within the calibration window? | stability > 0.5 | 34.7% |

Classification:
- **Score 4-5**: Strong (tradeable)
- **Score 3**: Moderate (tradeable)
- **Score 0-2**: Weak/Fail

The hardest tests to pass are variance ratio (21.8%) and rolling correlation stability (34.7%), which are the most discriminating. Half-life and Hurst pass at high rates because transient pairs are inherently short-duration and mean-reverting.

### Permutation Testing

Validates that observed co-clustering exceeds random chance (`validation/pair_validation.py`):

1. Sample 80 timestamps randomly
2. For each permutation: shuffle feature rows (breaking ticker-feature mapping)
3. Run the full pipeline: StandardScaler → PCA → OPTICS
4. Count permuted co-cluster occurrences
5. Z-score = (observed_count - null_mean) / null_std
6. Significant if Z > 1.96 (two-sided p < 0.05)

---

## 6. Trading Logic

Clustering is the **discovery mechanism**, not the trading signal. Here's how discovery connects to trading:

### Step 1: Build a Pair Registry

Run clustering across all timestamps. Count how often each pair co-clusters. Compute noise-adjusted frequency. Filter to pairs above the threshold:
- Phase 1: 15% (`config.py: noise_adj_freq_threshold = 0.15`)
- Phase 2: 8% (lower threshold because cross-sector pairs naturally co-cluster less often)

### Step 2: Validate

Apply the validation framework (Phase 1: transient 5-check; Phase 2: 5-test scored) to each pair in the registry. Only validated pairs proceed to backtesting.

### Step 3: Backtest

**Z-score mean-reversion strategy** (`trading/trading.py`):

- Compute spread: `spread = price_A - beta * price_B` (beta from OLS or Kalman)
- Rolling z-score: `z = (spread - rolling_mean) / rolling_std` (lookback = 20)
- **Long entry**: z <= -2.0 (spread is cheap)
- **Short entry**: z >= 2.0 (spread is rich)
- **Exit**: |z| <= 0.5 (spread has mean-reverted)

**Walk-forward validation**: 5 rolling splits, each with 67/33 calibration/OOS split. Sharpe only reported if >= 5 OOS trades.

**Enhanced backtest** (`screener/enhanced_backtest.py`) adds:
- **Z-score optimization**: Grid search over entry_z, exit_z, lookback on calibration data
- **Kalman hedge ratio**: Kalman filter run on calibration data only; terminal beta fixed for OOS (no adaptive updates, which would create artificial mean-reversion)
- **Transaction costs**: 10bps per trade

---

## 7. Results: Phase 1 (Semiconductors)

40 hand-picked semiconductor tickers. Three clustering algorithms compared on a common time range (2025-03-24 to 2026-02-23, hourly data).

### Algorithm Comparison

| Metric | OPTICS | KMeans | DBSCAN |
|--------|--------|--------|--------|
| Valid windows | 1,364 | 1,578 | 1,578 |
| Avg clusters/timestamp | 3.36 | 3.09 | 1.47 |
| Avg noise rate | 58.46% | 10.00% | 28.00% |
| OOS correlation (train/test freq) | 0.665 | 0.735 | 0.840 |
| Average rank | **1.50** | 2.00 | 2.50 |

OPTICS selected as primary algorithm: best at detecting variable-density clusters and transient formations. DBSCAN recommended for confirmation (highest OOS correlation).

### Formation Events

| Algorithm | Total Formations | Actionable | Unique Pairs | Mean Duration (h) | Median Duration (h) |
|-----------|-----------------|------------|-------------|-------------------|---------------------|
| OPTICS | 20,242 | 20,242 (100%) | 776 | 19.86 | 1.0 |
| KMeans | 28,136 | 18,829 (67%) | 780 | 25.39 | 9.0 |
| DBSCAN | 27,663 | 17,507 (63%) | 780 | 26.42 | 9.0 |

### 8 Consensus Pairs (Top-20 across all three algorithms)

ADI-NXPI, ADI-SWKS, ADI-TXN, AMAT-KLAC, AMAT-LRCX, KLAC-LRCX, NXPI-TXN, QRVO-SWKS

These reflect genuine semiconductor subsector structure: analog (ADI, NXPI, TXN, SWKS, QRVO), equipment (AMAT, KLAC, LRCX).

### Top 10 Pairs by Noise-Adjusted Frequency (OPTICS)

| Pair | Frequency |
|------|-----------|
| QRVO-SWKS | 0.383 |
| KLAC-LRCX | 0.364 |
| AMAT-LRCX | 0.361 |
| AMAT-KLAC | 0.326 |
| ADI-NXPI | 0.262 |
| ADI-TXN | 0.257 |
| MCHP-NXPI | 0.232 |
| QCOM-QRVO | 0.227 |
| ADI-SWKS | 0.222 |
| QCOM-SWKS | 0.214 |

### Pair Classification

| Algorithm | Transient | Stable | Sporadic | Unknown |
|-----------|-----------|--------|----------|---------|
| OPTICS | 681 (87.8%) | 6 (0.8%) | 11 (1.4%) | 78 (10.1%) |
| KMeans | 549 (70.4%) | 231 (29.6%) | - | - |
| DBSCAN | 474 (60.8%) | 306 (39.2%) | - | - |

424 pairs classified as transient across all three algorithms.

### Transient Validation Results

- **657 events tested**, 26 passed (4.0% pass rate)
- Passed events: avg P&L $1.94, avg win rate 76.9%
- **Random baseline: 889 events, 0.8% pass rate**
- **Lift over random: 5.0x** (4.0% vs 0.8%)

Top pairs by pass count: MCHP-TXN (4 passes), NXPI-POWI (3), CAMT-NVMI (2), LRCX-MU (2), NVDA-TSM (2), STX-WDC (2)

### Stable Pair Cointegration Failure

6 stable candidates identified. 0 passed the 3-criteria classical test (cointegration p<0.05, half-life 5-60d, Hurst <0.5). Near-misses (2/3 criteria): QRVO-SWKS, KLAC-LRCX, ADI-NXPI, ADI-TXN. This confirmed that cointegration is the wrong framework for transient data.

### Permutation Test

11 pairs statistically significant (Z > 1.96): QRVO-SWKS, ADI-TXN, ADI-SWKS, ADI-NXPI, QCOM-QRVO, QCOM-SWKS, MCHP-ON, CDNS-SNPS, MCHP-NXPI, STX-WDC, GFS-NXPI.

### Trading Performance

14 out of 26 tradeable pairs profitable out of sample (54%). Top Sharpe: 3.55.

---

## 8. Results: Phase 2 (Cross-Sector)

142 systematically screened tickers across 5 sectors (Technology, Healthcare, Energy, Financial Services, Industrials). Screening criteria: $2B+ market cap, 5M+ daily volume, $5+ price.

### Score Distribution

| Classification | Count | % |
|---------------|-------|---|
| Strong (score 4-5) | 605 | 16.6% |
| Moderate (score 3) | 1,543 | 42.3% |
| Weak (score 2) | 1,200 | 32.9% |
| Fail (score <2) | 295 | 8.1% |
| **Total** | **3,643** | |
| **Tradeable (score >= 3)** | **2,148** | **59.0%** |

### Individual Test Pass Rates

| Test | Pass Rate |
|------|-----------|
| ADF (stationarity) | 38.3% |
| Half-life (5-60 days) | 82.3% |
| Hurst exponent (<0.5) | 90.5% |
| Variance ratio (VR<1) | 21.8% |
| Rolling correlation stability | 34.7% |

### Intra-Sector vs Cross-Sector

| Type | Pairs | Tradeable | Strong | Avg Score | Pass Rate | Avg Sharpe | Profitable |
|------|-------|-----------|--------|-----------|-----------|------------|-----------|
| Intra-sector | 1,442 | 925 | 282 | 2.77 | 19.6% | 1.95 | 42.4% |
| Cross-sector | 2,201 | 1,223 | 323 | 2.62 | 14.7% | 1.59 | 40.8% |

Intra-sector pairs have higher scores and pass rates as expected, but cross-sector pairs still produce tradeable opportunities.

### Sector Performance

| Sector | Pairs | Passed | Pass Rate | Avg Sharpe | Best Sharpe | Profitable |
|--------|-------|--------|-----------|------------|-------------|-----------|
| Healthcare | 208 | 40 | 19.2% | 1.08 | 148.51 | 44.1% |
| Energy | 90 | 11 | 12.2% | 1.08 | 36.72 | 46.2% |
| Technology | 15 | 1 | 6.7% | 0.57 | 2.84 | 66.7% |
| Financial Svcs | 92 | 4 | 4.3% | 0.36 | 3.81 | 52.3% |
| Industrials | 88 | 4 | 4.5% | -0.40 | 31.19 | 37.1% |

### Permutation Testing

| Sector | Tested | Significant (Z>1.96) | Rate |
|--------|--------|---------------------|------|
| Technology | 1,640 | 230 | 14.0% |
| Financial Services | 105 | 32 | 30.5% |
| Industrials | 153 | 46 | 30.1% |
| Energy | 351 | 72 | 20.5% |
| Healthcare | 253 | 40 | 15.8% |
| **Total** | **2,502** | **420** | **16.8%** |

Top significant pairs by sector:
- **Technology**: ASX-TSM (z=13.1), LRCX-TSM (z=13.1), CTSH-INFY (z=12.0)
- **Energy**: CVX-XOM (z=13.0), PBR-PBR.A (z=12.6), COP-OXY (z=10.1)
- **Healthcare**: MRK-PFE (z=9.9), ABBV-PFE (z=8.8), JNJ-MRK (z=6.6)
- **Financial Services**: CRBG-IVZ (z=11.4), CRBG-V (z=8.5)
- **Industrials**: DAL-UAL (z=11.5), AAL-UAL (z=10.8), AAL-DAL (z=8.8)

### Backtest Performance

**Baseline** (OLS hedge, z=2.0, no transaction costs):
- 2,148 tradeable pairs
- 41% profitable, avg Sharpe 1.73, top Sharpe 2.56

**Enhanced** (Kalman hedge, optimized z-scores, 10bps transaction costs):
- 57% profitable (up from 41%), top Sharpe 18.47
- Kalman terminal beta approach avoids look-ahead bias from adaptive updates

**Walk-forward highlights**: RUN-VNET Sharpe 3.47, NVDA-ORCL Sharpe 2.59

### Top Pairs (by score then Sharpe, minimum 5 OOS trades)

| Pair | Type | Score | Freq | Sharpe | P&L |
|------|------|-------|------|--------|-----|
| ADT-NOW | cross | 4 | 0.099 | 2.56 | 1.10 |
| APH-TSM | intra | 4 | 0.349 | 1.93 | 44.07 |
| CRBG-NOW | cross | 4 | 0.159 | 1.64 | 4.41 |
| CRBG-WU | intra | 3 | 0.222 | 1.96 | 5.14 |

---

## 9. Future Directions

The infrastructure for the most promising extension already exists in the codebase.

### Trading Individual Formation Events in Real-Time

Currently, the pipeline trades pairs that *consistently* co-cluster above a frequency threshold. But `signals/transient.py` contains `generate_transient_signals()`, which can produce signals for *individual* formation events as they happen:

- Detect a new formation (two assets enter the same cluster)
- Calibrate on trailing 20 observations
- Generate z-score signals on the most recent 10 observations
- Trade the spread during the exploitation window

This would capture more opportunities, including pairs that co-cluster infrequently but produce strong signals when they do.

### Other Extensions

- **Duration prediction**: Train a model on historical formation durations to filter entries (only trade predicted long-duration formations)
- **Regime-based thresholds**: Adapt z-score entry/exit based on market volatility regime
- **Portfolio-level risk**: Aggregate exposure across active pairs, dynamic position sizing
- **Higher-order correlations**: Extend from pairs to triplets or full cluster-level relative value
