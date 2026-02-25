# Project Guide: Clustering-Driven Transient Correlation Detection

This document is the deep-dive companion to `README.md`. It explains what this project does, how every piece fits together, what we learned, and all results from both phases.

---

## 1. What This Project Does

This project asks two questions:

1. **Can unsupervised clustering detect real short-term relationships between assets?**
2. **If some of those relationships turn out to be persistent, can we trade them?**

Traditional pairs trading starts with cointegration: find two stocks with a permanent equilibrium relationship and trade the mean-reverting spread. This project starts from the opposite direction. We compute real-time features (volatility, beta, RSI, momentum, regime shifts) for each asset at every timestamp, then cluster them. When two assets land in the same cluster, it means their feature profiles look similar right now — a **transient correlation** has formed.

Most of these relationships are short-lived. Two stocks might cluster together for a few hours and then diverge. But the interesting finding is that some pairs cluster together *repeatedly*. They keep showing up in the same cluster across many timestamps. These persistently co-clustering pairs are the ones we investigate as potential trading candidates.

**The two-part logic:**

- **Clustering validates itself as a discovery method.** We use backward-looking transient validation to prove the clusters are detecting real statistical relationships (not noise). This is a method validation step, not a trading strategy.
- **Persistent pairs get traded.** Pairs that co-cluster consistently enough are tested with standard statistical tests on their full price history and backtested with proper calibration/out-of-sample splits. This is the actual trading pipeline.

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
│   ├── transient.py                Transient event validation (method validation)
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

Phase 1 (`research/`) is notebook-centric: the analysis lives directly in the notebooks, appropriate for exploratory proof-of-concept work on 40 tickers.

Phase 2 (`screener/`) factors all logic into Python modules (`analysis.py`, `enhanced_backtest.py`, `screening.py`, `universe.py`) that notebooks import. This was necessary because:

- **142 tickers across 5 sectors** meant the analysis code needed to be reusable across sectors
- **Testability**: functions in `.py` files can be unit tested; notebook cells cannot
- **Clean notebooks**: the Phase 2 notebooks read as high-level orchestration, not implementation detail
- **Iteration speed**: changing a threshold in `analysis.py` propagates to all notebooks without copy-pasting

---

## 3. How It Works (Pipeline)

There are two distinct paths through the code. Understanding which path does what is critical.

### Path A: Method Validation (Do the clusters mean anything?)

```
Raw Prices → Feature Engineering → Clustering → Formation Events
    │
    ▼
Transient Validation (signals/transient.py)
    For each formation event (backward-looking):
    - Already know the cluster existed (this is NOT real-time)
    - Skip 2 obs (execution lag)
    - Use next 20 obs to estimate hedge ratio (calibration)
    - Use next 40 obs to test spread quality (exploitation)
    - Check 5 criteria: correlation, spread CV, half-life, hedge drift, has signal
    │
    ▼
Compare clustered pairs vs random pairs
    Clustered pass rate: 4.0%
    Random pass rate: 0.8%
    → 5x lift proves clusters detect real relationships
```

**Important:** This path uses future data relative to the formation event. We find a cluster in the historical data and then look at what happened *after* it formed. You couldn't do this in real-time without already knowing the cluster exists. The purpose is to validate the clustering method, not to generate trades.

### Path B: Trading (Can we profit from persistent pairs?)

```
Raw Prices → Feature Engineering → Clustering across all timestamps
    │
    ▼
Count co-cluster frequency for every pair
    Compute noise-adjusted frequency
    Filter: only pairs above threshold (15% Phase 1, 8% Phase 2)
    These are pairs that CONSISTENTLY cluster together
    │
    ▼
Statistical Validation on FULL price history
    Phase 1: Cointegration + half-life + Hurst (3-criteria, on calibration split)
    Phase 2: 5-test scored framework (ADF, half-life, Hurst, VR, rolling corr)
    Tests run on calibration portion of the FULL daily price history
    NOT tied to specific cluster formation timestamps
    │
    ▼
Backtest on out-of-sample data
    OLS or Kalman hedge ratio from calibration → fixed for OOS
    Z-score mean-reversion on the spread
    Walk-forward validation with multiple splits
    │
    ▼
Results: Sharpe ratios, P&L, win rates
```

**Important:** This path treats clustering purely as a screening tool. It says: "which pairs should I even bother looking at?" The answer is pairs that co-cluster frequently. Then it forgets about clustering entirely and runs standard pairs trading analysis on the full price history.

### What connects the two paths

Path A proves the clustering method finds real relationships. Path B exploits a side effect: some pairs cluster together so often that they may have a persistent (not just transient) relationship worth trading. The clustering discovers candidates; standard statistical tests and backtesting determine if they're actually tradeable.

### What's NOT tested

Pairs that co-cluster only a few times (below the frequency threshold) are never validated or traded. We have no data on whether those brief relationships are individually tradeable. The transient validation (Path A) only runs on events from pairs already in the registry (above the frequency threshold). The random baseline comparison (Path A) tests random non-clustered pairs, not "infrequently clustered" pairs specifically.

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

**Naive frequency:** 15 / 100 = **15%** (misleading — most of the denominator is noise)
**Noise-adjusted frequency:** 15 / 30 = **50%** (accurate)

The noise-adjusted metric answers: "When both tickers are actual cluster members, how often are they in the *same* cluster?"

Formula from `trading/trading.py:compute_noise_adjusted_frequency()`:
```
valid_ts = (timestamps where A is non-noise) ∩ (timestamps where B is non-noise)
noise_adj_freq = co_cluster_count / len(valid_ts)
```

### Why Cointegration Fails Here (0% Pass Rate)

Cointegration requires I(1) variables with an I(0) linear combination — a permanent equilibrium. Transient correlations form and dissolve over short timeframes by design. Over the full sample period, the spread between two transiently correlated assets looks like a random walk (I(1)), so cointegration tests correctly reject it.

In Phase 1, 0 out of 6 stable pair candidates passed the cointegration test. This is expected, not a failure. It's why the 5-test framework was developed for Phase 2 — it tests short-term mean-reversion properties (stationarity, half-life, Hurst, variance ratio) that transient pairs *can* exhibit, even without long-run cointegration.

---

## 5. Validation Methodology

### Phase 1: Transient Event Validation (Method Validation, Not Trading)

This is Path A from Section 3. Its purpose is to prove the clustering method detects real statistical relationships. It is **backward-looking**: it finds formation events in historical data, already knows the cluster existed, then checks if the spread was well-behaved during that window. You could not run this in real-time without the future data that confirms the cluster formed.

When a formation event is detected (two assets enter the same cluster), `signals/transient.py` applies three consecutive windows starting from the formation timestamp:

1. **Execution lag** (2 observations): Skip, simulating real-world reaction time
2. **Calibration window** (20 observations): Estimate hedge ratio via OLS
3. **Exploitation window** (40 observations): Apply the hedge ratio, compute spread metrics, generate z-score signals, simulate P&L

Five checks determine pass/fail:

| Check | Threshold | What It Measures |
|-------|-----------|------------------|
| Return correlation | > 0.70 | Are the assets still moving together? |
| Spread CV | < 0.03 | Is the spread tight relative to price? |
| Half-life | 0 < hl < 8.0 hours | Does the spread mean-revert fast enough? |
| Hedge drift | < 0.20 | Has the hedge ratio stayed stable? |
| Has signal | At least one entry | Does the z-score actually trigger a trade? |

All five must pass. This is deliberately strict: only 4.0% of 657 tested events passed, but those that did averaged 76.9% win rate and positive P&L. Random (non-clustered) pairs passed at only 0.8%, giving a **5.0x lift over random**.

**What this proves:** When clustering says two assets are related, there really is a statistical relationship in the spread — 5x more often than random chance. The clusters are finding something real.

**What this doesn't prove:** That you could trade these individual events profitably in real-time. The validation has look-ahead bias (you already know the cluster formed before you start testing the spread).

### Phase 2: 5-Test Scored Framework (Trading Validation)

This is Path B from Section 3. It runs on pairs that passed the frequency threshold — pairs that consistently co-cluster. The tests run on the **full daily price history** (calibration split), not on specific cluster formation windows.

`screener/analysis.py` computes a 0-5 score:

| Test | Question It Answers | Threshold | Pass Rate |
|------|-------------------|-----------|-----------|
| ADF (stationarity) | Is the spread stationary? | p < 0.10 | 38.3% |
| Half-life | Does mean-reversion happen in a tradeable timeframe? | 5-60 days | 82.3% |
| Hurst exponent | Does the spread trend or mean-revert? | H < 0.5 | 90.5% |
| Variance ratio | Does the spread reject the random walk hypothesis? | VR < 1, p < 0.10 | 21.8% |
| Rolling correlation stability | Is the relationship stable across sub-windows? | stability > 0.5 | 34.7% |

Classification:
- **Score 4-5**: Strong (tradeable)
- **Score 3**: Moderate (tradeable)
- **Score 0-2**: Weak/Fail

The hardest tests to pass are variance ratio (21.8%) and rolling correlation stability (34.7%), which are the most discriminating. Half-life and Hurst pass at high rates because transient pairs are inherently short-duration and mean-reverting.

**Key distinction from Phase 1 transient validation:** These tests don't care *when* the pair clustered. They ask: "Over the entire calibration period, does this pair's spread have mean-reverting properties?" Clustering was just the mechanism that surfaced the pair as worth testing.

### Permutation Testing

Validates that observed co-clustering exceeds random chance (`validation/pair_validation.py`):

1. Sample 80 timestamps randomly
2. For each permutation: shuffle feature rows (breaking ticker-feature mapping)
3. Run the full pipeline: StandardScaler → PCA → OPTICS
4. Count permuted co-cluster occurrences
5. Z-score = (observed_count - null_mean) / null_std
6. Significant if Z > 1.96 (two-sided p < 0.05)

This answers: "Could these pairs have co-clustered this often by chance?" For significant pairs, the answer is no.

---

## 6. Trading Logic

Trading only happens on persistently co-clustering pairs. Clustering is the **discovery mechanism** — it narrows thousands of possible pair combinations down to a manageable set of candidates. Everything after that is standard pairs trading methodology.

### Step 1: Build a Pair Registry

Run clustering across all timestamps. Count how often each pair co-clusters. Compute noise-adjusted frequency. Filter to pairs above the threshold:
- Phase 1: 15% (`config.py: noise_adj_freq_threshold = 0.15`)
- Phase 2: 8% (lower because cross-sector pairs naturally co-cluster less often)

Pairs below the threshold are discarded entirely — they never get validated or backtested.

### Step 2: Validate the Spread (Full Price History)

For each registry pair, take the full daily price history and split it into calibration (67%) and out-of-sample (33%):

- **Phase 1** (`trading/trading.py`): 3-criteria classical test — cointegration (p<0.05), half-life (5-60 days), Hurst (<0.5). Must pass all 3. (This is where cointegration's 0% pass rate was discovered.)
- **Phase 2** (`screener/analysis.py`): 5-test scored framework — ADF, half-life, Hurst, variance ratio, rolling correlation. Score >= 3 to proceed.

The hedge ratio is computed on calibration data only. The spread is: `spread = price_A - beta * price_B`.

### Step 3: Backtest on Out-of-Sample Data

Only validated pairs get backtested. The hedge ratio from calibration is applied as-is to OOS data (no re-estimation).

**Z-score mean-reversion strategy** (`trading/trading.py`):

- Rolling z-score: `z = (spread - rolling_mean) / rolling_std` (lookback = 20)
- **Long entry**: z <= -2.0 (spread is cheap)
- **Short entry**: z >= 2.0 (spread is rich)
- **Exit**: |z| <= 0.5 (spread has mean-reverted)
- Sharpe only reported if >= 5 OOS trades

**Walk-forward validation** (`trading/trading.py`): 5 rolling splits through the data, each with 67/33 cal/OOS. Produces multiple independent OOS evaluations per pair.

**Enhanced backtest** (`screener/enhanced_backtest.py`) adds three improvements:
- **Z-score optimization**: Grid search over entry_z, exit_z, lookback on calibration data
- **Kalman hedge ratio**: Kalman filter run on calibration data only; terminal beta fixed for OOS. This avoids the look-ahead bias that would occur if the Kalman filter adapted continuously during OOS (creating artificial mean-reversion).
- **Transaction costs**: 10bps per trade

Three strategies compared: Baseline (static z=2.0, OLS, no costs), Enhanced (optimized z, OLS, 10bps), Kalman (Kalman terminal beta, optimized z, 10bps).

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

### Method Validation: Transient Event Results

This is the backward-looking validation that proves clustering detects real relationships (Path A). The 657 events tested are from pairs already in the registry (above the frequency threshold). The P&L and win rate numbers come from simulated trades within the exploitation window, but these are not real-time tradeable — the validation already knows the cluster existed.

- **657 events tested**, 26 passed (4.0% pass rate)
- Passed events: avg P&L $1.94, avg win rate 76.9%, avg trades 1.2
- **Random baseline: 889 events from non-clustered pairs, 0.8% pass rate**
- **Lift over random: 5.0x** (4.0% vs 0.8%)

This confirms: when clustering says two assets are related, the spread between them genuinely has better statistical properties than random pairs — 5x better.

Top pairs by pass count: MCHP-TXN (4 passes), NXPI-POWI (3), CAMT-NVMI (2), LRCX-MU (2), NVDA-TSM (2), STX-WDC (2)

### Stable Pair Cointegration Failure

6 stable candidates identified. 0 passed the 3-criteria classical test (cointegration p<0.05, half-life 5-60d, Hurst <0.5). Near-misses (2/3 criteria): QRVO-SWKS, KLAC-LRCX, ADI-NXPI, ADI-TXN. This confirmed that cointegration is the wrong framework for transient data and motivated the 5-test framework in Phase 2.

### Permutation Test

11 pairs statistically significant (Z > 1.96): QRVO-SWKS, ADI-TXN, ADI-SWKS, ADI-NXPI, QCOM-QRVO, QCOM-SWKS, MCHP-ON, CDNS-SNPS, MCHP-NXPI, STX-WDC, GFS-NXPI.

### Trading Performance (Persistent Pairs)

This is the actual trading path (Path B). Pairs that passed the frequency threshold and met at least 2/3 classical criteria were backtested on out-of-sample daily data with proper calibration/OOS splits.

14 out of 26 tradeable pairs profitable out of sample (54%). Top Sharpe: 3.55.

---

## 8. Results: Phase 2 (Cross-Sector)

142 systematically screened tickers across 5 sectors (Technology, Healthcare, Energy, Financial Services, Industrials). Screening criteria: $2B+ market cap, 5M+ daily volume, $5+ price.

### What "59% tradeable" means

3,643 pairs passed the frequency threshold (8% noise-adjusted) — meaning they co-cluster consistently enough to enter the registry. Each of those pairs then had the 5-test validation run on its **full daily price history** (calibration split). 2,148 of them scored >= 3.

So "59% tradeable" means: of the pairs that cluster together frequently enough, 59% also have statistically sound spread properties over the full history. It does NOT mean 59% of individual cluster formations are tradeable — we never tested that.

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

### Backtest Performance (Persistent Pairs Only)

All backtests run on out-of-sample data with hedge ratios estimated on calibration data only.

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

## 9. Known Gaps

### Infrequent co-clustering pairs are untested

Pairs that co-cluster a handful of times but fall below the frequency threshold are never validated or backtested. We don't know if those brief relationship episodes are individually tradeable. The transient validation (Path A) only tested events from registry pairs. The random baseline tested truly random (non-clustered) pairs, but "infrequently clustered" pairs are a different category that sits in between.

To fill this gap: run the 3-window transient validation on formation events from ALL pairs regardless of frequency, then compare pass rates across frequency buckets. This would reveal whether frequency is a necessary quality filter or just a conservative cutoff.

### Phase 2 skipped transient validation

Phase 2 went straight from clustering to the full-history 5-test framework. It never ran the backward-looking transient validation that Phase 1 did. So we don't have the "5x lift over random" confirmation for cross-sector data — only for semiconductors.

---

## 10. Future Directions

### Trading Individual Formation Events in Real-Time

The infrastructure exists in `signals/transient.py: generate_transient_signals()`. It calibrates on trailing 20 observations, scores the most recent 10, and returns a z-score signal. The challenge is removing the look-ahead bias: in real-time, you'd need to detect a formation as it happens (not after the fact) and immediately enter the calibration/exploitation pipeline.

This would capture opportunities in pairs that co-cluster infrequently — they wouldn't need to pass a frequency threshold because each event would be traded independently.

### Other Extensions

- **Duration prediction**: Train a model on historical formation durations to filter entries (only trade predicted long-duration formations)
- **Frequency-bucketed validation**: Test whether infrequent co-clustering pairs have tradeable events, or if frequency truly correlates with signal quality
- **Regime-based thresholds**: Adapt z-score entry/exit based on market volatility regime
- **Portfolio-level risk**: Aggregate exposure across active pairs, dynamic position sizing
- **Higher-order correlations**: Extend from pairs to triplets or full cluster-level relative value
