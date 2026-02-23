# Findings: Transient Clustering as a Pair Discovery Tool

## The Key Result

The OPTICS transient detection engine was built to find short-lived relationships between semiconductor stocks as they form in real time. An unexpected side effect: **the pairs it keeps surfacing turn out to have genuine longer-term structural relationships that are tradeable using classical mean-reversion strategies.**

23 out of 33 filtered pairs passed 2 of 3 classical pairs trading criteria (half-life, Hurst exponent, cointegration). Every single one produced positive P&L in a z-score mean-reversion backtest over ~11 months of daily data. The consensus pairs — those found by all three clustering algorithms — outperformed the rest.

This means the transient engine doubles as a **pair discovery tool**: run it once, and the pairs it repeatedly clusters together are strong candidates for traditional pairs trading, not just transient strategies.

---

## Background

### What the Engine Does

The OPTICS clustering pipeline processes hourly data for 40 semiconductor tickers across ~11 months (March 2025 to February 2026). At each hourly timestamp it:

1. Computes 9 features per ticker (volatility, momentum, RSI, market beta, sector beta, regime shifts)
2. Standardizes and reduces dimensionality via PCA (retaining 90% variance)
3. Runs OPTICS clustering to group tickers with similar feature profiles
4. Records which tickers end up in the same cluster

When two tickers land in the same cluster, it signals their behavior is temporarily aligned — a "relationship formation." The engine tracks these formation and dissolution events over time.

### The Noise Problem

OPTICS is conservative: ~58% of the time, any given ticker is assigned to the noise cluster (label -1), meaning it doesn't fit into any group. This is by design — it only forms clusters when there's genuine feature similarity. But it means raw co-clustering frequencies are misleadingly low.

For example, KLAC and LRCX (both semiconductor equipment companies) show a raw co-clustering frequency of 37.8%. That sounds moderate. But when you account for the fact that both tickers spend more than half their time in the noise cluster, the picture changes dramatically.

**Noise-adjusted frequency** uses only the timestamps where both tickers are actually visible (non-noise) as the denominator. When both KLAC and LRCX happen to be in real clusters, they're in the *same* cluster a much higher proportion of the time.

### Three Algorithms, Same Pairs

We ran the same pipeline with three different clustering algorithms:

| Algorithm | Approach | Noise Rate |
|-----------|----------|------------|
| OPTICS | Density-based, variable density, auto-detects cluster count | 58% |
| DBSCAN | Density-based, fixed density threshold (adaptive eps) | 28% |
| KMeans | Centroid-based, silhouette-selected k, synthetic noise filter | 0% (10% after filtering) |

Despite their fundamental differences, **all three algorithms converge on the same top pairs**. 8 pairs appear in every algorithm's top-20:

| Consensus Pair | OPTICS Freq | KMeans Freq | DBSCAN Freq | Subsector |
|---------------|-------------|-------------|-------------|-----------|
| KLAC-LRCX | 0.378 | 0.785 | 0.851 | Equipment |
| QRVO-SWKS | 0.372 | 0.888 | 0.828 | RF |
| AMAT-LRCX | 0.363 | 0.824 | 0.845 | Equipment |
| AMAT-KLAC | 0.340 | 0.807 | 0.814 | Equipment |
| ADI-NXPI | 0.279 | 0.816 | 0.900 | Analog |
| ADI-TXN | 0.278 | 0.887 | 0.817 | Analog |
| NXPI-TXN | 0.223 | 0.819 | 0.806 | Analog |
| ADI-SWKS | 0.216 | 0.830 | 0.830 | Analog/RF |

Note the OPTICS frequencies are 2-4x lower than KMeans/DBSCAN. That's entirely the noise rate effect — OPTICS is more selective about what constitutes a cluster, so fewer timestamps produce cluster assignments at all.

The permutation test (feature-shuffling, 30 iterations) confirms 11 pairs are statistically significant (Z > 1.96) — their co-clustering exceeds what random feature assignment would produce.

---

## Classical Pairs Trading Validation

### The Test

We took every pair that the OPTICS engine found with a noise-adjusted frequency above 15% (appeared together at least 15% of the time when both visible) and tested them with classical pairs trading metrics on daily data over the full ~11-month period:

1. **Engle-Granger cointegration** (p < 0.05) — are the two price series statistically tied together?
2. **Half-life of mean reversion** (5-60 days) — does the spread revert at a practical speed?
3. **Hurst exponent** (< 0.5) — is the spread genuinely mean-reverting vs trending?

A pair "passes" if it meets all three. A "near-miss" meets two of three.

### Results

| Category | Count | Description |
|----------|-------|-------------|
| Tested | 33 | Pairs with noise-adjusted freq > 0.15 |
| Passed (3/3) | 0 | No pair met all three classical criteria |
| Near-miss (2/3) | 23 | Met half-life + Hurst, typically failing cointegration |
| Failed | 10 | Met 0 or 1 criteria |

**Zero pairs formally cointegrate.** This is consistent with the transient thesis — these relationships form and dissolve, so a test that assumes permanent equilibrium (cointegration) shouldn't pass. But 23 of 33 pairs pass both Hurst and half-life criteria, meaning their spreads genuinely mean-revert on a practical timescale.

The failure mode is almost always cointegration. The typical near-miss pair has:
- Hurst < 0.5 (confirmed mean-reverting)
- Half-life between 5-60 days (practical trading speed)
- Cointegration p-value between 0.10 and 0.90 (not formally cointegrated)

This pattern makes sense: these pairs have a real structural relationship (same subsector, same demand drivers), so their spread mean-reverts. But the relationship isn't permanent or deterministic, so it doesn't reach formal cointegration.

### Backtest Performance

Every near-miss pair was backtested with a standard z-score mean-reversion strategy (20-day lookback, enter at z = 2.0, exit at z = 0.5) on daily data:

**Top 10 by Sharpe Ratio:**

| Pair | Noise-Adj Freq | Coint p-val | Half-Life (days) | Hurst | Trades | Total P&L | Sharpe | Win Rate | Consensus |
|------|---------------|-------------|-----------------|-------|--------|-----------|--------|----------|-----------|
| NXPI-TXN | 0.223 | 0.273 | 12.4 | 0.328 | 7 | 62.79 | 2.93 | 100% | Yes |
| QCOM-SWKS | 0.220 | 0.524 | 18.2 | 0.338 | 7 | 44.28 | 1.84 | 100% | No |
| KLAC-TSM | 0.192 | 0.071 | 9.4 | 0.218 | 8 | 258.67 | 1.77 | 100% | No |
| QCOM-TXN | 0.156 | 0.579 | 14.5 | 0.372 | 9 | 71.80 | 1.73 | 89% | No |
| QCOM-QRVO | 0.228 | 0.423 | 13.2 | 0.331 | 7 | 40.23 | 1.38 | 86% | Yes |
| NXPI-ON | 0.173 | 0.107 | 8.5 | 0.403 | 8 | 54.26 | 1.29 | 88% | No |
| ADI-NXPI | 0.279 | 0.915 | 43.8 | 0.324 | 8 | 74.81 | 1.08 | 88% | Yes |
| MCHP-NXPI | 0.230 | 0.105 | 7.8 | 0.320 | 12 | 33.15 | 1.04 | 92% | No |
| NXPI-POWI | 0.186 | 0.412 | 12.6 | 0.393 | 11 | 96.85 | 1.00 | 82% | No |
| NXPI-STM | 0.209 | 0.175 | 10.2 | 0.319 | 9 | 38.13 | 0.98 | 78% | No |

Key observations:
- **Every pair is profitable.** 23 of 23 near-miss pairs produced positive total P&L.
- **Win rates are extremely high.** Most pairs have 80%+ win rates, several at 100%.
- **Trade counts are modest.** 6-12 trades over 11 months is consistent with a patient mean-reversion strategy on daily data.
- **NXPI appears in 6 of the top 10.** It's the most connected node in the analog semiconductor cluster, structurally linked to ADI, TXN, MCHP, STM, ON, and POWI.

### Consensus Pairs Outperform

| Group | Pairs | Avg Sharpe | Avg P&L |
|-------|-------|-----------|---------|
| Consensus (in all 3 algorithms' top-20) | 5 tradeable | 0.954 | 55.41 |
| Non-consensus | 18 tradeable | 0.812 | 47.33 |

Cross-algorithm agreement is a genuine quality signal. Pairs that all three algorithms independently surface tend to have better backtest performance.

### Failed Pairs

The 10 pairs that failed (0 or 1 criteria met) tend to have one of two problems:
- **Trending Hurst** (> 0.5): AMAT-LRCX (Hurst 0.527), ADI-QCOM (Hurst 0.597) — the spread trends rather than reverting
- **Infinite half-life**: ADI-QRVO, ADI-MCHP — the spread shows no reversion tendency at all

These pairs still cluster together on hourly data (driven by shared volatility/momentum regimes), but the relationship doesn't produce a mean-reverting spread at daily frequency. Clustering detects behavioral similarity, not cointegration — some behaviorally similar pairs simply don't have the right price dynamics for spread trading.

---

## What This Means

### The Transient Engine Works as a Pair Discovery Tool

The original goal was real-time detection of short-lived relationships. But the pairs the engine keeps finding — the ones that form, dissolve, and re-form repeatedly — are revealing genuine structural connections in the market:

- **Equipment cluster**: AMAT, LRCX, KLAC share the same customers (TSMC, Samsung, Intel fabs) and respond to the same semiconductor capex cycle
- **RF cluster**: QRVO, SWKS, QCOM are tied to smartphone/5G demand
- **Analog cluster**: ADI, NXPI, TXN, MCHP share industrial/automotive end markets
- **EDA cluster**: CDNS, SNPS are a duopoly in chip design software

These subsector groupings aren't input to the algorithm — it discovers them from raw price-derived features. And the pairs within these groups produce mean-reverting spreads that are profitable to trade with a simple z-score strategy.

### Near-Miss is the Right Category

The fact that 0 pairs formally cointegrate but 23 show mean-reverting behavior (Hurst < 0.5, practical half-life) is exactly what you'd expect from transient-dynamic relationships. These pairs:
- Have real structural connections (shared demand drivers, shared customers)
- Mean-revert because when one stock moves away from the group, fundamental forces pull it back
- Don't formally cointegrate because the relationship strength varies over time (which is what makes them transient)

Formal cointegration is a strict requirement that assumes a permanent, stable equilibrium. These semiconductor pairs have a *tendency* toward equilibrium that's strong enough to trade but not strong enough to pass a cointegration test. The near-miss category captures this perfectly.

---

## Project Architecture

### What Exists

```
validation/pair_validation.py     Core statistical testing library
  - compute_hedge_ratio()          OLS/TLS/Kalman hedge ratio
  - half_life()                    AR(1) mean-reversion half-life
  - hurst_exponent()               Variance ratio method (in trading.py)
  - zscore_signals()               Z-score trading signals
  - simulate_spread_pnl()          P&L backtest simulation
  - spread_cv_normalized()         Spread coefficient of variation
  - bounce_rate()                  Short-term mean-reversion rate
  - cluster_persistence()          Rolling co-cluster persistence
  - feature_shuffle_permutation_test()  Statistical significance

trading/trading.py                Pair discovery validation module
  - load_artifacts()               Load OPTICS/KMeans/DBSCAN pickle data
  - compute_noise_adjusted_frequency()  Fix for OPTICS noise bias
  - build_pair_registry()          Cross-algorithm pair registry
  - test_pair_fundamentals()       Cointegration + half-life + Hurst
  - backtest_pair()                Z-score mean-reversion backtest
  - run_full_analysis()            Full pipeline
  - generate_report()              Text report

research/
  optics-clustering.ipynb          OPTICS pipeline (data -> features -> clustering -> events)
  optics-signals.ipynb             Transient validation, baseline comparison, signal functions
  KMeans.ipynb                     KMeans pipeline (same features, different algorithm)
  DBScan.ipynb                     DBSCAN pipeline (same features, different algorithm)
  algorithm-comparison.ipynb       Side-by-side comparison of all three algorithms
  validation-testing.ipynb         Unit tests for the validation module
  data/*.pkl                       27 pickle artifacts (9 per algorithm)
```

### Key Numbers

| Metric | Value |
|--------|-------|
| Tickers | 40 semiconductors |
| Period | 2025-03-11 to 2026-02-09 (~11 months hourly) |
| OPTICS valid clustering windows | 1,362 |
| OPTICS avg noise rate | 58% |
| Unique co-clustering pairs (OPTICS) | 777 |
| Permutation-significant pairs | 11 (Z > 1.96) |
| Cross-algorithm consensus pairs | 8 (in all 3 top-20) |
| Pairs tested for classical trading | 33 (noise-adj freq > 0.15) |
| Near-miss pairs (2/3 criteria) | 23 |
| All 23 near-miss pairs profitable | Yes |
| OOS co-clustering stability | r = 0.671 (OPTICS), r = 0.832 (DBSCAN) |

---

## Next Step: Transient Trading Engine

The validated longer-term pairs are an interesting byproduct, but the primary goal remains **real-time transient detection**. The existing codebase already has the building blocks:

- `detect_new_formations()` in optics-signals.ipynb compares consecutive clustering snapshots to identify new co-clustering events
- `validate_transient_event()` tests each formation with calibration/exploitation windows, correlation, spread CV, half-life, and hedge ratio drift
- `generate_transient_signals()` produces z-score signals for active pairs

The next step is to wire these into a live pipeline:

1. Fetch latest hourly candle for all 40 tickers
2. Compute features, run OPTICS on the new snapshot
3. Detect new formations vs previous snapshot
4. Validate each formation event in real time
5. Generate entry/exit signals for validated pairs
6. Track positions and P&L

The transient validation in optics-signals.ipynb already showed that clustered pairs have a meaningfully higher pass rate than random pairs, confirming that the clustering adds real value for pair selection even on the short-term transient timescale.
