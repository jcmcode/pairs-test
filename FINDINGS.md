# Findings: Transient Clustering as a Pair Discovery Tool

> **Methodology note:** All backtest results use out-of-sample data (last 33% of daily observations). Hedge ratios and statistical metrics are estimated on the calibration window (first 67%) only. Cointegration is tested in both directions (min p-value). This eliminates look-ahead bias present in earlier versions of this analysis.

## The Key Result

The OPTICS transient detection engine was built to find short-lived relationships between semiconductor stocks as they form in real time. The pairs it keeps surfacing turn out to have genuine longer-term structural relationships — **26 of 33 filtered pairs pass at least 2 of 3 classical pairs trading criteria** (half-life, Hurst exponent, cointegration) on the calibration window.

However, out-of-sample backtesting paints a more nuanced picture: roughly half of tradeable pairs are profitable OOS, while others show losses. The engine is effective as a **pair discovery tool** — it reliably identifies structurally related pairs with mean-reverting spreads — but not every discovered pair translates to profitable trading without further filtering.

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

We took every pair that the OPTICS engine found with a noise-adjusted frequency above 15% (appeared together at least 15% of the time when both visible) and tested them with classical pairs trading metrics. The daily data is split 67/33 into calibration and out-of-sample windows:

1. **Engle-Granger cointegration** (p < 0.05, tested both directions) — are the two price series statistically tied together?
2. **Half-life of mean reversion** (5-60 days) — does the spread revert at a practical speed?
3. **Hurst exponent** (< 0.5) — is the spread genuinely mean-reverting vs trending?

All metrics are computed on the calibration window only. Backtests run on the OOS window. A pair "passes" if it meets all three. A "near-miss" meets two of three.

### Results

| Category | Count | Description |
|----------|-------|-------------|
| Tested | 33 | Pairs with noise-adjusted freq > 0.15 |
| Passed (3/3) | 4 | Met cointegration + half-life + Hurst on calibration window |
| Near-miss (2/3) | 22 | Typically met half-life + Hurst, failing cointegration |
| Failed | 7 | Met 0 or 1 criteria |

With bidirectional cointegration testing (min p-value of both orderings), 4 pairs now formally pass all three criteria — ADI-NXPI, ADI-QRVO, ADI-SWKS, and QCOM-QRVO. The remaining 22 near-misses typically pass Hurst and half-life but fail cointegration, consistent with the transient thesis: these relationships form and dissolve, so a test assuming permanent equilibrium often fails.

The typical near-miss pair has:
- Hurst < 0.5 (confirmed mean-reverting)
- Half-life between 5-60 days (practical trading speed)
- Cointegration p-value between 0.10 and 0.90 (not formally cointegrated on the calibration window)

### Out-of-Sample Backtest Performance

All passed and near-miss pairs were backtested with a standard z-score mean-reversion strategy (20-day lookback, enter at z = 2.0, exit at z = 0.5) on the **OOS window only** (~33% of daily data, approximately the last 3.5 months):

**Top 10 by Sharpe Ratio (OOS):**

| Pair | Noise-Adj Freq | Coint p-val | Half-Life (days) | Hurst | Trades | Total P&L | Sharpe | Win Rate | Consensus |
|------|---------------|-------------|-----------------|-------|--------|-----------|--------|----------|-----------|
| NXPI-ON | 0.173 | 0.271 | 8.6 | 0.413 | 2 | 6.56 | 3.55 | 100% | No |
| CDNS-SNPS | 0.199 | 0.620 | 21.5 | 0.447 | 3 | 36.26 | 2.08 | 100% | No |
| QCOM-QRVO | 0.228 | 0.019 | 5.3 | 0.205 | 2 | 47.46 | 1.44 | 100% | No |
| KLAC-TSM | 0.192 | 0.006 | 3.6 | 0.181 | 2 | 47.55 | 1.35 | 100% | No |
| LRCX-NVMI | 0.152 | 0.422 | 13.2 | 0.440 | 2 | 16.19 | 1.04 | 100% | No |
| NXPI-POWI | 0.186 | 0.568 | 13.5 | 0.372 | 5 | 37.32 | 0.85 | 80% | No |
| MCHP-ON | 0.200 | 0.039 | 5.2 | 0.313 | 3 | 3.62 | 0.65 | 67% | No |
| ADI-TXN | 0.278 | 0.875 | 52.4 | 0.422 | 3 | 10.32 | 0.33 | 33% | Yes |
| MCHP-NXPI | 0.230 | 0.237 | 8.6 | 0.332 | 2 | 2.00 | 0.33 | 50% | No |
| QRVO-SWKS | 0.372 | 0.232 | 9.2 | 0.409 | 2 | 0.62 | 0.28 | 50% | Yes |

Key observations:
- **Roughly half of tradeable pairs are profitable OOS.** 14 of 26 tradeable pairs produced positive P&L on out-of-sample data, a significant reduction from the earlier in-sample results.
- **Trade counts are low.** The OOS window is shorter (~3.5 months), yielding 2-5 trades per pair — enough for directional signal but not statistically robust.
- **Top performers remain strong.** NXPI-ON (Sharpe 3.55), CDNS-SNPS (2.08), QCOM-QRVO (1.44) show genuine OOS profitability.
- **Some pairs that looked great in-sample now show losses.** ADI-NXPI (Sharpe -1.76), ADI-QRVO (-3.04), AVGO-TSM (-0.76) — the hedge ratios fitted on calibration data didn't hold in the OOS window.
- **The honest picture**: the engine finds structurally related pairs, but only a subset produce profitable OOS trading results with a simple z-score strategy.

### Consensus vs Non-Consensus (OOS)

| Group | Tradeable Pairs | Passed | Near-miss | Avg Sharpe (OOS) | Avg P&L (OOS) |
|-------|----------------|--------|-----------|-----------------|---------------|
| Consensus (in all 3 algorithms' top-20) | 6 | 2 | 4 | -0.251 | -13.73 |
| Non-consensus | 20 | 2 | 18 | 0.056 | -8.76 |

In the OOS window, consensus pairs actually underperform non-consensus pairs on average. This reversal from the earlier in-sample results illustrates the danger of look-ahead bias: the consensus signal appeared strongest precisely because those pairs had the most in-sample overfitting. With honest OOS evaluation, cross-algorithm agreement does not reliably predict better backtest performance — though it does reliably identify structurally related pairs.

### Failed Pairs

The 7 pairs that failed (0 or 1 criteria met on calibration data) tend to have:
- **Trending Hurst** (> 0.5): AMAT-LRCX (0.571), AMAT-KLAC (0.626), AMAT-ASML (0.590) — equipment pairs where the spread trends rather than reverting
- **Long half-life**: STX-WDC (96.6 days) — technically mean-reverting but too slow to trade

These pairs still cluster together on hourly data (driven by shared volatility/momentum regimes), but the relationship doesn't produce a mean-reverting spread at daily frequency on the calibration window.

---

## What This Means

### The Transient Engine Works as a Pair Discovery Tool — With Caveats

The original goal was real-time detection of short-lived relationships. The pairs the engine keeps finding reveal genuine structural connections in the market:

- **Equipment cluster**: AMAT, LRCX, KLAC share the same customers (TSMC, Samsung, Intel fabs) and respond to the same semiconductor capex cycle
- **RF cluster**: QRVO, SWKS, QCOM are tied to smartphone/5G demand
- **Analog cluster**: ADI, NXPI, TXN, MCHP share industrial/automotive end markets
- **EDA cluster**: CDNS, SNPS are a duopoly in chip design software

These subsector groupings aren't input to the algorithm — it discovers them from raw price-derived features. The pairs within these groups produce mean-reverting spreads on the calibration window, and about half remain profitable on out-of-sample data.

### The Look-Ahead Bias Lesson

Earlier versions of this analysis computed hedge ratios on the full data period and backtested on the same data, producing uniformly positive results (23/23 pairs profitable, 80-100% win rates). After implementing a proper calibration/OOS split:
- **14 of 26 tradeable pairs remain profitable** (down from 23/23)
- **Average Sharpe dropped significantly** — some pairs that appeared excellent in-sample (ADI-NXPI Sharpe 1.08) show losses OOS (Sharpe -1.76)
- **The top performers are genuine**: NXPI-ON, CDNS-SNPS, QCOM-QRVO, KLAC-TSM show strong OOS Sharpe ratios (1.3-3.6)

This demonstrates why out-of-sample validation is essential. The engine reliably discovers structurally related pairs, but a simple z-score strategy with a fixed hedge ratio doesn't always survive the OOS test.

### Near-Miss Remains the Right Category

With bidirectional cointegration testing, 4 pairs now formally pass all criteria. But the majority (22/26) are near-misses — mean-reverting behavior without formal cointegration. This is consistent with transient dynamics: the relationship is real but time-varying, strong enough to detect via clustering but not permanent enough for the Engle-Granger test.

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
| Passed (3/3 criteria on cal window) | 4 |
| Near-miss pairs (2/3 criteria) | 22 |
| Tradeable pairs profitable OOS | 14 of 26 (54%) |
| Top OOS Sharpe | NXPI-ON: 3.55, CDNS-SNPS: 2.08 |
| Methodology | 67% cal / 33% OOS split, bidirectional cointegration |

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
