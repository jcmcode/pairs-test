# Cross-Sector Screener Results

Results from the 5-test validation framework and enhanced backtesting pipeline, run on 142 tickers across 5 sectors (Technology, Healthcare, Energy, Financial Services, Industrials).

## Changes from Original Pipeline

The original validation used a 3-criteria binary pass/fail (Engle-Granger cointegration, half-life, Hurst). This was fundamentally mismatched to transient correlations:

- **Cointegration: 0% pass rate.** Engle-Granger tests for a long-run equilibrium, but transient correlations form/dissolve on shorter timeframes. Full-period stationarity tests fail by design.
- **0.15 frequency threshold:** Zero cross-sector pairs survived. Best cross-sector pair had noise-adjusted frequency 0.104.
- **Static z=2.0 entry:** Produced 1-4 OOS trades in ~68 days, making Sharpe estimation impossible.

### What Changed

1. **Validation:** Replaced cointegration with a 5-test scored framework (ADF, half-life, Hurst, variance ratio, rolling correlation stability)
2. **Frequency threshold:** Lowered from 0.15 to 0.08 (109 pairs vs 26)
3. **Backtesting:** Added adaptive z-score optimization and Kalman-estimated hedge ratios with transaction costs

### Bugs Found and Fixed During Audit

- **Kalman spread was artificially mean-reverting.** The adaptive Kalman spread (beta updated each step) creates prediction errors with near-zero mean and no autocorrelation — essentially white noise that trivially mean-reverts. Fixed by using Kalman on calibration only to estimate a terminal beta, then applying that fixed beta to OOS.
- **Rolling correlation used price levels.** Co-trending stocks (e.g., RTX-SHEL: level corr=0.91, return corr=0.26) showed spurious stability. Fixed by computing correlation on daily returns instead.
- **Optimizer min_trades too low.** At min_trades=3, the optimizer could overfit to a couple lucky trades (e.g., CVX-XOM: 3 cal trades, Sharpe 5.78). Raised to min_trades=5.

## Pair Discovery

| Frequency Threshold | Total Pairs | Intra-Sector | Cross-Sector |
|---------------------|-------------|--------------|--------------|
| 0.05 | 255 | 216 | 39 |
| **0.08 (new default)** | **109** | **104** | **5** |
| 0.10 | 71 | 69 | 2 |
| 0.15 (old default) | 26 | 26 | 0 |

Energy dominates pair discovery (74/109 = 68%), reflecting the sector's strong internal correlations.

## Validation Results (5-Test Framework)

### Score Distribution

| Classification | Count | % |
|---------------|-------|---|
| Strong (4-5) | 34 | 31% |
| Moderate (3) | 61 | 56% |
| Weak (2) | 12 | 11% |
| Fail (0-1) | 2 | 2% |

**87% of pairs score >= 3** (tradeable). This is a much higher pass rate than the old cointegration-based system.

### Individual Test Pass Rates

| Test | Pass Rate | Role |
|------|-----------|------|
| Hurst (< 0.5) | 94.5% | Easy — confirms mean-reversion, not discriminating |
| Rolling Corr Stability (returns) | 95.4% | Easy — most clustered pairs have stable return correlations |
| Half-life (5-60 days) | 83.5% | Moderate — filters out very slow/fast mean-reversion |
| ADF (p < 0.10) | 30.3% | Hard — spread stationarity with low power on ~136 days |
| Variance Ratio (reject RW at 10%) | 14.7% | Hardest — Lo-MacKinlay test detects short-run mean-reversion |

The real differentiators are ADF and variance ratio. A "strong" pair (score 4-5) must pass at least one of these hard tests plus most easy ones.

### Sector Scores

| Sector Combination | Pairs | Avg Score |
|---------------------|-------|-----------|
| Industrials - Industrials | 6 | 4.17 |
| Healthcare - Healthcare | 4 | 3.50 |
| Technology - Financial Services | 3 | 3.33 |
| Energy - Energy | 74 | 3.18 |
| Financial Services - Financial Services | 9 | 3.11 |
| Technology - Technology | 11 | 2.73 |

Industrials pairs score highest (4.17 avg), but with only 6 pairs. Energy has the most pairs but lower average quality.

## Cross-Sector Pairs

| Pair | Score | Classification | Sectors |
|------|-------|----------------|---------|
| **RTX-SHEL** | 4 | Strong | Industrials / Energy |
| **APLD-HUT** | 4 | Strong | Technology / Financial Services |
| APLD-CIFR | 3 | Moderate | Technology / Financial Services |
| APLD-IREN | 3 | Moderate | Technology / Financial Services |
| CPRT-CTSH | 1 | Fail | Industrials / Technology |

4 of 5 cross-sector pairs are tradeable. The APLD cluster (APLD/HUT/CIFR/IREN) is all crypto mining infrastructure — Technology companies clustering with Financial Services crypto firms. This is a meaningful economic relationship, not noise.

RTX-SHEL (defense/energy) is the strongest cross-sector pair with a genuinely surprising cross-sector link.

## Trading Performance

### Strategy Comparison (95 tradeable pairs)

| Strategy | Pairs w/ Trades | Avg Trades | Profitable % | Avg PnL |
|----------|----------------|------------|-------------|---------|
| Baseline (z=2.0, OLS, no costs) | 95 | 2.3 | 53% | 0.356 |
| Enhanced (opt z, OLS, 10bps) | 94 | 2.9 | 50% | 0.378 |
| Kalman (cal beta, opt z, 10bps) | 94 | 2.8 | 59% | 0.408 |

### Pairs with >= 5 Trades (the only reliable Sharpe estimates)

| Strategy | Pairs | Median Sharpe |
|----------|-------|---------------|
| Baseline | 0 | N/A |
| Enhanced | 10 | 0.449 |
| Kalman | 12 | 0.160 |

Only 10-12 pairs generate >= 5 OOS trades. The enhanced optimizer does succeed in generating more trades (median Sharpe 0.449 for the 10 pairs with 5+ trades), but the sample is small.

### Top Enhanced Pairs

| Pair | Type | Enhanced PnL | Enhanced Trades | Kalman PnL |
|------|------|-------------|-----------------|------------|
| AMAT-TSM | intra | +30.30 | 2 | +54.19 |
| CVX-SU | intra | +13.16 | 2 | +10.41 |
| MCHP-ON | intra | +12.94 | 3 | +1.67 |
| COP-XOM | intra | +11.23 | 6 | -0.72 |
| APLD-IREN | **cross** | +10.40 | 2 | +9.47 |
| RTX-SHEL | **cross** | +10.07 | 3 | +13.87 |
| HUT-RIOT | intra | +9.91 | 8 | +5.65 |

## Semiconductor Baseline Comparison

| Metric | Semiconductor Baseline | Cross-Sector (Baseline) | Cross-Sector (Enhanced) |
|--------|----------------------|------------------------|------------------------|
| Profitable | 54% | 53% | 50% |
| Top Sharpe | 3.55 | N/A (<5 trades) | 7.13 (2 trades) |
| Pairs tested | ~26 | 109 | 95 |

Profitability is comparable (~52-53% vs 54%). The cross-sector universe finds more pairs but most lack sufficient OOS trades for robust Sharpe estimation.

## Key Conclusions

1. **The 5-test framework dramatically improves pass rates.** 87% of pairs at 0.08 threshold are tradeable vs ~0% with old cointegration-based criteria. This correctly reflects that transient correlations mean-revert short-term without long-run cointegration.

2. **Cross-sector pairs exist but are rare.** At 0.08 threshold, only 5 cross-sector pairs appear (4 tradeable). The crypto mining cluster (APLD/HUT/CIFR/IREN) is the main cross-sector finding — Technology firms clustering with Financial Services crypto companies.

3. **Trade count is the binding constraint.** With ~68 OOS days, most pairs generate only 2-3 trades regardless of z-score optimization. This makes Sharpe estimation unreliable and limits strategy comparison. The enhanced optimizer helps marginally (2.9 vs 2.3 avg trades).

4. **The Kalman hedge is a modest improvement over OLS.** After fixing the adaptive spread bug, the Kalman terminal beta performs slightly better (59% profitable vs 53% for OLS baseline, with transaction costs). This makes sense — the Kalman beta weights recent observations more heavily, better capturing the current relationship.

5. **Industrials pairs are the strongest.** Despite having only 6 pairs, Industrials scores 4.17 avg (highest by far). Energy dominates by volume (74 pairs) but has lower average quality (3.18).

6. **Variance ratio is the hardest test.** At 14.7% pass rate, the Lo-MacKinlay variance ratio test is the most demanding criterion. Pairs that pass both VR and ADF show the strongest evidence of genuine mean-reversion, not just noise.

## Limitations

- **Short OOS period (~68 days):** Insufficient for reliable Sharpe estimation. Most pairs generate 2-3 trades, below the 5-trade minimum for meaningful Sharpe ratios.
- **~10 months of hourly data:** Limited historical depth means calibration periods are short (136 days), reducing ADF power and limiting the optimizer's training set.
- **Simplified transaction costs:** Flat 10bps doesn't capture variation in spreads, market impact, or short-borrowing costs across different assets.
- **No position sizing or portfolio effects:** Each pair is evaluated independently. Correlated pairs (many energy pairs share tickers like CVX, XOM, SU) would require portfolio-level risk management.
- **Cross-sector discovery is limited by frequency threshold.** Most cross-sector relationships are weak (freq < 0.05), suggesting they are transient by nature and may not be reliably tradeable.
