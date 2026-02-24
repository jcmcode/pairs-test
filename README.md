# Clustering-Driven Pair Discovery

Transient correlation detection using unsupervised clustering for pairs trading. Two-phase research project: proof of concept on 40 semiconductors, then extended to 142 tickers across 5 sectors.

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook
```

## Repository Structure

```
pairs-test/
│
├── validation/                     Core statistical testing library
│   └── pair_validation.py          validate_pair(), hedge ratios, permutation tests
│
├── signals/                        Signal detection and feature engineering
│   ├── features.py                 9-feature computation (vol, beta, RSI, regime shifts)
│   ├── detection.py                Clustering pipeline, formation/dissolution events
│   ├── transient.py                Transient event validation
│   └── stable.py                   Stable pair tracking
│
├── trading/                        Backtesting and trading logic
│   └── trading.py                  Z-score strategy, walk-forward validation, pair analysis
│
├── screener/                       Phase 2: Cross-sector screening module
│   ├── screening.py                Ticker screening via yfscreen
│   ├── universe.py                 Universe construction (3-layer pipeline)
│   ├── analysis.py                 5-test validation framework
│   ├── enhanced_backtest.py        Adaptive z-score, Kalman hedge, transaction costs
│   ├── features_adapter.py         Feature computation adapter for screener
│   ├── config.py                   Screener configuration
│   └── notebooks/                  Analysis pipeline (run in order)
│       ├── 01-screen-universe.ipynb    Screen sectors, build combined universe
│       ├── 02-clustering.ipynb         Run OPTICS on full universe
│       ├── 03-signals-validation.ipynb Transient + stable validation
│       ├── 04-algorithm-comparison.ipynb   Multi-algorithm consensus + permutation test
│       └── 05-cross-sector-comparison.ipynb  5-test validation, backtesting, final results
│
├── research/                       Phase 1: Semiconductor exploratory notebooks
│   ├── optics-clustering.ipynb     OPTICS pipeline (features → clustering → events)
│   ├── optics-signals.ipynb        Transient validation and signal generation
│   ├── KMeans.ipynb                KMeans pipeline on same features
│   ├── DBScan.ipynb                DBSCAN pipeline on same features
│   └── algorithm-comparison.ipynb  Side-by-side comparison of all three algorithms
│
├── deliverables/                   Final outputs
│   ├── results_summary.md          Comprehensive results (both phases)
│   ├── build_presentation.py       Generates presentation.pptx (25 slides)
│   └── report.tex                  LaTeX technical report
│
├── tests/                          Test suite
│   ├── test_validation.py          Validation module tests
│   ├── test_signals.py             Signal detection tests
│   ├── test_trading.py             Trading module tests
│   └── test_config.py              Configuration tests
│
├── config.py                       Global configuration
├── requirements.txt                Python dependencies
└── CLAUDE.md                       AI assistant context
```

## Project Phases

### Phase 1: Semiconductors (research/)

Proof of concept on 40 semiconductor tickers. Three clustering algorithms (OPTICS, DBSCAN, KMeans) converge on 8 consensus pairs reflecting genuine subsector structure. 14/26 tradeable pairs profitable out of sample (54%).

### Phase 2: Cross-Sector (screener/)

Extension to 142 tickers across Technology, Healthcare, Energy, Financial Services, and Industrials. Key developments:

- **5-test scored validation** replaces cointegration (0% pass rate to 58.4% tradeable)
- **3,712 pairs** scored: 596 strong, 1,572 moderate
- **Enhanced backtesting**: Kalman hedge ratios (59% profitable), adaptive z-score optimization, 10bps transaction costs
- **Walk-forward validation**: RUN-VNET Sharpe 3.47, NVDA-ORCL 2.59
- **Permutation testing**: 943/8,086 pairs statistically significant (Z > 1.96)

## Key Results

See `deliverables/results_summary.md` for comprehensive results, or `deliverables/report.tex` for the full technical report.

## Running the Analysis

**Phase 1 (semiconductors):** Open notebooks in `research/` starting with `optics-clustering.ipynb`.

**Phase 2 (cross-sector):** Run screener notebooks in order:
```
01-screen-universe.ipynb → 02-clustering.ipynb → 03-signals-validation.ipynb
→ 04-algorithm-comparison.ipynb → 05-cross-sector-comparison.ipynb
```

**Rebuild presentation:** `python3 deliverables/build_presentation.py`

**Run tests:** `pytest tests/`
