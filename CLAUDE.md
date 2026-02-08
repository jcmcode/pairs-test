# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an event-driven correlation detection research project. The goal is to identify short-term relationship formations between assets using clustering algorithms, analyze historical data to understand what factors preceded those formations, and eventually predict when new relationships will form.

Unlike traditional pairs trading (which seeks cointegrated pairs with mean reversion), this project looks for **transient correlations** that form on shorter time frames, often driven by news events or market conditions. When two assets cluster together, it signals a relationship has formed; the validation module then tests the strength and persistence of that relationship.

### Architecture

The project is organized into three main components:

1. **Validation Module** (`validation/pair_validation.py`): Core statistical testing library
   - Cluster persistence: Measures how consistently two assets remain in the same cluster over a rolling window. Higher persistence = stronger, more reliable relationship.
   - Spread computation: Calculates the spread between two assets using a beta hedge ratio
   - Mean reversion metrics: Computes autocorrelation (AR(1) lag-1 and lag-1 φ coefficient) and bounce rate to assess the quality of the relationship
   - Main entry point: `validate_pair()` function that computes all metrics for a single pair at a given timestamp

2. **Research Notebooks** (`research/`): Exploratory analysis and algorithm testing
   - Clustering algorithms: KMeans, DBScan, OPTICS (variants for different parameter tuning)
     - **OPTICS** is the primary algorithm of interest—best at detecting variable-density clusters and short-term relationship formations
   - Pair analysis: gold-vs-dxy investigates specific commodity pairs as test cases
   - Validation testing: validation-testing.ipynb validates the pair_validation module logic

3. **Trading Module** (`trading/trading.py`): Placeholder for future trading execution logic

### Key Concepts

- **Relationship Formation**: Two assets that cluster together (same cluster ID at the same timestamp) indicate a relationship has formed. The validation module measures how strong and persistent this relationship is.
- **Cluster Persistence**: A relationship is more reliable if both assets stay in the same cluster consistently over a rolling window H (default 5 periods).
- **Mean Reversion Metrics**: AR(1) φ coefficient and lag-1 autocorrelation quantify how quickly the spread reverts toward equilibrium.
- **Bounce Rate**: Measures the proportion of cases where the spread reverts by at least 30% within a 2-period horizon, indicating the relationship has predictable short-term mean reversion behavior.

### Workflow

1. Run clustering algorithms (KMeans, DBScan, OPTICS) on historical price data with rolling windows
2. Identify timestamps where two assets cluster together (relationship formation events)
3. Use the validation module to measure cluster persistence, mean reversion strength, and other metrics for each event
4. Analyze the characteristics of strong relationship formations (higher persistence, better bounce rates)
5. Look back at what preceded these events to identify leading indicators
6. Build predictive models to forecast future relationship formations

## Development Environment

Install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, yfinance, scikit-learn, statsmodels, pandas-ta-classic, pykalman, jupyter, matplotlib, seaborn, plotly

## Common Commands

**Run notebooks**: `jupyter notebook` then navigate to research folder

**Test pair validation**: Use `validation/pair_validation.py` with test data in notebooks, or write a test script that imports and calls `validate_pair()` with sample DataFrames

**Explore pair validation metrics**: validation-testing.ipynb demonstrates how the validation module works and shows example metrics

## Important Notes for Implementation

- The `validate_pair()` function expects aligned timestamps between `prices` DataFrame and `cluster_history` DataFrame
- All validation metrics return `np.nan` if insufficient data; handle this in calling code
- The bounce rate calculation uses rolling z-scores, not pre-optimized thresholds; this avoids look-ahead bias
- The `drop_noise_now` parameter in `validate_pair()` filters pairs where either asset has cluster_id == -1 (DBSCAN noise cluster)
- Short-term relationship detection: This project differs from cointegration analysis. Focus on clusters that form and dissolve relatively quickly, not long-term equilibrium relationships
- Algorithm comparison: All three clustering algorithms (KMeans, DBScan, OPTICS) should be tested on the same historical data to compare which best captures event-driven relationship formations
- OPTICS advantages: Variable density clusters, better at finding clusters that form transiently (which is the target use case)
