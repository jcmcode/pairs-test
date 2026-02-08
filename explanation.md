# OPTICS Clustering & Prediction Notebooks: Complete Explanation

## Overview

This document explains the two main OPTICS-based analysis notebooks in the research folder:

1. **`optics-clustering-run.ipynb`** - The main clustering and validation pipeline
2. **`optics-prediction.ipynb`** - Factor attribution and predictive modeling based on clustering results

These notebooks implement a workflow to detect transient relationship formations between assets using the OPTICS clustering algorithm, validate the strength of discovered relationships, and then predict when future formations will occur.

---

## High-Level Workflow

```
optics-clustering-run.ipynb
    ↓
[Data Fetch] → [Feature Engineering] → [OPTICS Clustering]
    ↓
[Formation/Dissolution Detection] → [Correctness Checks]
    ↓
[Pair Validation]
    ↓
[Save Data to Pickle Files]
    ↓
optics-prediction.ipynb
    ↓
[Load Data] → [Factor Attribution] → [Predictive Model] → [Results]
```

---

# OPTICS-CLUSTERING-RUN.ipynb

## Purpose
This is the **main analysis notebook** that:
- Fetches historical price data for a portfolio of semiconductor and tech stocks
- Engineers clustering features (momentum, volatility, correlation, etc.)
- Applies the OPTICS clustering algorithm with rolling windows
- Detects when pairs START and STOP clustering together (formation/dissolution events)
- Validates the quality and tradability of discovered relationships
- Saves results for the prediction notebook

---

## Cell-by-Cell Breakdown

### Cell 1: Imports
**What it does**: Loads all required libraries

**Key imports**:
- `yfinance` - Fetch price data
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning (OPTICS, PCA, preprocessing)
- `statsmodels` - Statistical testing (cointegration, Granger causality)
- `matplotlib`, `seaborn`, `plotly` - Visualization

---

### Cells 2-4: Data Fetch

**What it does**: Fetches historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance

**Assets included**:
- **S&P 500** (^GSPC) - Used as market beta reference
- **Major semiconductor & chip companies**: NVDA, TSM, AMD, INTEL, AVGO, QCOM, ASML, etc.
- **Memory/storage companies**: MU, WDC, STX
- **Design software/IP**: ARM, SNPS, CDNS
- **Other tech**: Various specialized semiconductor suppliers

**Why these stocks?**: The semiconductor sector is ideal for transient clustering detection because:
- Companies in this sector move together during tech booms/busts
- News events (earnings, chip shortages) create transient correlations
- Relationships form and dissolve relatively quickly (not permanently cointegrated)

**Data period**: Typically downloads 1-2 years of daily or hourly data

---

### Cells 5: Feature Engineering

**What it does**: Creates clustering features designed to capture market regimes and transient conditions

**Feature categories created**:

#### 1. **Momentum Features**
- Returns (simple percentage change)
- Rate of change (ROC) - momentum over different lookback windows
- Moving average convergence divergence (MACD)
- Relative Strength Index (RSI) - momentum oscillator (0-100 scale)

**Why**: When two stocks are in similar momentum states, they're more likely to cluster together

#### 2. **Volatility Features**
- Historical volatility (rolling std of returns)
- Average True Range (ATR) - volatility adjusted for gaps

**Why**: Volatility clustering indicates when stocks are moving in similar regime conditions

#### 3. **Correlation Features**
- Rolling correlation with S&P 500 (market beta)
- Rolling correlation between each pair of stocks

**Why**: Stocks with similar market sensitivity tend to move together

#### 4. **Trend Features**
- Highs/lows relative to moving average
- Distance from 20-day, 50-day moving averages

**Why**: Stocks in the same trend environment cluster better

#### 5. **Technical Indicators**
- Bollinger Bands (relative position)
- Stochastic oscillator
- Williams %R
- Various other momentum/volume indicators

**Output**: A DataFrame called `ts_df` with multi-index (Datetime, Ticker) and columns for each feature. Each row is a stock at a timestamp with all its feature values.

**Data structure**:
```
                        Price  Returns  Volatility  RSI  MACD  Correlation_SPX  ...
Datetime    Ticker
2024-01-01  NVDA      150.25    0.032     0.025    65.3  2.14       0.72
2024-01-01  TSMC      120.00   -0.015     0.018    48.1  0.32       0.68
2024-01-01  AMD       110.50    0.022     0.021    59.2  1.87       0.71
...
```

---

### Cells 6-7: OPTICS Clustering

**What it does**: Applies the OPTICS (Ordering Points To Identify the Clustering Structure) algorithm to detect clusters of similarly-behaving stocks

**Algorithm overview**:

OPTICS is a density-based clustering algorithm that:
1. Orders all points (stocks) by their reachability distance
2. Creates a reachability plot showing the cluster structure
3. Automatically identifies clusters at multiple density levels

**Key differences from other algorithms**:
- **KMeans**: Requires specifying number of clusters in advance; assumes spherical clusters
- **DBScan**: Fixed epsilon (density threshold); struggles with variable-density clusters
- **OPTICS** (preferred): Variable density, automatically finds clusters, excellent for transient relationships

**Why OPTICS for this project**:
- Semiconductor sector has variable-density relationships (some pairs cluster tightly, others loosely)
- Relationships form and dissolve (variable density over time)
- No need to pre-specify number of clusters
- Creates a hierarchy of clusters at different density levels

**Implementation details**:

```python
# Configure OPTICS
clusterer = OPTICS(min_samples=3, xi=0.05, min_cluster_size=3)

# Fit on feature data
clusterer.fit(features_at_timestamp)

# Get cluster labels for each stock
cluster_labels = clusterer.labels_
```

**Parameters explained**:
- `min_samples`: Minimum points in a neighborhood to form a cluster core (3-5 typical)
- `xi`: Steepness threshold in reachability plot (0.01-0.1 range; lower = more clusters)
- `min_cluster_size`: Minimum cluster size (typically 2-3)

**What gets stored**: For each timestamp, a vector of cluster IDs (e.g., [0, 0, 1, 1, 2, -1, -1, ...])
- **Cluster ID 0, 1, 2, etc.** = Valid clusters (stocks in these clusters show similar behavior)
- **Cluster ID -1** = Noise (stock didn't cluster with others at this timestamp)

**Output**: `cluster_history` DataFrame with timestamps and cluster IDs for each stock

---

### Cells 8-10: Cluster Formation & Dissolution Event Detection

**What it does**: Identifies the MOMENTS when two stocks START clustering together (formation events) and when they STOP (dissolution events)

**Why this matters**:
- Just counting "co-clustering frequency" isn't enough for prediction
- We want to detect the TRANSITION POINTS when relationships form/dissolve
- These transition points are the events we want to predict

**Algorithm**:

For each pair of stocks, track their cluster assignments across time:

```
Time   Stock1_Cluster   Stock2_Cluster   Status
T-2         0                1           NOT clustered
T-1         0                1           NOT clustered
T           0                0           FORMATION EVENT ← Formation detected!
T+1         0                0           Clustered
T+2         1                0           DISSOLUTION EVENT ← Dissolution detected!
T+3         1                0           NOT clustered
```

**Formation event**: Pair (A, B) goes from different clusters → same cluster at time T
**Dissolution event**: Pair (A, B) goes from same cluster → different clusters at time T

**Output**:
- `df_formations` - DataFrame listing all formation events with:
  - Timestamp
  - Pair ID (stock1, stock2)
  - Formation type (formation vs dissolution)
  - Duration (how long it lasted, if applicable)

---

### Cells 10: Save Data for Prediction Notebook

**What it does**: Serializes key data structures using Python pickle

**Files saved to `research/data/` folder**:
- `ts_df.pkl` - All feature data across time
- `df_formations.pkl` - Formation/dissolution events
- `pair_classification.pkl` - Which pairs ever clustered together
- `cluster_history.pkl` - Full cluster assignments at each timestamp

**Why**: Allows the prediction notebook to run independently without re-computing clustering

---

### Cells 11-12: Correctness Checks

**What it does**: Validates that the clustering results are meaningful and not random

#### 1. **Random Baseline (Permutation Test)**

Randomly shuffles cluster labels and checks if we still get "formations":
```
Real clustering: NVDA and TSMC cluster together 47 times
Random baseline: NVDA and TSMC cluster together 2 times (by chance)

Result: Real formations are 23.5x more frequent than random
        → Clustering is detecting real signal, not noise
```

**Logic**: If formations were just random, we'd see similar rates with shuffled labels

#### 2. **Out-of-Sample Split**

Splits data into train/test:
- Train on first 70% of data
- Test on last 30%
- Check if patterns from train appear in test

**Why**: Verifies that discovered relationships aren't just overfitting to historical data

#### 3. **OPTICS Sensitivity Analysis**

Runs OPTICS with different parameter values:
```
xi = 0.01  (strict): Fewer, tighter clusters
xi = 0.05  (medium): Balanced detection
xi = 0.10  (loose):  More clusters, including weak relationships
```

Then checks: Do we find similar pairs clustering across all parameter settings?

**Why**: If formations only appear with one specific parameter, they're probably artifacts

---

### Cells 13-19: Pair Validation

**What it does**: Takes the pairs that OPTICS found clustering together and validates they represent tradeable relationships

**This is where the project's core statistical library comes in** (`validation/pair_validation.py`)

#### Key Validation Metrics:

For each pair at each formation timestamp, compute:

##### 1. **Cluster Persistence** (strength metric)
Measures how consistently the pair stays in the same cluster over a rolling window (default 5 periods)

```
Example: NVDA & TSMC at timestamp T
Window: T to T+4
    T:   both in cluster 0 ✓
    T+1: both in cluster 0 ✓
    T+2: both in cluster 0 ✓
    T+3: both in cluster 1 ✗ (dissolved)
    T+4: both in cluster 2 ✗

Persistence = 3/5 = 60%
→ Relationship persisted 3 out of 5 periods (decent strength)
```

**Interpretation**:
- High persistence (>80%) = Stronger, more reliable relationship
- Low persistence (<30%) = Weak, likely to dissolve quickly

##### 2. **Spread Computation**
Calculates the price spread between two stocks using a beta-hedged ratio:

```
Spread = Price(A) - β * Price(B)

Where β is the hedge ratio estimated via linear regression
This normalizes for different price levels and volatilities
```

**Why**: The spread is what actually mean-reverts in a pairs trade

##### 3. **Mean Reversion Metrics**

###### a) **AR(1) φ (phi) Coefficient**
Measures autocorrelation in the spread at lag-1:

```
Spread(t) = φ * Spread(t-1) + noise

φ = 0.95  → Slow mean reversion (high persistence)
φ = 0.50  → Medium mean reversion
φ = 0.10  → Fast mean reversion (good for short-term trades)
```

**Interpretation**:
- Values close to 1 = Slow reversion (non-stationary, risky)
- Values close to 0 = Fast reversion (mean-reverting, tradeable)
- Typical target: 0.3-0.7 range (some persistence, but mean-reverting)

###### b) **Bounce Rate**
Proportion of times the spread reverts by at least 30% within 2 periods:

```
Example: NVDA-TSMC spread
Event: Spread reaches +2 standard deviations (widest)
Question: Within next 2 periods, does it bounce back by ≥30%?

Results across 50 similar events: 38 bounced, 12 didn't
Bounce rate = 38/50 = 76%

Interpretation: 76% of the time, spreads revert predictably
               → Good edge for mean reversion trading
```

###### c) **Cointegration Test (Johansen)**
Statistical test for whether two prices are truly cointegrated (long-term equilibrium)

```
Cointegration rank:
- Rank 0: No cointegration (prices drift apart)
- Rank 1: One cointegrating relationship (prices revert together)
- Rank 2+: Fully integrated (prices move together always)
```

##### 4. **Significance Tests**

Tests whether metrics are statistically significant (not just random):
- Cointegration p-value < 0.05
- AR(1) autocorrelation significant
- Bounce rate compared to baseline

---

### Output: Validation Results

A DataFrame with one row per tradeable pair, showing:

```python
df_validation = pd.DataFrame({
    'pair_id': ['NVDA-TSMC', 'NVDA-AMD', ...],
    'timestamp': [datetime, datetime, ...],
    'cluster_persistence': [0.80, 0.65, ...],
    'spread_mean': [1.23, -0.45, ...],
    'ar1_phi': [0.42, 0.67, ...],
    'bounce_rate': [0.76, 0.52, ...],
    'coint_pval': [0.001, 0.045, ...],
    'is_tradeable': [True, False, ...],  # Passes all checks
})
```

**Filtering tradeable pairs**:
```
Criteria:
- Cluster persistence > 60%
- AR(1) φ in range (0.2, 0.9)  # Mean-reverting but persistent
- Bounce rate > 50%             # Reliable reversion pattern
- Cointegration p-value < 0.05  # Statistically significant
```

**Result**: Top X pairs ranked by composite score (persistence × bounce_rate × coint_significance)

---

# OPTICS-PREDICTION.IPYNB

## Purpose
This notebook **answers two questions**:
1. **Factor Attribution**: What feature values are unusual when pair formations occur?
2. **Prediction**: Can we forecast future formations using features from the prior timestamp?

This notebook builds on the clustering results from `optics-clustering-run.ipynb`.

---

## Cell-by-Cell Breakdown

### Cell 1: Imports

Loads data science libraries:
- `RandomForestClassifier`, `GradientBoostingClassifier` - ML models
- `TimeSeriesSplit` - Time-aware cross-validation
- `classification_report`, `roc_auc_score`, `precision_recall_curve` - Evaluation metrics

---

### Cells 2-3: Load Data from Clustering Notebook

**What it does**: Loads pickle files saved by `optics-clustering-run.ipynb`

**Files loaded**:
- `ts_df.pkl` - Feature data (all timestamps and stocks)
- `df_formations.pkl` - Formation events
- `cluster_history.pkl` - Cluster assignments
- `pair_classification.pkl` - Pair metadata

**Why separate notebooks?**:
- Clustering is computationally expensive (1-2 hours)
- Prediction analysis is fast (<5 minutes)
- Allows iterating on prediction without re-running clustering
- Clean separation of concerns

---

### Cells 4-5: Factor Attribution

**What it does**: For each formation event, captures the feature values at that timestamp and compares them to baseline (non-formation timestamps)

**Process**:

For each formation event, collect:
```
Timestamp: T (when NVDA and TSMC formed a cluster)
Pair: (NVDA, TSMC)
Features at time T:
  - NVDA: [RSI=75, Volatility=0.032, Momentum=0.15, ...]
  - TSMC: [RSI=72, Volatility=0.029, Momentum=0.14, ...]

Average pair features: [(75+72)/2=73.5, (0.032+0.029)/2=0.0305, ...]
```

**Baseline distribution**:
For each feature, compute mean and std across all timestamps (including non-formation times)

**Z-score normalization**:
```
Z-score = (Feature_at_formation - Mean_baseline) / Std_baseline

Example:
RSI at formation events: average 73.5
RSI baseline: mean 50, std 10
Z-score = (73.5 - 50) / 10 = 2.35

Interpretation: RSI is 2.35 standard deviations above normal
               → Formations tend to occur when RSI is high
               → Stocks are in overbought territory
```

**Output**: `df_comparison` DataFrame showing:
```python
df_comparison = pd.DataFrame({
    'Feature': ['RSI', 'Volatility', 'Momentum', 'MACD', ...],
    'Mean_at_Formation': [73.5, 0.032, 0.15, ...],
    'Mean_Baseline': [50.0, 0.025, 0.08, ...],
    'Std_Baseline': [10.0, 0.008, 0.05, ...],
    'Z_Score': [2.35, 0.87, 1.40, ...],
})
```

**Interpretation**:
- Large positive Z-score = Feature is unusually high at formations
- Large negative Z-score = Feature is unusually low at formations
- Z-score near 0 = Feature has no special value at formations

---

### Cells 6-7: Factor Attribution Visualization

**What it does**: Creates plots showing which features are most predictive of formations

**Visualization 1: Z-Score Bar Chart**
```
Feature               Z-Score
RSI                   ████████ 2.35
Volatility            ███░░░░░░ 0.87
Momentum              ████░░░░░ 1.40
MACD                  ██░░░░░░░ 0.52
Correlation_SPX       ██████░░░ 1.75
...
```

**Interpretation**: RSI and correlation are the strongest predictors

**Visualization 2: Feature Distribution Overlays**
Shows histograms of each feature:
- Blue = All timestamps (baseline)
- Red = Formation timestamps only

If formations cluster in one part of the distribution, it's a useful predictor.

---

### Cells 8-9: Build Predictive Dataset

**What it does**: Creates a machine learning dataset where we try to predict formations

**Dataset structure**:

For each (pair, timestamp) combination:
```python
Features: Average feature values of both stocks at timestamp T-1
          [RSI_avg, Volatility_avg, Momentum_avg, ...]

Label: Did this pair form a cluster at timestamp T?
       1 = Yes (formation event)
       0 = No (no formation)
```

**Example**:

```
Pair       Timestamp  RSI_avg  Volatility_avg  Momentum_avg  Label
NVDA-TSMC  2024-01-10  73.5     0.032           0.15         1 (formed)
NVDA-TSMC  2024-01-11  52.3     0.025           0.08         0 (didn't form)
NVDA-TSMC  2024-01-12  61.1     0.028           0.11         0 (didn't form)
NVDA-AMD   2024-01-10  68.2     0.029           0.12         1 (formed)
...
```

**Class imbalance note**:
- Formation events (label=1) are rare (~2-5% of all timestamps)
- Most timestamps are non-formation (label=0)
- ML model must handle this imbalance (typically via class_weight parameter)

---

### Cells 10: Train Predictive Model with Time-Series Cross-Validation

**What it does**: Trains machine learning models to predict formations

#### Why Time-Series Cross-Validation?

Regular cross-validation (random split) would cause **look-ahead bias**:
```
BAD (look-ahead bias):
Randomly split data:
  Train: Timestamp 100, 50, 200, 150
  Test:  Timestamp 25, 75, 125, 175
→ Model sees future data during training

GOOD (time-series split):
Split chronologically:
  Fold 1: Train=[1-50], Test=[51-75]
  Fold 2: Train=[1-100], Test=[101-125]
  Fold 3: Train=[1-150], Test=[151-175]
→ Model always predicts the future
```

**Time-Series Split Process**:
```
Dataset: Timestamps 1 to 500

Fold 1:
  Train: 1-100
  Test:  101-125

Fold 2:
  Train: 1-200
  Test:  201-225

Fold 3:
  Train: 1-300
  Test:  301-325

Fold 4:
  Train: 1-400
  Test:  401-425
```

#### Models Trained

**1. Random Forest Classifier**

Ensemble of decision trees that vote on the prediction.

```
How it works:
- Tree 1: If RSI > 70 and Volatility > 0.03, then formation likely
- Tree 2: If Momentum > 0.10 and Correlation > 0.7, then formation likely
- Tree 3: If MACD > 0 and Price_trend up, then formation likely
- ...
- Vote: Formation if majority of trees say yes
```

**Pros**:
- Handles non-linear relationships
- Identifies feature importance automatically
- Fast to train

**Cons**:
- Can overfit on historical data
- Less interpretable than linear models

**2. Gradient Boosting Classifier**

Sequentially builds trees, each one correcting errors of previous trees.

```
Tree 1: Predicts with 60% accuracy
       Error cases: cases where it was wrong

Tree 2: Focuses on Tree 1's error cases
       Improves accuracy to 70%

Tree 3: Focuses on Tree 2's error cases
       Improves accuracy to 78%

...etc
```

**Pros**:
- Typically better accuracy than Random Forest
- Still captures non-linear patterns
- Weighted on hardest cases

**Cons**:
- Slower to train
- More hyperparameters to tune
- Higher risk of overfitting

#### Output

For each model, compute:

```python
predictions = model.predict(X_test)      # 0 or 1
probabilities = model.predict_proba(X_test)  # [0.3, 0.7] for each sample

# Evaluation metrics
accuracy = (predictions == y_test).mean()
auc_score = roc_auc_score(y_test, probabilities[:, 1])
precision, recall = precision_recall_curve(y_test, probabilities[:, 1])
```

**Interpretation**:
- **Accuracy**: % of correct predictions (but misleading for imbalanced data)
- **AUC**: Area Under ROC curve (0.5=random, 1.0=perfect; typical target >0.70)
- **Precision**: Of predicted formations, how many were correct?
- **Recall**: Of actual formations, how many did we catch?

---

### Cell 11: Visualize Prediction Results

**What it does**: Creates plots showing model performance and feature importance

#### Visualization 1: Feature Importance

```
Feature               Importance
RSI                   ████████████ 0.18
Volatility            ██████████░░ 0.15
Momentum              █████████░░░ 0.13
Correlation_SPX       █████████░░░ 0.13
MACD                  ███████░░░░░ 0.10
...
```

**Interpretation**:
- RSI is most important for predicting formations
- Top 5 features drive 70% of the model's decisions
- Bottom features contribute little

**How it's calculated** (for Random Forest):
```
Feature importance = How much each feature reduces impurity across all trees
                   = Weighted by how many samples use that feature
```

#### Visualization 2: ROC Curve

```
True Positive Rate (y-axis)
|        /
|       /
|      / ← Model prediction curve
|     /
|    / ← Random baseline (diagonal)
|   /
|  /
└─────────────────
False Positive Rate (x-axis)
```

**Interpretation**:
- Curve high above diagonal = Good model
- Curve near diagonal = Model no better than random
- Area under curve (AUC) = 0.85 typically means "good"

#### Visualization 3: Precision-Recall Curve

```
Precision (y-axis) - "Of predicted formations, how many were real?"
|
| ███████████
| █         █
| █         █
| █         █
| █         █  ← High precision, low recall
| █         █
| █         █
└────────────
   Recall - "Of real formations, how many did we catch?"
```

**Trade-off**:
- High precision, low recall = Conservative (only predict when very sure)
- Low precision, high recall = Aggressive (catch all formations, but false alarms)

**For trading**: Typically want high precision (avoid false alarms) even if recall is lower

---

## Summary: How the Notebooks Work Together

### **optics-clustering-run.ipynb** (The Detective):
1. Fetches price and feature data
2. Runs OPTICS to find clusters
3. Detects when pairs START clustering (formation events)
4. Validates that these relationships are tradeable
5. **Saves results for the predictor**

### **optics-prediction.ipynb** (The Forecaster):
1. Loads the detected formations
2. **Asks**: What features are unusual at formation times? (Factor Attribution)
3. **Asks**: Can we predict formations using prior features? (Predictive Modeling)
4. Trains ML models to forecast future formations
5. Evaluates whether formations are predictable

---

## Key Insight: The Full Loop

```
Past ← Feature values at T-1 (RSI, Volatility, etc.)
       ↓
       Can we use these to predict formations at T?
       ↓
       YES → Unusual feature patterns precede formations
            → We can build a model to predict them
            → Use model to trigger trade alerts

       NO  → Formations are essentially random
            → No predictive edge (need different approach)
```

---

## Practical Use Cases

### Use Case 1: **Identify Trading Opportunities**
1. Run `optics-clustering-run.ipynb` on recent market data
2. Get list of pairs from `df_validation` (highest persistence and bounce rate)
3. Execute mean-reversion trades on the spread

### Use Case 2: **Predict Upcoming Formations**
1. Train predictor on 6 months of historical data
2. Use `optics-prediction.ipynb`'s model on current features
3. Pairs with high formation probability get added to watch list
4. When actual formation occurs (detected via OPTICS on live data), execute trade

### Use Case 3: **Factor Research**
1. Use factor attribution analysis from `optics-prediction.ipynb`
2. Identify which features consistently precede formations (e.g., "RSI divergences")
3. Build factor-based trading rules

---

## Common Errors & Solutions

### Error: "KeyError: ts_df not defined"
**Solution**: Run the feature engineering cell (Cell 5) in optics-clustering-run before clustering

### Error: "Empty cluster_history"
**Solution**: OPTICS found no clusters. Try loosening xi parameter (increase from 0.05 to 0.10)

### Error: "No formations detected"
**Solution**: May require more data, different time frame, or different assets

### Error: "prediction fails - missing pickle files"
**Solution**: Run optics-clustering-run.ipynb completely before running optics-prediction.ipynb

---

## Performance Expectations

**Typical results** (varies by market conditions):

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| Clusters detected per day | 2-5 | Normal activity |
| Pairs clustering daily | 5-20 | Varies by market regime |
| Tradeable pairs | 10-30% | After validation filters |
| Cluster persistence | 60-85% | Moderate to strong relationships |
| Formation prediction AUC | 0.60-0.75 | Modest predictive power |
| Bounce rate | 50-70% | Decent mean reversion |

---

## Next Steps for Enhancement

1. **Real-time prediction**: Deploy model to score live features every 5 minutes
2. **Risk management**: Add stop-loss logic based on bounce rate
3. **Multi-timeframe**: Run on hourly + daily data for confluence signals
4. **More factors**: Add volume, order flow, options implied vol
5. **Pair ranking**: Rank by expected edge (formation probability × bounce rate × persistence)

