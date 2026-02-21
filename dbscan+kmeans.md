# Implementation Plan: DBSCAN & KMeans Clustering Comparison

## Purpose

Run the exact same pipeline currently implemented with OPTICS using DBSCAN and KMeans to answer:

1. Do they find the same top pairs?
2. Do any algorithms find tradeable pairs that OPTICS missed?
3. How do noise rates, cluster counts, and episode durations compare?
4. Which algorithm is best suited for transient relationship detection?

The comparison must be apples-to-apples: same data, same features, same preprocessing, same validation, same metrics.

---

## What Already Exists (Don't Rebuild)

### Reuse Directly

| Component | Location | Notes |
|-----------|----------|-------|
| Feature engineering | optics-clustering.ipynb cells 1-3 | Produces `ts_df` with 13 features, 63,021 rows. No changes needed. |
| `ts_df.pkl` | research/data/ | Already saved. DBSCAN/KMeans notebooks can load this directly instead of re-fetching data and recomputing features. |
| Validation module | validation/pair_validation.py | All functions are algorithm-agnostic. `compute_hedge_ratio()`, `half_life()`, `zscore_signals()`, `simulate_spread_pnl()`, `feature_shuffle_permutation_test()` all work on any cluster_history DataFrame. |
| Formation/dissolution detection | optics-clustering.ipynb Phase 2B | The event detection logic operates on `pair_coclustering` dict. Same code works regardless of which algorithm produced the cluster labels. |
| Transient validation | optics-signals.ipynb | `validate_transient_event()` and baseline comparison are algorithm-agnostic. |
| Permutation test | `feature_shuffle_permutation_test()` in pair_validation.py | Takes `optics_params` dict -- needs to be generalized to accept any clustering callable. See below. |

### Must Modify

| Component | What to Change |
|-----------|---------------|
| Clustering loop | Replace OPTICS with DBSCAN or KMeans (the StandardScaler -> PCA -> cluster step) |
| Permutation test | Currently hardcoded to use OPTICS. Needs a `clustering_fn` parameter instead. |
| Quality filters | OPTICS-specific noise thresholds may not apply to KMeans (which has no noise concept) |

---

## Architecture: Two New Notebooks + One Comparison Notebook

```
research/
├── optics-clustering.ipynb          # Existing (don't touch)
├── optics-signals.ipynb             # Existing (don't touch)
├── dbscan-clustering.ipynb          # NEW: DBSCAN pipeline
├── kmeans-clustering.ipynb          # NEW: KMeans pipeline
├── algorithm-comparison.ipynb       # NEW: side-by-side analysis
└── data/
    ├── ts_df.pkl                    # Shared (loaded by all notebooks)
    ├── dbscan_cluster_history.pkl   # NEW
    ├── dbscan_formations.pkl        # NEW
    ├── dbscan_pair_stability.pkl    # NEW
    ├── ... (other dbscan artifacts)
    ├── kmeans_cluster_history.pkl   # NEW
    ├── kmeans_formations.pkl        # NEW
    ├── kmeans_pair_stability.pkl    # NEW
    ├── ... (other kmeans artifacts)
    └── ... (existing optics artifacts)
```

Each clustering notebook saves artifacts with a prefix (`dbscan_`, `kmeans_`). The comparison notebook loads all three sets.

---

## DBSCAN Notebook: `dbscan-clustering.ipynb`

### Cell 1: Load Shared Data

```python
import pickle, os
data_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'data')
with open(os.path.join(data_dir, 'ts_df.pkl'), 'rb') as f:
    ts_df = pickle.load(f)
```

No data fetching, no feature engineering. Just load `ts_df.pkl`.

### Cell 2: DBSCAN Clustering Loop

Same structure as optics-clustering.ipynb Phase 1, but replace:

```python
# OPTICS (current)
optics = OPTICS(min_samples=3, metric='euclidean', xi=0.05, min_cluster_size=3)
optics.fit(X_pca)
labels = optics.labels_

# DBSCAN (replacement)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=eps_value, min_samples=3, metric='euclidean')
dbscan.fit(X_pca)
labels = dbscan.labels_
```

#### The `eps` Problem

DBSCAN's critical parameter is `eps` (neighborhood radius). Unlike OPTICS which auto-adapts density thresholds, DBSCAN needs a fixed `eps` that works across all timestamps. The feature space changes at each timestamp (different PCA projections, different scales after StandardScaler), so a single global `eps` will be too loose at some timestamps and too tight at others.

**Solution: Adaptive eps per timestamp using k-distance heuristic.**

For each timestamp snapshot:
1. After StandardScaler -> PCA, compute the k-nearest-neighbor distance for each point (k = min_samples - 1 = 2)
2. Sort these distances
3. Find the "elbow" -- the point where the distance curve bends sharply
4. Use this as `eps` for that timestamp

Implementation:
```python
from sklearn.neighbors import NearestNeighbors

def adaptive_eps(X_pca, k=2, quantile=0.90):
    """
    Estimate DBSCAN eps from the k-distance graph.
    Uses the 90th percentile of k-distances as a robust elbow estimate.
    """
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X_pca)
    distances, _ = nn.kneighbors(X_pca)
    k_distances = np.sort(distances[:, k])
    return float(np.percentile(k_distances, quantile * 100))
```

The `quantile` parameter controls how aggressive DBSCAN is. Higher quantile = larger eps = fewer noise points = larger clusters. Test with quantile values of 0.80, 0.85, 0.90, 0.95.

**Alternative approach**: Run DBSCAN with multiple fixed eps values (0.5, 1.0, 1.5, 2.0, 2.5) across all timestamps and report results for each. This is simpler and gives a clear sensitivity picture.

#### Quality Filters

DBSCAN has a noise concept (label = -1), same as OPTICS. Use the same quality filters:
- Skip if no clusters found (all noise)
- Skip if noise > 75%
- Skip if over-fragmented

#### Output

Same structure as OPTICS: `cluster_history` DataFrame with columns `[Datetime, Ticker, Cluster_ID, Num_Clusters, Noise_Pct, ...]`

### Cell 3: Pair Stability Analysis

Exact same code as optics-clustering.ipynb Phase 2. Iterates through cluster_history, counts co-clustering per pair, computes frequencies. No algorithm-specific logic.

Output: `dbscan_pair_stability`, `dbscan_pair_co_cluster_freq`

### Cell 4: Formation/Dissolution Detection

Exact same code as optics-clustering.ipynb Phase 2B. Same `MIN_GAP_HOURS=5`, same episode detection logic, same actionable filtering (`MIN_EPISODE_HOURS=5`).

Output: `dbscan_formations`, `dbscan_formations_actionable`, `dbscan_durations`, `dbscan_pair_classification`

### Cell 5: Correctness Checks

1. **Permutation test**: Same feature-shuffling approach, but use DBSCAN instead of OPTICS in the permutation loop. This requires modifying `feature_shuffle_permutation_test()` in pair_validation.py to accept a generic clustering callable (see Validation Module Changes section below).

2. **OOS split**: Same 67/33 split, same correlation calculation.

3. **Parameter sensitivity**: Test 4-5 eps configurations (or 4-5 quantile values if using adaptive eps) with fixed min_samples=3. Report avg_clusters, avg_noise, std_clusters for each.

### Cell 6: Save Artifacts

Save all outputs with `dbscan_` prefix to `research/data/`.

---

## KMeans Notebook: `kmeans-clustering.ipynb`

### Cell 1: Load Shared Data

Same as DBSCAN notebook.

### Cell 2: KMeans Clustering Loop

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_pca)
labels = kmeans.labels_
```

#### The `k` Problem

KMeans requires specifying the number of clusters upfront. This is a fundamental difference from OPTICS/DBSCAN which discover the number automatically. Key considerations:

1. **KMeans has no noise concept.** Every ticker gets assigned to a cluster, even if it doesn't belong to any group. This is a significant difference -- OPTICS/DBSCAN's noise label (-1) means "this ticker isn't similar to anything right now," which is a valuable signal. KMeans forces assignment, which will create spurious co-clustering between dissimilar tickers.

2. **Fixed k across all timestamps** is problematic because the number of natural clusters varies (OPTICS found 1-7 clusters per timestamp with a mean of 3.3).

**Solution A: Fixed k with silhouette-based selection.**

For each timestamp:
1. Run KMeans with k = 2, 3, 4, 5, 6
2. Compute silhouette score for each k
3. Select the k with the highest silhouette score
4. Use those labels

This is the most principled approach but is slower (5 fits per timestamp instead of 1).

**Solution B: Fixed k = 4 (the OPTICS median rounded).**

Use k=4 for all timestamps (OPTICS averaged 3.3 clusters). Simpler, faster, but ignores timestamps where the true structure has 2 or 6 clusters.

**Solution C: Create a synthetic noise label.**

After KMeans with optimal k, compute each point's distance to its cluster centroid. If the distance exceeds a threshold (e.g., 90th percentile of all distances in that timestamp), relabel that point as -1 (noise). This makes KMeans results more comparable to OPTICS/DBSCAN by filtering out forced assignments.

**Recommended: Use Solution A (silhouette-based k selection) with Solution C (distance-based noise filtering).** This gives the fairest comparison to OPTICS.

Implementation:
```python
from sklearn.metrics import silhouette_score

def kmeans_adaptive(X_pca, k_range=range(2, 7), noise_percentile=90, random_state=42):
    """
    Run KMeans with silhouette-based k selection and distance-based noise filtering.
    """
    best_k, best_score, best_labels, best_model = 2, -1, None, None

    for k in k_range:
        if k >= len(X_pca):
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_pca)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_pca, labels)
        if score > best_score:
            best_k, best_score, best_labels, best_model = k, score, labels, km

    if best_labels is None:
        return np.full(len(X_pca), -1)

    # Distance-based noise filtering
    distances = best_model.transform(X_pca).min(axis=1)
    threshold = np.percentile(distances, noise_percentile)
    noise_mask = distances > threshold
    filtered_labels = best_labels.copy()
    filtered_labels[noise_mask] = -1

    return filtered_labels
```

#### Quality Filters

After noise filtering, apply the same quality filters as OPTICS/DBSCAN:
- Skip if all points are noise (unlikely with KMeans but possible after filtering)
- Skip if noise > 75%
- Skip if over-fragmented

### Cell 3-6: Same as DBSCAN

Pair stability, formation/dissolution, correctness checks, save artifacts. All algorithm-agnostic code.

#### Parameter Sensitivity for KMeans

Test configurations:
- k_range variations: [2-5], [2-6], [2-7], [3-6]
- noise_percentile variations: 85, 90, 95 (or no noise filtering for one config)
- With and without silhouette selection (fixed k=3 vs fixed k=4 vs adaptive)

---

## Validation Module Changes

### Generalize `feature_shuffle_permutation_test()`

Current signature:
```python
def feature_shuffle_permutation_test(
    ts_df, features_to_cluster, optics_params,
    pair_co_cluster_freq, total_valid_windows,
    n_permutations=30, n_sample_timestamps=80,
)
```

The function currently hardcodes `OPTICS(**optics_params)`. Change to accept a callable:

```python
def feature_shuffle_permutation_test(
    ts_df, features_to_cluster, clustering_fn,
    pair_co_cluster_freq, total_valid_windows,
    n_permutations=30, n_sample_timestamps=80,
)
```

Where `clustering_fn` is a function that takes `X_pca` (numpy array) and returns `labels` (numpy array of cluster IDs, -1 for noise).

Each notebook would call it like:

```python
# OPTICS
def optics_cluster(X):
    model = OPTICS(min_samples=3, xi=0.05, min_cluster_size=3)
    model.fit(X)
    return model.labels_

# DBSCAN
def dbscan_cluster(X):
    eps = adaptive_eps(X)
    model = DBSCAN(eps=eps, min_samples=3)
    model.fit(X)
    return model.labels_

# KMeans
def kmeans_cluster(X):
    return kmeans_adaptive(X)

perm_result = feature_shuffle_permutation_test(
    ts_df=ts_df,
    features_to_cluster=features_to_cluster,
    clustering_fn=optics_cluster,  # or dbscan_cluster or kmeans_cluster
    pair_co_cluster_freq=pair_co_cluster_freq,
    total_valid_windows=total_valid_windows,
)
```

This is a backward-compatible change if we keep the old `optics_params` as an optional argument with a deprecation path, but since the notebooks control the call sites, it's cleaner to just update the signature.

---

## Comparison Notebook: `algorithm-comparison.ipynb`

### Cell 1: Load All Artifacts

Load the three sets of artifacts (OPTICS, DBSCAN, KMeans) from `research/data/`.

### Cell 2: Clustering Quality Comparison

Side-by-side table:

| Metric | OPTICS | DBSCAN | KMeans |
|--------|--------|--------|--------|
| Valid timestamps | | | |
| Avg clusters/timestamp | | | |
| Avg noise rate | | | |
| Std clusters | | | |
| Total unique pairs | | | |
| Pairs >30% frequency | | | |
| Pairs >10% frequency | | | |

**Visualization**: Box plots of clusters-per-timestamp and noise-rate-per-timestamp for each algorithm.

### Cell 3: Top Pair Overlap Analysis

**Key question**: Do the three algorithms find the same top pairs?

1. Get top 20 pairs by co-clustering frequency from each algorithm
2. Compute Jaccard similarity between the three top-20 sets
3. Show which pairs appear in all three, which in exactly two, which are unique to one algorithm

If all three algorithms find AMAT-LRCX, QRVO-SWKS, KLAC-LRCX as top pairs, that's strong evidence these are real relationships regardless of algorithm choice.

**Visualization**: Venn diagram (or UpSet plot) of top-20 pair overlap.

### Cell 4: Co-Clustering Frequency Correlation

For pairs found by multiple algorithms, correlate their co-clustering frequencies.

```python
# Merge pair stability tables
merged = optics_stability.merge(dbscan_stability, on='Pair', suffixes=('_optics', '_dbscan'))
corr = merged['Co_Cluster_Frequency_optics'].corr(merged['Co_Cluster_Frequency_dbscan'])
```

High correlation (>0.8) = algorithms agree on relative pair strength.
Low correlation (<0.5) = algorithms are capturing different structure.

**Visualization**: Scatter plot of OPTICS frequency vs DBSCAN frequency, colored by pair subsector.

### Cell 5: Permutation Test Comparison

Compare Z-scores across algorithms for the same pairs.

| Pair | OPTICS Z | DBSCAN Z | KMeans Z |
|------|----------|----------|----------|
| AMAT-LRCX | 3.49 | ? | ? |
| QRVO-SWKS | 3.45 | ? | ? |
| ... | ... | ... | ... |

**Key question**: Do pairs that are significant under OPTICS also reach significance under DBSCAN/KMeans?

### Cell 6: Formation Event Comparison

| Metric | OPTICS | DBSCAN | KMeans |
|--------|--------|--------|--------|
| Total formations | 2,069 | ? | ? |
| Actionable (>=5h) | 1,665 | ? | ? |
| Mean duration | 15.8h | ? | ? |
| Median duration | 10.0h | ? | ? |

**Key question**: Does KMeans (which forces all tickers into clusters) produce many more formation events with shorter durations? Does DBSCAN produce fewer events with longer durations?

**Visualization**: Duration distribution histograms overlaid for all three algorithms.

### Cell 7: Transient Validation Pass Rates

Run `validate_transient_event()` on OOS formation events from each algorithm using the same `TRANSIENT_CONFIG`. Compare:

| Metric | OPTICS | DBSCAN | KMeans |
|--------|--------|--------|--------|
| Events tested | | | |
| Pass rate | | | |
| Avg P&L (passed) | | | |
| Avg correlation | | | |
| Avg spread CV | | | |
| Avg half-life | | | |

Also run the random baseline for each algorithm to compute the lift:

| Algorithm | Clustered pass rate | Random pass rate | Lift |
|-----------|-------------------|-----------------|------|
| OPTICS | | | |
| DBSCAN | | | |
| KMeans | | | |

**This is the most important comparison.** The algorithm with the highest lift (clustered / random) is best at selecting pairs that have genuine tradeable properties.

### Cell 8: OOS Stability Comparison

| Algorithm | OOS correlation (r) | p-value |
|-----------|-------------------|---------|
| OPTICS | 0.723 | <0.0001 |
| DBSCAN | ? | ? |
| KMeans | ? | ? |

Higher OOS correlation = more stable, reproducible clustering.

### Cell 9: Unique Discoveries

**Key question**: Do DBSCAN or KMeans find tradeable pairs that OPTICS missed?

1. Get pairs that pass transient validation from each algorithm
2. Find pairs that pass under DBSCAN/KMeans but NOT under OPTICS
3. Investigate these pairs: are they real relationships that OPTICS missed due to density assumptions, or are they spurious?

### Cell 10: Summary & Recommendation

Final table with all metrics, and a narrative conclusion about which algorithm is best for:
- Transient pair detection (the primary use case)
- Long-term pair discovery (secondary use case)
- Real-time scanning (practical deployment)

---

## Expected Differences Between Algorithms

### OPTICS vs DBSCAN

OPTICS and DBSCAN are both density-based, so they should find similar structure. The key difference:

- **OPTICS** handles variable-density clusters (can find a tight cluster of 3 equipment stocks and a looser cluster of 6 analog stocks in the same snapshot)
- **DBSCAN** with a fixed eps will either miss the tight cluster (eps too large) or fragment the loose cluster (eps too small)
- **Adaptive eps** partially solves this, but it's still one threshold per snapshot

**Expected**: DBSCAN will find similar top pairs but with lower co-clustering frequencies (more timestamps where the fixed/adaptive eps doesn't match the natural density). Noise rates may be higher or lower depending on eps selection.

### OPTICS/DBSCAN vs KMeans

Fundamentally different algorithms:

- **KMeans forces every point into a cluster**. This creates co-clustering between dissimilar tickers that OPTICS/DBSCAN would leave as noise. Expect more unique pairs, higher co-clustering frequencies, but many of them spurious.
- **KMeans assumes spherical clusters**. Feature space structure may not be spherical after PCA.
- **KMeans is sensitive to initialization**. Use n_init=10 to mitigate.

**Expected**: KMeans will find more pairs (no noise filtering means everything co-clusters sometimes), but the transient validation pass rate will be lower (more false positives). The lift over random baseline will likely be lower than OPTICS/DBSCAN. The noise-filtering step (Solution C above) is critical to making KMeans results comparable.

### Bottom Line Prediction

OPTICS will likely remain the best algorithm for this use case because:
1. Variable-density handling matches the semiconductor subsector structure (tight equipment cluster, looser analog cluster)
2. Native noise detection avoids forced co-clustering
3. The xi parameter gives fine-grained control over cluster extraction

But DBSCAN may be competitive with adaptive eps, and KMeans may surface interesting pairs through its different inductive bias (centroid-based rather than density-based). The comparison is worth running even if OPTICS "wins" -- confirming the algorithm choice is itself a valuable finding.

---

## Implementation Checklist

### Validation Module Update
- [ ] Refactor `feature_shuffle_permutation_test()` to accept `clustering_fn` callable instead of `optics_params`
- [ ] Update optics-clustering.ipynb to use new signature (wrap OPTICS in a lambda/function)
- [ ] Test that OPTICS results are unchanged after refactor

### DBSCAN Notebook
- [ ] Cell 1: Load `ts_df.pkl`
- [ ] Implement `adaptive_eps()` function
- [ ] Cell 2: DBSCAN clustering loop (same timestamps as OPTICS)
- [ ] Cell 3: Pair stability analysis (copy from OPTICS)
- [ ] Cell 4: Formation/dissolution detection (copy from OPTICS)
- [ ] Cell 5: Correctness checks (permutation test with DBSCAN, OOS split, eps sensitivity)
- [ ] Cell 6: Save all artifacts with `dbscan_` prefix

### KMeans Notebook
- [ ] Cell 1: Load `ts_df.pkl`
- [ ] Implement `kmeans_adaptive()` function (silhouette k selection + distance noise filter)
- [ ] Cell 2: KMeans clustering loop
- [ ] Cell 3: Pair stability analysis (copy from OPTICS)
- [ ] Cell 4: Formation/dissolution detection (copy from OPTICS)
- [ ] Cell 5: Correctness checks (permutation test with KMeans, OOS split, k/noise sensitivity)
- [ ] Cell 6: Save all artifacts with `kmeans_` prefix

### Comparison Notebook
- [ ] Cell 1: Load all three artifact sets
- [ ] Cell 2: Clustering quality comparison table
- [ ] Cell 3: Top pair overlap (Jaccard, Venn)
- [ ] Cell 4: Co-clustering frequency correlation
- [ ] Cell 5: Permutation test Z-score comparison
- [ ] Cell 6: Formation event statistics comparison
- [ ] Cell 7: Transient validation pass rate comparison (the key metric)
- [ ] Cell 8: OOS stability comparison
- [ ] Cell 9: Unique discoveries analysis
- [ ] Cell 10: Summary and recommendation
