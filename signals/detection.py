"""
Cluster formation and dissolution detection.

Extracted from optics-clustering.ipynb (formation/dissolution event detection)
and optics-signals.ipynb (detect_new_formations).
"""

import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS

from config import ClusteringConfig, FeatureConfig, FormationConfig, DEFAULT_CONFIG


def run_clustering_snapshot(snapshot_features, cfg: ClusteringConfig = None):
    """
    Run OPTICS clustering on a single timestamp snapshot.

    Parameters
    ----------
    snapshot_features : DataFrame
        Rows are tickers, columns are features. Must have >= min_stocks_per_snapshot rows.
    cfg : ClusteringConfig

    Returns
    -------
    labels : ndarray of cluster IDs (-1 = noise), or None if invalid.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.clustering

    if len(snapshot_features) < cfg.min_stocks_per_snapshot:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(snapshot_features.values)

    pca = PCA(n_components=cfg.pca_variance_threshold)
    X_pca = pca.fit_transform(X_scaled)

    optics = OPTICS(
        min_samples=cfg.min_samples,
        metric=cfg.metric,
        xi=cfg.xi,
        min_cluster_size=cfg.min_cluster_size,
    )
    optics.fit(X_pca)

    noise_pct = (optics.labels_ == -1).sum() / len(optics.labels_)
    n_clusters = len(set(optics.labels_)) - (1 if -1 in optics.labels_ else 0)

    if n_clusters < 1 or noise_pct > cfg.max_noise_pct:
        return None

    return optics.labels_


def build_cluster_history(ts_df, cfg: ClusteringConfig = None, feat_cfg: FeatureConfig = None):
    """
    Run OPTICS clustering across all timestamps in ts_df.

    Parameters
    ----------
    ts_df : DataFrame
        MultiIndex (Datetime, Ticker) with feature columns.
    cfg : ClusteringConfig
    feat_cfg : FeatureConfig
        Specifies which columns to use for clustering.

    Returns
    -------
    cluster_history : DataFrame
        Columns: Ticker, Datetime, Cluster_ID.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.clustering
    if feat_cfg is None:
        feat_cfg = DEFAULT_CONFIG.features

    features = list(feat_cfg.features_to_cluster)
    timestamps = ts_df.index.get_level_values('Datetime').unique()

    results = []
    for ts in timestamps:
        try:
            snapshot = ts_df.xs(ts, level='Datetime')[features].dropna()
        except KeyError:
            continue

        labels = run_clustering_snapshot(snapshot, cfg)
        if labels is None:
            continue

        tickers = snapshot.index.tolist()
        for ticker, label in zip(tickers, labels):
            results.append({
                'Ticker': ticker,
                'Datetime': ts,
                'Cluster_ID': int(label),
            })

    return pd.DataFrame(results)


def detect_new_formations(cluster_history_latest, previous_clusters):
    """
    Compare latest clustering snapshot to previous snapshot.
    Identify new co-clustering events (pair transitions from
    not-co-clustered to co-clustered).

    Parameters
    ----------
    cluster_history_latest : DataFrame
        Latest snapshot with columns ['Ticker', 'Cluster_ID'].
    previous_clusters : dict
        Mapping ticker -> cluster_id from the previous timestamp.

    Returns
    -------
    list of tuple (ticker_a, ticker_b) representing new formations.
    """
    current_clusters = {}
    for _, row in cluster_history_latest.iterrows():
        if row['Cluster_ID'] != -1:
            current_clusters[row['Ticker']] = row['Cluster_ID']

    current_pairs = set()
    for cid in set(current_clusters.values()):
        members = sorted([t for t, c in current_clusters.items() if c == cid])
        for a, b in itertools.combinations(members, 2):
            current_pairs.add((a, b))

    prev_pairs = set()
    prev_non_noise = {t: c for t, c in previous_clusters.items() if c != -1}
    for cid in set(prev_non_noise.values()):
        members = sorted([t for t, c in prev_non_noise.items() if c == cid])
        for a, b in itertools.combinations(members, 2):
            prev_pairs.add((a, b))

    return list(current_pairs - prev_pairs)


def compute_co_cluster_freq(cluster_history):
    """
    Count how often each pair co-clusters across all timestamps.

    Returns
    -------
    pair_freq : dict mapping (ticker_a, ticker_b) -> count
    total_windows : int
    """
    pair_freq = {}
    for ts in cluster_history['Datetime'].unique():
        snapshot = cluster_history[cluster_history['Datetime'] == ts]
        for cid in snapshot['Cluster_ID'].unique():
            if cid == -1:
                continue
            members = sorted(snapshot[snapshot['Cluster_ID'] == cid]['Ticker'].tolist())
            for s1, s2 in itertools.combinations(members, 2):
                pair_freq[(s1, s2)] = pair_freq.get((s1, s2), 0) + 1

    total_windows = cluster_history['Datetime'].nunique()
    return pair_freq, total_windows


def detect_formation_events(cluster_history, pair_co_cluster_freq,
                            cfg: FormationConfig = None):
    """
    Detect formation and dissolution events from cluster history.

    A formation event occurs when a pair transitions from not-co-clustered
    to co-clustered after a minimum gap.

    Parameters
    ----------
    cluster_history : DataFrame
        Columns: Ticker, Datetime, Cluster_ID.
    pair_co_cluster_freq : dict
        Mapping (ticker_a, ticker_b) -> co-cluster count.
    cfg : FormationConfig

    Returns
    -------
    df_formations : DataFrame with Formation_Time, Pair, Ticker_1, Ticker_2.
    df_durations : DataFrame with formation/dissolution times and durations.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.formation

    all_pairs = list(pair_co_cluster_freq.keys())
    all_timestamps = sorted(cluster_history['Datetime'].unique())

    # Build co-clustering indicator per pair per timestamp
    pair_coclustering = {}
    for ts in all_timestamps:
        snapshot = cluster_history[cluster_history['Datetime'] == ts]
        coclustered = set()
        for cid in snapshot['Cluster_ID'].unique():
            if cid == -1:
                continue
            members = sorted(snapshot[snapshot['Cluster_ID'] == cid]['Ticker'].tolist())
            for s1, s2 in itertools.combinations(members, 2):
                coclustered.add((s1, s2))

        for pair in all_pairs:
            if pair not in pair_coclustering:
                pair_coclustering[pair] = []
            pair_coclustering[pair].append(1 if pair in coclustered else 0)

    formations = []
    durations = []

    for pair, series in pair_coclustering.items():
        ts_series = list(zip(all_timestamps, series))
        in_cluster = False
        formation_ts = None
        formation_idx = None
        gap_count = cfg.min_gap_hours

        for i, (ts, val) in enumerate(ts_series):
            if val == 1:
                if not in_cluster and gap_count >= cfg.min_gap_hours:
                    in_cluster = True
                    formation_ts = ts
                    formation_idx = i
                    formations.append({
                        'Ticker_1': pair[0],
                        'Ticker_2': pair[1],
                        'Pair': f"{pair[0]}-{pair[1]}",
                        'Formation_Time': ts,
                    })
                elif not in_cluster:
                    # Re-entered cluster before min_gap_hours elapsed —
                    # treat as a new formation so every duration record
                    # has a corresponding formation entry.
                    in_cluster = True
                    formation_ts = ts
                    formation_idx = i
                    formations.append({
                        'Ticker_1': pair[0],
                        'Ticker_2': pair[1],
                        'Pair': f"{pair[0]}-{pair[1]}",
                        'Formation_Time': ts,
                    })
                gap_count = 0
            else:
                if in_cluster:
                    gap_count += 1
                    if gap_count >= cfg.min_gap_hours:
                        last_idx = i - gap_count
                        dissolution_ts = all_timestamps[min(last_idx + 1, len(all_timestamps) - 1)]
                        dur = max(1.0, (all_timestamps[last_idx] - all_timestamps[formation_idx]).total_seconds() / 3600.0) if formation_idx is not None else 1.0

                        if formation_ts is not None:
                            durations.append({
                                'Ticker_1': pair[0],
                                'Ticker_2': pair[1],
                                'Pair': f"{pair[0]}-{pair[1]}",
                                'Formation_Time': formation_ts,
                                'Dissolution_Time': dissolution_ts,
                                'Duration_Hours': dur,
                            })

                        in_cluster = False
                        formation_ts = None
                        formation_idx = None
                else:
                    gap_count += 1

        # Handle pairs still in cluster at end
        if in_cluster and formation_ts is not None:
            last_idx = len(ts_series) - 1
            for j in range(len(ts_series) - 1, -1, -1):
                if ts_series[j][1] == 1:
                    last_idx = j
                    break
            dur = max(1.0, (all_timestamps[last_idx] - all_timestamps[formation_idx]).total_seconds() / 3600.0) if formation_idx is not None else 1.0
            dissolution_ts = all_timestamps[min(last_idx + 1, len(all_timestamps) - 1)]
            durations.append({
                'Ticker_1': pair[0],
                'Ticker_2': pair[1],
                'Pair': f"{pair[0]}-{pair[1]}",
                'Formation_Time': formation_ts,
                'Dissolution_Time': dissolution_ts,
                'Duration_Hours': dur,
            })

    return pd.DataFrame(formations), pd.DataFrame(durations)
