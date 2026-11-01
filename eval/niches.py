"""Niche evaluation: cluster spot embeddings and score them against labels.

Owner: Deeptha S (P1). Mayur (P4) runs this for every method in E3-E6,
so baselines and our model are scored by exactly the same code.

Labels are only used here, for scoring. Spots with a missing label
(NaN, None, "" or "NA") are left out of the score but still clustered.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

MISSING = {"", "nan", "none", "na"}


def cluster(Z, k, seed=0, n_init=10, standardize=True):
    """K-means on embeddings Z (N, D) -> integer niche labels (N,)."""
    Z = np.asarray(Z, dtype=float)
    if standardize:
        Z = StandardScaler().fit_transform(Z)
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(Z)


def _valid(labels):
    lab = np.asarray(labels, dtype=object)
    bad = np.array([v is None or (isinstance(v, float) and np.isnan(v))
                    or str(v).strip().lower() in MISSING for v in lab])
    return ~bad


def score(labels_true, labels_pred):
    """ARI and NMI on spots that have a label. Returns a dict."""
    keep = _valid(labels_true)
    t = np.asarray(labels_true, dtype=object)[keep].astype(str)
    p = np.asarray(labels_pred)[keep]
    return {
        "ari": float(adjusted_rand_score(t, p)),
        "nmi": float(normalized_mutual_info_score(t, p)),
        "n_scored": int(keep.sum()),
    }


def n_label_groups(labels_true):
    """Number of distinct non-missing labels (used as k for clustering)."""
    keep = _valid(labels_true)
    return len(set(np.asarray(labels_true, dtype=object)[keep].astype(str)))


def evaluate(Z, labels_true, k=None, seed=0):
    """Cluster with k = number of labelled groups (unless given), then score."""
    k = k or n_label_groups(labels_true)
    pred = cluster(Z, k, seed=seed)
    out = score(labels_true, pred)
    out["k"] = k
    return out, pred


def results_rows(experiment_id, dataset, sample_id, method, mode, seed, fold,
                 scores):
    """Rows in the shared results/<experiment_id>.csv format."""
    return [
        {"experiment_id": experiment_id, "dataset": dataset,
         "sample_id": sample_id, "method": method, "mode": mode,
         "seed": seed, "fold": fold, "metric": m, "value": scores[m]}
        for m in ("ari", "nmi")
    ]
