"""First Hodge-aware model run on REAL HEST breast flows (TENX53).

Usage:
  python scripts/run_model_real.py data/starter.h5ad "data/flows/New Folder With Items/TENX53_flows_commot.h5ad"
  options: --epochs 60  --seeds 0 1 2  --niches 5

What it does (Deeptha's model and pretraining, unchanged):
  1. builds the cell complex of the slide and loads Disha's 32 COMMOT pathway flows
  2. pretrains Hodge-SCNN and the no-split SCNN by hiding 15% of edges and
     reconstructing them (no labels used)
  3. clusters the spot embeddings into niches (k-means) and compares them with
     niches from expression alone
  4. checks how stable the niches are across random seeds

TENX53 has no expert labels, so there is NO accuracy score here. This is a
first real run: loss curves, niche maps, and stability.
Outputs: figures/model_real_TENX53.png, results/model_real_TENX53.csv
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
# make the repo root importable, so `model` is found when run as scripts/run_model_real.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from hodgecc.visium import build_visium_complex  # noqa: E402
from model.pretrain import run  # noqa: E402

MODES = ["hodge", "nosplit"]


def kmeans_labels(X, k, seed):
    X = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def main(starter, flows_path, epochs, seeds, k):
    a = ad.read_h5ad(starter)
    r, c = a.obs["array_row"].values, a.obs["array_col"].values
    xy = np.asarray(a.obsm["spatial"], float)
    _, _, B1, B2 = build_visium_complex(r, c)
    N, E = B1.shape

    fl = ad.read_h5ad(flows_path)
    F = np.asarray(fl.uns["flows"], float)
    assert F.shape[0] == E, "flow file does not match this complex"
    F = F / np.sqrt((F ** 2).mean(axis=0, keepdims=True) + 1e-12)   # unit RMS per pathway
    print(f"spots {N}, edges {E}, triangles {B2.shape[1]}, pathways {F.shape[1]}")

    # expression-only niches (no flows, no Hodge)
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    pcs = PCA(n_components=20, random_state=0).fit_transform(X)
    expr = {s: kmeans_labels(pcs, k, s) for s in seeds}

    results = {}
    t0 = time.time()
    for mode in MODES:
        runs = []
        for s in seeds:
            Z, hist = run(B1, B2, F, mode=mode, epochs=epochs, seed=s)
            runs.append((kmeans_labels(Z, k, s), hist))
            print(f"  {mode:8s} seed {s}: masked MSE {hist[0]:.3f} -> {hist[-1]:.3f}"
                  f"   ({time.time()-t0:.0f}s)")
        results[mode] = runs

    def stability(label_list):
        ps = [adjusted_rand_score(label_list[i], label_list[j])
              for i in range(len(label_list)) for j in range(i + 1, len(label_list))]
        return float(np.mean(ps)) if ps else float("nan")

    rows = []
    summary = {}
    for mode in MODES:
        labs = [l for l, _ in results[mode]]
        h0 = np.mean([h[0] for _, h in results[mode]])
        h1 = np.mean([h[-1] for _, h in results[mode]])
        agree = np.mean([adjusted_rand_score(labs[i], expr[seeds[i]]) for i in range(len(seeds))])
        summary[mode] = dict(loss0=h0, loss1=h1, stability=stability(labs), vs_expression=agree)
        rows.append(dict(method=mode, niches=k, seeds=len(seeds), loss_first=h0, loss_last=h1,
                         seed_stability_ari=summary[mode]["stability"],
                         agreement_with_expression_ari=agree))
    rows.append(dict(method="expression-kmeans", niches=k, seeds=len(seeds),
                     seed_stability_ari=stability([expr[s] for s in seeds])))
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv("results/model_real_TENX53.csv", index=False)
    print()
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # figure: loss curves + three niche maps
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 4, 1)
    for mode, col in (("hodge", "#1F6FB5"), ("nosplit", "#A9B2CB")):
        H = np.array([h for _, h in results[mode]])
        ax.plot(H.mean(0), color=col, lw=2, label={"hodge": "Hodge-SCNN", "nosplit": "no Hodge split"}[mode])
        ax.fill_between(range(H.shape[1]), H.min(0), H.max(0), color=col, alpha=0.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("masked reconstruction error")
    ax.set_title("Pretraining on real flows"); ax.legend(); ax.grid(alpha=0.3)
    panels = [("Hodge-SCNN niches", results["hodge"][0][0]),
              ("No-split SCNN niches", results["nosplit"][0][0]),
              ("Expression-only niches", expr[seeds[0]])]
    for i, (title, lab) in enumerate(panels, start=2):
        ax = fig.add_subplot(1, 4, i)
        ax.scatter(xy[:, 0], xy[:, 1], c=lab, cmap="tab10", s=5)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{title} (k={k}; colours arbitrary)", fontsize=9)
    fig.suptitle("TENX53 (HEST breast IDC): first Hodge-aware model run, no labels", fontweight="bold")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/model_real_TENX53.png", dpi=150)
    print("saved figures/model_real_TENX53.png and results/model_real_TENX53.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("starter"); ap.add_argument("flows")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--niches", type=int, default=5)
    a = ap.parse_args()
    main(a.starter, a.flows, a.epochs, a.seeds, a.niches)
