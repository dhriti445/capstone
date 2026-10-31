
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = Path("processed")

data = np.load(
    OUT / "TENX53_spatial_signals.npz",
    allow_pickle=True
)

X = data["signals"].astype(np.float64)
genes = data["gene_names"].astype(str)

complex_data = np.load(OUT / "TENX53_complex_index.npz")
edges = complex_data["edges"].astype(np.int64)

n, p = X.shape
u, v = edges[:, 0], edges[:, 1]

# Center each gene across spots.
X -= X.mean(axis=0, keepdims=True)

# Moran's I with binary, undirected spatial weights.
# Each edge contributes in both directions.
denom = np.sum(X ** 2, axis=0)
edge_products = np.sum(X[u] * X[v], axis=0)

moran_i = np.divide(
    n * edge_products,
    len(edges) * denom,
    out=np.full(p, np.nan),
    where=denom > 1e-12,
)

results = pd.DataFrame({
    "gene": genes,
    "morans_i": moran_i,
})

results = results.replace([np.inf, -np.inf], np.nan)
results = results.dropna().sort_values(
    "morans_i", ascending=False
)

results.to_csv(
    OUT / "TENX53_morans_i_results.csv",
    index=False
)

print(f"Spots: {n} | Genes: {p} | Edges: {len(edges)}")
print("\nHighest Moran's I (spatial clustering):")
print(results.head(10).to_string(index=False))

print("\nLowest Moran's I (spatial contrast):")
print(results.tail(10).sort_values("morans_i").to_string(index=False))

print("\nSaved: processed/TENX53_morans_i_results.csv")
