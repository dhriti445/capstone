
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

OUT = Path("processed")
OUT.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Load the existing spatial expression signals
# --------------------------------------------------
data = np.load(
    OUT / "TENX53_spatial_signals.npz",
    allow_pickle=True
)

X = data["signals"].astype(np.float64)  # spots x genes
coords = data["coords"].astype(np.float64)
genes = data["gene_names"].astype(str)

n_spots, n_genes = X.shape

assert coords.shape == (n_spots, 2)
assert len(genes) == n_genes

print("TENX53 SPATIAL EXPRESSION ANALYSIS")
print("=" * 48)
print(f"Spots: {n_spots}")
print(f"Genes: {n_genes}")

# --------------------------------------------------
# 2. Load the existing spatial edges
# --------------------------------------------------
complex_data = np.load(OUT / "TENX53_complex_index.npz")
edges = complex_data["edges"].astype(np.int64)

assert edges.ndim == 2 and edges.shape[1] == 2
assert edges.min() >= 0 and edges.max() < n_spots

u, v = edges[:, 0], edges[:, 1]

# --------------------------------------------------
# 3. Compute expression differences across edges
# --------------------------------------------------
# Each edge has a consistent orientation (u -> v).
# The sign depends on orientation; squared differences
# do not.
differences = X[v] - X[u]

mean_squared_difference = np.mean(differences ** 2, axis=0)
mean_absolute_difference = np.mean(np.abs(differences), axis=0)

# Conventional variance across spots
gene_variance = np.var(X, axis=0)

# Spatial roughness relative to gene variance.
# Larger values indicate greater variation across
# neighboring spots relative to total expression variance.
spatial_roughness = np.divide(
    mean_squared_difference,
    2.0 * gene_variance,
    out=np.zeros_like(mean_squared_difference),
    where=gene_variance > 1e-12,
)

# --------------------------------------------------
# 4. Measure correlation between expression and roughness
# --------------------------------------------------
valid = (
    np.isfinite(spatial_roughness)
    & np.isfinite(gene_variance)
    & (gene_variance > 1e-12)
)

rho, p_value = spearmanr(
    gene_variance[valid],
    spatial_roughness[valid],
)

print(f"\nSpatial edges: {len(edges)}")
print(f"Genes with nonzero variance: {valid.sum()}")
print(f"Spearman correlation (variance vs roughness): {rho:.4f}")
print(f"Correlation p-value: {p_value:.4e}")

# --------------------------------------------------
# 5. Save gene-level results
# --------------------------------------------------
results = pd.DataFrame({
    "gene": genes,
    "gene_variance": gene_variance,
    "mean_squared_edge_difference": mean_squared_difference,
    "mean_absolute_edge_difference": mean_absolute_difference,
    "spatial_roughness": spatial_roughness,
})

results = results.replace([np.inf, -np.inf], np.nan)
results = results.sort_values("spatial_roughness", ascending=False)

csv_path = OUT / "TENX53_spatial_expression_results.csv"
results.to_csv(csv_path, index=False)

# --------------------------------------------------
# 6. Plot the 20 genes with highest spatial roughness
# --------------------------------------------------
top = results.head(20).sort_values("spatial_roughness")

plt.figure(figsize=(10, 8))
plt.barh(top["gene"], top["spatial_roughness"])
plt.xlabel("Spatial roughness")
plt.ylabel("Gene")
plt.title("TENX53: Top 20 Genes by Spatial Roughness")
plt.tight_layout()

plot_path = OUT / "TENX53_top_spatial_roughness.png"
plt.savefig(plot_path, dpi=200)
plt.close()

# --------------------------------------------------
# 7. Plot variance against spatial roughness
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    gene_variance[valid],
    spatial_roughness[valid],
    s=8,
    alpha=0.5,
)
plt.xlabel("Gene-expression variance across spots")
plt.ylabel("Spatial roughness")
plt.title("TENX53: Expression Variance vs Spatial Roughness")
plt.tight_layout()

scatter_path = OUT / "TENX53_variance_vs_roughness.png"
plt.savefig(scatter_path, dpi=200)
plt.close()

print("\nSaved outputs:")
print(csv_path)
print(plot_path)
print(scatter_path)

print("\nTop 10 genes by spatial roughness:")
print(results.head(10).to_string(index=False))
