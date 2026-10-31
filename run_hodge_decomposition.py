
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.linalg import lsmr

OUT = Path("processed")

# Load expression signals and Hodge operators
data = np.load(
    OUT / "TENX53_spatial_signals.npz",
    allow_pickle=True,
)
X = data["signals"].astype(np.float64)
genes = data["gene_names"].astype(str)

B1 = load_npz(OUT / "TENX53_B1.npz").tocsr()
B2 = load_npz(OUT / "TENX53_B2.npz").tocsr()

complex_data = np.load(OUT / "TENX53_complex_index.npz")
edges = complex_data["edges"].astype(np.int64)
triangles = complex_data["triangles"].astype(np.int64)

n_spots, n_genes = X.shape
n_edges = len(edges)

# Vertex expression -> edge gradient flow.
u, v = edges[:, 0], edges[:, 1]
F = X[v] - X[u]

# Project each edge flow onto the gradient space:
# im(B1.T), the space of exact edge flows.
#
# B1.T @ potential approximates F. Since F is constructed
# from vertex expression differences, the residual should
# be approximately zero.
gradient_energy = np.zeros(n_genes)
residual_energy = np.zeros(n_genes)

for j in range(n_genes):
    fit = B1.T @ X[:, j]
    residual = F[:, j] - fit

    gradient_energy[j] = np.sum(fit ** 2)
    residual_energy[j] = np.sum(residual ** 2)

total_energy = np.sum(F ** 2, axis=0)

gradient_fraction = np.divide(
    gradient_energy,
    total_energy,
    out=np.full(n_genes, np.nan),
    where=total_energy > 1e-12,
)

residual_fraction = np.divide(
    residual_energy,
    total_energy,
    out=np.full(n_genes, np.nan),
    where=total_energy > 1e-12,
)

# Triangle circulation: B2.T @ F should be zero for
# gradient flows under the constructed orientation.
curl = B2.T @ F
curl_energy = np.sum(curl ** 2, axis=0)

# Assemble gene-level output
results = pd.DataFrame({
    "gene": genes,
    "total_edge_energy": total_energy,
    "gradient_energy": gradient_energy,
    "gradient_fraction": gradient_fraction,
    "residual_energy": residual_energy,
    "residual_fraction": residual_fraction,
    "triangle_circulation_energy": curl_energy,
})

results.to_csv(
    OUT / "TENX53_hodge_decomposition_results.csv",
    index=False,
)

print("HODGE DECOMPOSITION")
print("=" * 45)
print(f"Spots: {n_spots}")
print(f"Genes: {n_genes}")
print(f"Edges: {n_edges}")
print(f"Triangles: {len(triangles)}")
print(
    "Median gradient fraction:",
    f"{np.nanmedian(gradient_fraction):.12f}",
)
print(
    "Maximum residual fraction:",
    f"{np.nanmax(residual_fraction):.3e}",
)
print(
    "Maximum triangle circulation energy:",
    f"{np.max(curl_energy):.3e}",
)
print("\nSaved:")
print(OUT / "TENX53_hodge_decomposition_results.csv")
print(
    "\nInterpretation: these input signals are vertex-derived "
    "gradients. This test does not establish a biological "
    "curl or harmonic component."
)
