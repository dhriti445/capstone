
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("processed")

# Load expression signals and graph structure.
data = np.load(
    OUT / "TENX53_spatial_signals.npz",
    allow_pickle=True,
)
X = data["signals"].astype(np.float64)
genes = data["gene_names"].astype(str)

complex_data = np.load(OUT / "TENX53_complex_index.npz")
edges = complex_data["edges"].astype(np.int64)
triangles = complex_data["triangles"].astype(np.int64)

n_spots, n_genes = X.shape
n_edges = len(edges)
n_triangles = len(triangles)

# Canonical edge orientation: smaller vertex ID -> larger vertex ID.
edge_lookup = {
    (int(u), int(v)): i
    for i, (u, v) in enumerate(edges)
}

# Build the oriented boundary of each triangle.
# For triangle (a,b,c), a < b < c:
# boundary = (b,c) - (a,c) + (a,b)
triangle_edges = np.empty((n_triangles, 3), dtype=np.int64)
triangle_signs = np.array([1, -1, 1], dtype=np.float64)

for i, (a, b, c) in enumerate(triangles):
    triangle_edges[i] = [
        edge_lookup[(int(b), int(c))],
        edge_lookup[(int(a), int(c))],
        edge_lookup[(int(a), int(b))],
    ]

# Expression-induced edge flow:
# f(u -> v) = x(v) - x(u)
u, v = edges[:, 0], edges[:, 1]
edge_flow = X[v] - X[u]

# Circulation around each triangle.
# A vertex-derived gradient flow should have zero circulation,
# up to floating-point precision.
circulation = (
    edge_flow[triangle_edges]
    * triangle_signs[None, :, None]
).sum(axis=1)

max_abs_circulation = np.max(np.abs(circulation))
mean_abs_circulation = np.mean(np.abs(circulation))

print("HODGE EDGE-SIGNAL SANITY CHECK")
print("=" * 42)
print(f"Spots: {n_spots}")
print(f"Genes: {n_genes}")
print(f"Edges: {n_edges}")
print(f"Triangles: {n_triangles}")
print(f"Max absolute triangle circulation: {max_abs_circulation:.3e}")
print(f"Mean absolute triangle circulation: {mean_abs_circulation:.3e}")

# Summarize edge-flow energy per gene.
edge_energy = np.mean(edge_flow ** 2, axis=0)

results = pd.DataFrame({
    "gene": genes,
    "edge_flow_energy": edge_energy,
})

results = results.sort_values("edge_flow_energy", ascending=False)
results.to_csv(OUT / "TENX53_hodge_signal_sanity.csv", index=False)

print("\nTop 10 genes by edge-flow energy:")
print(results.head(10).to_string(index=False))
print("\nSaved: processed/TENX53_hodge_signal_sanity.csv")

if max_abs_circulation < 1e-10:
    print("\nPASS: Edge signals are consistent with gradient flows.")
else:
    print("\nWARNING: Check triangle orientation and edge indexing.")
