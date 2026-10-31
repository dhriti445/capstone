
import anndata as ad
import numpy as np
import networkx as nx

from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, save_npz

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT = Path("processed/TENX53_hvg_v3.h5ad")
OUTDIR = Path("processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load spatial coordinates
# --------------------------------------------------
adata = ad.read_h5ad(INPUT)
coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
n = adata.n_obs

assert coords.shape == (n, 2)
assert np.isfinite(coords).all()

# Reproduce the previously validated radius.
tree = cKDTree(coords)
distances, _ = tree.query(coords, k=2)
radius = 1.05 * np.median(distances[:, 1])

# --------------------------------------------------
# 2. Build graph with deterministic vertex/edge IDs
# --------------------------------------------------
pairs = sorted(tree.query_pairs(radius))

G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(pairs)

edges = sorted(
    (min(u, v), max(u, v))
    for u, v in G.edges()
)
edge_index = {edge: i for i, edge in enumerate(edges)}

assert nx.is_connected(G)

# --------------------------------------------------
# 3. Build the 2-dimensional clique complex
# --------------------------------------------------
# Include a triangle whenever all three of its edges
# are present in the spatial graph.
triangles = sorted(
    tuple(sorted(clique))
    for clique in nx.enumerate_all_cliques(G)
    if len(clique) == 3
)

# --------------------------------------------------
# 4. B1: vertex-edge boundary matrix
# --------------------------------------------------
# Edge (u,v) is oriented u -> v.
# Its boundary is v - u.
r1, c1, v1 = [], [], []

for j, (u, v) in enumerate(edges):
    r1.extend([u, v])
    c1.extend([j, j])
    v1.extend([-1, 1])

B1 = coo_matrix(
    (v1, (r1, c1)),
    shape=(n, len(edges)),
    dtype=np.int64
).tocsr()

# --------------------------------------------------
# 5. B2: edge-triangle boundary matrix
# --------------------------------------------------
# For oriented triangle (u,v,w):
# boundary = (v,w) - (u,w) + (u,v).
r2, c2, v2 = [], [], []

for j, (u, v, w) in enumerate(triangles):
    boundary_edges = [
        ((v, w), 1),
        ((u, w), -1),
        ((u, v), 1),
    ]

    for edge, sign in boundary_edges:
        r2.append(edge_index[edge])
        c2.append(j)
        v2.append(sign)

B2 = coo_matrix(
    (v2, (r2, c2)),
    shape=(len(edges), len(triangles)),
    dtype=np.int64
).tocsr()

# --------------------------------------------------
# 6. Verify the chain-complex identity
# --------------------------------------------------
boundary_product = B1 @ B2

if boundary_product.nnz != 0:
    raise ValueError("Boundary identity B1 @ B2 = 0 failed.")

# --------------------------------------------------
# 7. Construct the combinatorial Hodge 1-Laplacian
# --------------------------------------------------
# L1 = B1.T @ B1 + B2 @ B2.T
L1_lower = B1.T @ B1
L1_upper = B2 @ B2.T
L1 = (L1_lower + L1_upper).tocsr()

# --------------------------------------------------
# 8. Save matrices and index mappings
# --------------------------------------------------
save_npz(OUTDIR / "TENX53_B1.npz", B1)
save_npz(OUTDIR / "TENX53_B2.npz", B2)
save_npz(OUTDIR / "TENX53_L1.npz", L1)

np.savez_compressed(
    OUTDIR / "TENX53_complex_index.npz",
    edges=np.asarray(edges, dtype=np.int32).reshape(-1, 2),
    triangles=np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
    radius=np.asarray(radius),
    vertex_ids=adata.obs_names.to_numpy(),
)

# --------------------------------------------------
# 9. Report
# --------------------------------------------------
print("Hodge operator construction complete")
print("------------------------------------")
print("Vertices:", n)
print("Edges:", len(edges))
print("Triangles:", len(triangles))
print("Radius (pixels):", radius)
print("B1 shape:", B1.shape)
print("B2 shape:", B2.shape)
print("Boundary identity:", boundary_product.nnz == 0)
print("L1 shape:", L1.shape)
print("L1 nonzero entries:", L1.nnz)
print("Connected components:", nx.number_connected_components(G))
print("\nSaved files:")
print(OUTDIR / "TENX53_B1.npz")
print(OUTDIR / "TENX53_B2.npz")
print(OUTDIR / "TENX53_L1.npz")
print(OUTDIR / "TENX53_complex_index.npz")

adata.file.close()
