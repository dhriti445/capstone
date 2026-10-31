
import anndata as ad
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

path = "hest_samples/st/TENX53.h5ad"
adata = ad.read_h5ad(path, backed="r")

coords = np.asarray(adata.obsm["spatial"], dtype=float)
n = len(coords)

tree = cKDTree(coords)
nearest_distances, _ = tree.query(coords, k=2)
radius = 1.05 * np.median(nearest_distances[:, 1])

pairs = np.array(
    list(tree.query_pairs(radius)),
    dtype=int
).reshape(-1, 2)

# Undirected adjacency matrix
rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
data = np.ones(len(rows), dtype=np.uint8)

adjacency = coo_matrix(
    (data, (rows, cols)),
    shape=(n, n)
).tocsr()

n_components, labels = connected_components(
    adjacency,
    directed=False
)

edge_lengths = np.linalg.norm(
    coords[pairs[:, 0]] - coords[pairs[:, 1]],
    axis=1
)

print("Radius:", radius)
print("Nodes:", n)
print("Undirected edges:", len(pairs))
print("Connected components:", n_components)
print("Largest component size:", np.bincount(labels).max())

print("\nEdge-length statistics (pixels):")
for q in [0, 25, 50, 75, 95, 100]:
    print(f"{q}th percentile:", np.percentile(edge_lengths, q))

print("\nEdges longer than 1.5x median nearest-neighbor distance:",
      int(np.sum(edge_lengths > 1.5 * np.median(nearest_distances[:, 1]))))

adata.file.close()
