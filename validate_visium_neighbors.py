
import anndata as ad
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter
import matplotlib.pyplot as plt

path = "hest_samples/st/TENX53.h5ad"
adata = ad.read_h5ad(path, backed="r")
coords = np.asarray(adata.obsm["spatial"], dtype=float)

tree = cKDTree(coords)

# Examine the nearest 7 points, including the point itself.
distances, indices = tree.query(coords, k=7)

# The first distance is zero (the point itself).
# Inspect the six closest other spots.
six_neighbor_distances = distances[:, 1:7]

print("Distances to the six nearest other spots:")
for rank in range(6):
    d = six_neighbor_distances[:, rank]
    print(
        f"Neighbor {rank + 1}: "
        f"median={np.median(d):.3f}, "
        f"25th={np.percentile(d, 25):.3f}, "
        f"75th={np.percentile(d, 75):.3f}, "
        f"max={d.max():.3f}"
    )

# Candidate immediate-neighbor radius based on the observed
# nearest-neighbor scale. This is diagnostic, not final.
candidate_radius = 1.05 * np.median(distances[:, 1])
pairs = tree.query_pairs(candidate_radius)

degree = np.zeros(len(coords), dtype=int)
for i, j in pairs:
    degree[i] += 1
    degree[j] += 1

print("\nCandidate radius:", candidate_radius)
print("Undirected edges:", len(pairs))
print("Degree distribution:", dict(sorted(Counter(degree).items())))
print("Isolated spots:", int(np.sum(degree == 0)))
print("Spots with six neighbors:", int(np.sum(degree == 6)))

plt.figure(figsize=(8, 5))
plt.hist(degree, bins=np.arange(degree.max() + 2) - 0.5)
plt.xticks(range(degree.max() + 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Number of spots")
plt.title("TENX53 candidate spatial graph degree distribution")
plt.tight_layout()
plt.savefig("TENX53_neighbor_degrees.png", dpi=200)
plt.show()

adata.file.close()
