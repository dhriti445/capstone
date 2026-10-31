
import anndata as ad
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

path = "hest_samples/st/TENX53.h5ad"
adata = ad.read_h5ad(path, backed="r")

coords = np.asarray(adata.obsm["spatial"], dtype=float)

# Each spot's nearest other spot
tree = cKDTree(coords)
distances, indices = tree.query(coords, k=2)
nearest_distances = distances[:, 1]

print("Nearest-neighbor distance statistics (pixels):")
print("Minimum:", nearest_distances.min())
print("25th percentile:", np.percentile(nearest_distances, 25))
print("Median:", np.median(nearest_distances))
print("75th percentile:", np.percentile(nearest_distances, 75))
print("Maximum:", nearest_distances.max())

# Inspect distance distribution before choosing a graph rule
plt.figure(figsize=(8, 5))
plt.hist(nearest_distances, bins=50)
plt.xlabel("Nearest-neighbor distance (pixels)")
plt.ylabel("Number of spots")
plt.title("TENX53 spatial spot spacing")
plt.tight_layout()
plt.savefig("TENX53_neighbor_distances.png", dpi=200)
plt.show()

adata.file.close()
