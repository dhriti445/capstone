
import anndata as ad
import numpy as np
from scipy import sparse

path = "hest_samples/st/TENX53.h5ad"

adata = ad.read_h5ad(path, backed="r")

print("Shape:", adata.shape)
print("X dtype:", adata.X.dtype)
print("Layers:", list(adata.layers.keys()))
print("Raw available:", adata.raw is not None)

# Inspect a small portion of the matrix without loading
# the entire dataset into memory.
X = adata.X[:100, :1000]

if sparse.issparse(X):
    values = X.data
else:
    values = np.asarray(X).ravel()

print("\nSampled matrix checks:")
print("Minimum:", values.min() if values.size else None)
print("Maximum:", values.max() if values.size else None)
print("All finite:", np.isfinite(values).all())
print("All nonnegative:", (values >= 0).all())
print(
    "Integer-like:",
    np.allclose(values, np.round(values))
)

print("\nFirst 10 observation IDs:")
print(adata.obs_names[:10].tolist())

print("\nFirst 10 gene IDs:")
print(adata.var_names[:10].tolist())

print("\nSpatial coordinate sample:")
print(adata.obsm["spatial"][:5])

adata.file.close()
