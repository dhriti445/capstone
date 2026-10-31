
import anndata as ad
import numpy as np
from scipy import sparse

path = "hest_samples/st/TENX53.h5ad"
adata = ad.read_h5ad(path, backed="r")

print("Shape:", adata.shape)
print("X dtype:", adata.X.dtype)
print("Layers:", list(adata.layers.keys()))
print("Observation columns:", list(adata.obs.columns))
print("Gene columns:", list(adata.var.columns))
print("Spatial coordinates:", list(adata.obsm.keys()))

print("\nExisting QC summary:")
for col in ["total_counts", "n_genes_by_counts", "pct_counts_mito"]:
    if col in adata.obs:
        print(f"\n{col}:")
        print(adata.obs[col].describe())

print("\nSpatial coordinate shape:",
      adata.obsm["spatial"].shape if "spatial" in adata.obsm else "Missing")

adata.file.close()
