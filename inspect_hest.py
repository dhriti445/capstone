
import scanpy as sc

path = "hest_data/st/TENX53.h5ad"

adata = sc.read_h5ad(path)

print("Dataset shape:", adata.shape)
print("Spots:", adata.n_obs)
print("Genes:", adata.n_vars)

print("\nSpot metadata:")
print(adata.obs.columns.tolist())

print("\nGene metadata:")
print(adata.var.columns.tolist())

print("\nSpatial coordinates:")
print(adata.obsm.keys())

print("\nAvailable layers:")
print(list(adata.layers.keys()))

print("\nUnstructured metadata:")
print(list(adata.uns.keys()))
