
import anndata as ad
import numpy as np

path = "hest_samples/st/INT1.h5ad"
adata = ad.read_h5ad(path, backed="r")

print("File:", path)
print("Shape:", adata.shape)
print("X dtype:", adata.X.dtype)
print("X encoding:", adata.X.__class__.__name__)

print("\nObservation metadata:")
print(adata.obs.head())

print("\nGene metadata:")
print(adata.var.head())

print("\nUnstructured metadata:")
for key in adata.uns.keys():
    print(key)

adata.file.close()
