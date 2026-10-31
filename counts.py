
import anndata as ad
import numpy as np
from scipy import sparse

adata = ad.read_h5ad("hest_samples/st/INT1.h5ad", backed="r")

# Read only a small block
block = adata[:100, :500].X
values = block.data if sparse.issparse(block) else np.asarray(block).ravel()

print("Shape:", adata.shape)
print("Stored dtype:", adata.X.dtype)
print("Minimum:", values.min())
print("Maximum:", values.max())
print("Integer-like:", np.allclose(values, np.round(values)))
print("Existing QC metrics:", [
    c for c in adata.obs.columns
    if "count" in c.lower() or "gene" in c.lower()
])

adata.file.close()
