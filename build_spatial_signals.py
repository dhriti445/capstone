
import anndata as ad
import numpy as np
from scipy import sparse
from pathlib import Path

INPUT = "processed/TENX53_hvg_v3.h5ad"
OUTPUT = Path("processed/TENX53_spatial_signals.npz")
GENES_OUTPUT = Path("processed/TENX53_hvg_genes.txt")

adata = ad.read_h5ad(INPUT)

# Select the 2,000 HVGs in their saved rank order.
hvg_mask = adata.var["highly_variable"].to_numpy()
hvg_var = adata.var.loc[hvg_mask].copy()
hvg_var = hvg_var.sort_values("highly_variable_rank")

gene_indices = adata.var_names.get_indexer(hvg_var.index)
X = adata.X[:, gene_indices]

# Convert only the selected matrix to dense form.
# 4,898 x 2,000 float32 values require about 39 MB.
if sparse.issparse(X):
    signals = X.toarray().astype(np.float32)
else:
    signals = np.asarray(X, dtype=np.float32)

coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
spot_ids = adata.obs_names.to_numpy()

assert signals.shape == (adata.n_obs, 2000)
assert coords.shape == (adata.n_obs, 2)
assert np.isfinite(signals).all()
assert np.isfinite(coords).all()

# Save data and alignment information.
np.savez_compressed(
    OUTPUT,
    signals=signals,
    coords=coords,
    spot_ids=spot_ids,
    gene_names=hvg_var.index.to_numpy()
)

GENES_OUTPUT.write_text("\n".join(hvg_var.index) + "\n")

print("Signal matrix:", signals.shape)
print("Spatial coordinates:", coords.shape)
print("Unique spot IDs:", len(np.unique(spot_ids)))
print("Unique selected genes:", len(np.unique(hvg_var.index)))
print("Signal range:", float(signals.min()), "to", float(signals.max()))
print("Saved:", OUTPUT)
print("Gene list:", GENES_OUTPUT)

adata.file.close()
