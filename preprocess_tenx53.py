
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

path = "hest_samples/st/TENX53.h5ad"
adata = ad.read_h5ad(path)

X = adata.X

print("Shape:", X.shape)
print("Storage type:", type(X).__name__)
print("Dtype:", X.dtype)

# Compute row sums and detected-gene counts.
# Sparse operations avoid densifying the full matrix.
if sparse.issparse(X):
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    col_sums = np.asarray(X.sum(axis=0)).ravel()
else:
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    col_sums = np.asarray(X.sum(axis=0)).ravel()

print("\nFull-matrix checks:")
print("All row sums finite:", np.isfinite(row_sums).all())
print("All row sums nonnegative:", (row_sums >= 0).all())
print("Nonzero genes:", int(np.sum(col_sums > 0)))
print("Zero-count spots:", int(np.sum(row_sums == 0)))
print("Zero-count genes:", int(np.sum(col_sums == 0)))

print("\nRecomputed QC summaries:")
for name, values in [
    ("total_counts", row_sums),
    ("n_genes_detected", detected),
]:
    print(f"\n{name}:")
    print(pd.Series(values).describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).round(2))

print("\nExisting vs recomputed QC:")
for col, values in [
    ("total_counts", row_sums),
    ("n_genes_by_counts", detected),
]:
    existing = adata.obs[col].to_numpy(dtype=float)
    print(
        col,
        "| exact matches:", int(np.sum(np.isclose(existing, values))),
        "/", len(values),
        "| max absolute difference:",
        float(np.max(np.abs(existing - values)))
    )

print("\nExisting mitochondrial percentage summary:")
print(adata.obs["pct_counts_mito"].describe().round(3))

print("\nMatrix values are not normalized by this script.")
