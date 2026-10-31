
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import json

SOURCE = Path("hest_samples/st/TENX53.h5ad")
OUTDIR = Path("processed")
OUTPUT = OUTDIR / "TENX53_processed.h5ad"
REPORT = OUTDIR / "TENX53_preprocessing_report.txt"

OUTDIR.mkdir(parents=True, exist_ok=True)

# 1. Load original data without modifying the source file.
adata = ad.read_h5ad(SOURCE)
print("Original shape:", adata.shape)

X = adata.X

# 2. Validate the full stored matrix.
values = X.data if sparse.issparse(X) else np.asarray(X).ravel()

if not np.isfinite(values).all():
    raise ValueError("Non-finite expression values found.")

if (values < 0).any():
    raise ValueError("Negative expression values found.")

if not np.allclose(values, np.round(values)):
    raise ValueError("Matrix is not integer-like; stopping.")

# 3. Remove genes with zero counts across all spots.
if sparse.issparse(X):
    gene_totals = np.asarray(X.sum(axis=0)).ravel()
else:
    gene_totals = np.asarray(X.sum(axis=0)).ravel()

keep_genes = gene_totals > 0
removed_genes = int((~keep_genes).sum())

adata = adata[:, keep_genes].copy()

# Preserve count values in a dedicated layer.
counts = adata.X.copy()
adata.layers["counts"] = counts

# 4. Keep every spot for this initial pass.
# No arbitrary spot filtering is applied.
library_sizes = np.asarray(
    counts.sum(axis=1)
).ravel()

if np.any(library_sizes <= 0):
    raise ValueError("Zero-count spots found; review before normalization.")

# 5. Normalize each spot to 10,000 total counts, then log1p.
target_sum = 10_000.0
scale = target_sum / library_sizes

if sparse.issparse(counts):
    normalized = sparse.diags(scale) @ counts
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data).astype(np.float32)
    adata.X = normalized
else:
    normalized = counts.astype(np.float32)
    normalized *= scale[:, None]
    adata.X = np.log1p(normalized).astype(np.float32)

# 6. Record the preprocessing choices.
adata.uns["preprocessing"] = {
    "input_file": str(SOURCE),
    "zero_count_genes_removed": removed_genes,
    "spots_filtered": 0,
    "normalization": "total counts per spot scaled to 10000",
    "transformation": "natural log1p",
    "normalized_matrix": "X",
    "count_matrix": "layers['counts']",
    "spatial_coordinates_preserved": True,
}

# 7. Validate output before saving.
assert adata.n_obs == 4898
assert adata.n_vars == 24387
assert "spatial" in adata.obsm
assert adata.obsm["spatial"].shape == (adata.n_obs, 2)
assert "counts" in adata.layers

normalized_values = (
    adata.X.data
    if sparse.issparse(adata.X)
    else np.asarray(adata.X).ravel()
)

assert np.isfinite(normalized_values).all()
assert (normalized_values >= 0).all()

# 8. Save processed data and a text report.
adata.write_h5ad(OUTPUT, compression="gzip")

report = [
    "TENX53 PREPROCESSING REPORT",
    f"Input: {SOURCE}",
    f"Output: {OUTPUT}",
    f"Original shape: 4898 x 36601",
    f"Processed shape: {adata.n_obs} x {adata.n_vars}",
    f"Zero-count genes removed: {removed_genes}",
    "Spots removed: 0",
    "Normalization: total counts per spot scaled to 10000",
    "Transformation: natural log1p",
    "X contains normalized log1p expression.",
    "layers['counts'] preserves the retained count-like matrix.",
    "Spatial coordinates preserved.",
    "Original input file was not modified.",
]

REPORT.write_text("\n".join(report) + "\n")

print("\nPreprocessing complete.")
print("Processed shape:", adata.shape)
print("Genes removed:", removed_genes)
print("Spots retained:", adata.n_obs)
print("Output:", OUTPUT)
print("Report:", REPORT)
print("Normalized expression range:",
      float(normalized_values.min()),
      "to", float(normalized_values.max()))
