import anndata as ad
import scanpy as sc
from pathlib import Path

INPUT = "processed/TENX53_processed.h5ad"
OUTPUT = Path("processed/TENX53_hvg.h5ad")

adata = ad.read_h5ad(INPUT)
print("Input shape:", adata.shape)

# This reproduces the current selection method.
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor="seurat",
    layer=None,
    subset=False
)

print("Highly variable genes:",
      int(adata.var["highly_variable"].sum()))

print("Available HVG columns:", adata.var.columns.tolist())

# Save the complete dataset with the HVG annotations.
adata.write_h5ad(OUTPUT, compression="gzip")
print("Saved:", OUTPUT)
print("Output shape:", adata.shape)

hvg = adata.var.loc[adata.var["highly_variable"]]
print("\nFirst 20 selected genes:")
print(hvg.head(20).to_string())

print("\nHVG selection complete.")
