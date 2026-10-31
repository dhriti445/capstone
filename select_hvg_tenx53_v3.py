
import anndata as ad
import scanpy as sc
from pathlib import Path

INPUT = "processed/TENX53_processed.h5ad"
OUTPUT = Path("processed/TENX53_hvg_v3.h5ad")

adata = ad.read_h5ad(INPUT)

print("Input shape:", adata.shape)
print("Using count layer:", "counts" in adata.layers)

if "counts" not in adata.layers:
    raise ValueError("Expected counts layer is missing.")

# Seurat v3 expects count data, not log-normalized expression.
sc.pp.highly_variable_genes(
    adata,
    layer="counts",
    flavor="seurat_v3",
    n_top_genes=2000,
    subset=False
)

n_hvg = int(adata.var["highly_variable"].sum())
print("Highly variable genes:", n_hvg)

columns = [
    col for col in [
        "highly_variable",
        "means",
        "variances",
        "variances_norm",
        "highly_variable_rank"
    ]
    if col in adata.var.columns
]

print("\nAvailable statistics:", columns)

if "highly_variable_rank" in adata.var.columns:
    top = adata.var.loc[
        adata.var["highly_variable"]
    ].sort_values("highly_variable_rank")
else:
    top = adata.var.loc[adata.var["highly_variable"]]

print("\nTop 20 selected genes:")
print(top[columns].head(20).to_string())

adata.write_h5ad(OUTPUT, compression="gzip")

print("\nSaved:", OUTPUT)
print("Final shape:", adata.shape)
print("Original normalized X preserved.")
print("Count layer preserved.")
