
import anndata as ad
import scanpy as sc
from pathlib import Path

INPUT = "processed/TENX53_processed.h5ad"
OUTDIR = Path("processed")
OUTPUT = OUTDIR / "TENX53_hvg.h5ad"

adata = ad.read_h5ad(INPUT)

print("Input shape:", adata.shape)

# Preserve normalized expression in X and count-like values in layers.
# Selections are made using the normalized expression matrix.
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor="seurat",
    layer=None,
    subset=False
)

print("Highly variable genes:", int(adata.var["highly_variable"].sum()))

# Save the full processed dataset with HVG annotations.
# We retain all genes for now; downstream analyses can select the HVGs.
adata.write_h5ad(OUTPUT, compression="gzip")

print("Saved:", OUTPUT)
print("Output shape:", adata.shape)
print("\nTop variable genes:")
print(
    adata.var.loc[
        adata.var["highly_variable"],
        ["highly_variable", "means", "variances"]
    ].sort_values("variances", ascending=False).head(20)
)
