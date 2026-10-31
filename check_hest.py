
from pathlib import Path
import anndata as ad

root = Path("hest_samples/st")
files = sorted(root.glob("*.h5ad"))

print("H5AD files found:", len(files))

# Inspect only the first sample initially
if not files:
    raise FileNotFoundError("No .h5ad files found in hest_samples/st")

path = files[0]
print("\nChecking:", path)

adata = ad.read_h5ad(path, backed="r")

print("Shape:", adata.shape)
print("Observations:", adata.n_obs)
print("Genes:", adata.n_vars)
print("Spatial coordinates:", list(adata.obsm.keys()))
print("Observation columns:", list(adata.obs.columns))
print("Variable columns:", list(adata.var.columns))

adata.file.close()
print("\nSample integrity check complete.")
