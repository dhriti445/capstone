
from pathlib import Path
import numpy as np
import anndata as ad
from scipy import sparse

root = Path("hest_samples/st")
files = sorted(root.glob("*.h5ad"))

print("Total H5AD files:", len(files))

for path in files[:5]:
    print(f"\n--- {path.name} ---")

    a = ad.read_h5ad(path, backed="r")

    print("Shape:", a.shape)
    print("X type:", type(a.X).__name__)
    print("Layers:", list(a.layers.keys()))

    # Inspect a small matrix block without loading the full dataset
    block = a[:min(100, a.n_obs), :min(500, a.n_vars)].X

    if sparse.issparse(block):
        values = block.data
    else:
        values = np.asarray(block).ravel()

    if values.size:
        print("Min:", values.min())
        print("Max:", values.max())
        print("Integer-like:", np.allclose(values, np.round(values)))
        print("Nonnegative:", np.all(values >= 0))

    print("Spatial coordinates:", list(a.obsm.keys()))
    a.file.close()
