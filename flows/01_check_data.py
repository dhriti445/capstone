"""Step 1 - check the HEST .h5ad against Dhriti's complex.

Run:  python 01_check_data.py
Prints a short report and saves outputs/figures/01_spots_and_holes.png
"""
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

import config as C
from flows import load_complex, build_incidence, align_to_complex
from plots import plot_spots, plot_holes, median_edge_length
C.FIG_DIR.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(C.H5AD_IN)
edges, triangles, spot_ids = load_complex(C.COMPLEX_DIR)
print(f"h5ad:    {adata.n_obs} spots x {adata.n_vars} genes")
print(f"complex: {len(spot_ids)} spots, {len(edges)} edges, {len(triangles)} triangles")

# 1. gene names must be symbols (e.g. 'TGFB1'), not Ensembl IDs ('ENSG...')
print("first genes:", list(adata.var_names[:5]))
if str(adata.var_names[0]).startswith("ENSG"):
    print("!! var_names are Ensembl IDs. Look in adata.var.columns for a symbol column and run:")
    print("   adata.var_names = adata.var['<symbol column>'].astype(str)")

# 2. spot order must match the complex
adata, dropped = align_to_complex(adata, spot_ids)
print(f"aligned to complex order ({dropped} spots in the .h5ad were not in the complex, e.g. removed by QC)")

# 3. coordinates
xy = np.asarray(adata.obsm["spatial"])[:, :2]
d = median_edge_length(xy, edges)
print(f"median edge length: {d:.1f} (in the units of obsm['spatial']; Visium spots are 100 um apart)")

# 4. complex is consistent
B1, B2 = build_incidence(adata.n_obs, edges, triangles)
chi = adata.n_obs - len(edges) + len(triangles)
print(f"B1 @ B2 == 0 OK; Euler characteristic {chi} -> beta1 = {1 - chi} holes (if one connected piece)")

# 5. picture: total counts + holes
counts = np.asarray(adata.X.sum(axis=1)).ravel()
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
plot_spots(adata, np.log1p(counts), ax=axes[0], title="log total counts per spot")
_, n_holes = plot_holes(adata, edges, triangles, ax=axes[1], title="Complex boundary: outer (grey), holes (red)")
fig.tight_layout()
fig.savefig(C.FIG_DIR / "01_spots_and_holes.png", dpi=150)
print(f"numbered hole loops drawn: {n_holes}")
print("saved", C.FIG_DIR / "01_spots_and_holes.png")
