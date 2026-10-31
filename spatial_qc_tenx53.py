
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "hest_samples/st/TENX53.h5ad"
adata = ad.read_h5ad(path, backed="r")

# Inspect coordinates
coords = np.asarray(adata.obsm["spatial"])
assert coords.shape == (adata.n_obs, 2)
assert np.isfinite(coords).all()

print("Coordinate bounds:")
print("X:", coords[:, 0].min(), "to", coords[:, 0].max())
print("Y:", coords[:, 1].min(), "to", coords[:, 1].max())
print("Unique coordinate pairs:", np.unique(coords, axis=0).shape[0])

# Inspect existing tissue annotations and QC metrics
print("\nIn-tissue values:")
print(adata.obs["in_tissue"].value_counts(dropna=False))

qc_cols = [
    "total_counts",
    "n_genes_by_counts",
    "pct_counts_mito",
]
print("\nQC summary:")
print(adata.obs[qc_cols].describe().round(2))

# Plot coordinate layout, colored by existing QC metrics.
# These are pixel coordinates; invert the y-axis for image convention.
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, col in zip(axes, qc_cols):
    values = adata.obs[col].to_numpy(dtype=float)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=5,
        linewidths=0
    )
    ax.set_title(col)
    ax.set_xlabel("Spatial x (pixels)")
    ax.set_ylabel("Spatial y (pixels)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.75)

plt.tight_layout()
plt.savefig("TENX53_spatial_qc.png", dpi=200)
plt.show()

adata.file.close()
