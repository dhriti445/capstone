"""Inspect the cached DLPFC files: what does the 'cluster' column hold?"""
import os
import sys
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

IDS = ["151507", "151508", "151509", "151510", "151669", "151670",
       "151671", "151672", "151673", "151674", "151675", "151676"]
os.makedirs("figures", exist_ok=True)
for sid in IDS:
    p = hf_hub_download(repo_id="han-shu/st_datasets", repo_type="dataset",
                        filename=f"DLPFC/DLPFC_{sid}.h5ad")
    a = sc.read_h5ad(p)
    cl = a.obs["cluster"]
    vc = cl.value_counts(dropna=False).sort_index()
    print(f"{sid}: spots={a.n_obs} genes={a.n_vars} dtype={cl.dtype} "
          f"distinct={cl.nunique()} NaN={int(cl.isna().sum())} "
          f"in_tissue_sum={int(a.obs['in_tissue'].sum())}")
    print("   values:", dict(vc))
    if sid == IDS[0]:
        print("   obsm:", list(a.obsm.keys()), " layers:", list(a.layers.keys()),
              " uns:", list(a.uns.keys()))
        print(a.obs.head(3).to_string())
    if sid in ("151673", "151669"):
        xy = (np.asarray(a.obsm["spatial"], float) if "spatial" in a.obsm else
              np.column_stack([a.obs["array_col"], -a.obs["array_row"]]))
        fig, ax = plt.subplots(figsize=(5, 5))
        sca = ax.scatter(xy[:, 0], xy[:, 1], c=cl.astype(str).astype("category").cat.codes,
                         s=4, cmap="tab10")
        if "spatial" in a.obsm:
            ax.invert_yaxis()
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{sid}: colour = 'cluster'")
        fig.savefig(f"figures/dlpfc_cluster_{sid}.png", dpi=120)
        plt.close(fig)
