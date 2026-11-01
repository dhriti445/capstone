"""Build the cell complex for one real Visium slice and report its topology.

Usage:  python scripts/real_slice_complex.py path/to/slice.h5ad
Needs obs columns array_row, array_col and obsm['spatial'].
Outputs:
  figures/real_<sample>.png            tissue, mesh and holes
  data/complex/<sample>_complex.npz    edges, triangles, spot ids (share this)
  data/complex/<sample>_complex.h5ad   optional copy with adata.uns['complex']
"""
import os
import sys
import numpy as np
import gudhi
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hodgecc.visium import build_visium_complex, enclosed_gaps
from hodgecc.io import save_complex


def main(path):
    adata = sc.read_h5ad(path)
    if "in_tissue" in adata.obs:
        adata = adata[adata.obs["in_tissue"] == 1].copy()
    sample = str(adata.obs["sample_id"].iloc[0]) if "sample_id" in adata.obs \
        else os.path.splitext(os.path.basename(path))[0]
    r = adata.obs["array_row"].values
    c = adata.obs["array_col"].values
    n = adata.n_obs

    edges, tris, B1, B2 = build_visium_complex(r, c)
    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i])
    for e in edges:
        st.insert([int(e[0]), int(e[1])])
    for t in tris:
        st.insert([int(x) for x in t])
    st.compute_persistence()
    b0, b1 = st.betti_numbers()[:2]
    gaps = enclosed_gaps(r, c)

    print("sample:", sample)
    print("spots:", n, " edges:", len(edges), " triangles:", len(tris))
    print("Euler N-E+T:", n - len(edges) + len(tris), " (= b0 - b1 =", b0 - b1, ")")
    print("components b0:", b0, " holes b1:", b1)
    print("hole sizes (missing spots each):", [len(g) for g in gaps])

    # map lattice sites to pixel coordinates with an affine fit
    xy = np.asarray(adata.obsm["spatial"], float)
    A = np.column_stack([c, r, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)

    fig, ax = plt.subplots(figsize=(9, 7))
    seg = np.stack([xy[edges[:, 0]], xy[edges[:, 1]]], axis=1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(seg, colors="0.75", linewidths=0.4))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, color="0.4")
    for k, g in enumerate(gaps):
        pts = np.array([[cc, rr, 1.0] for rr, cc in g]) @ coef
        ax.scatter(pts[:, 0], pts[:, 1], s=60, color="red", marker="x")
        ax.annotate(f"{len(g)}", pts.mean(0), color="red", fontsize=9,
                    xytext=(6, 6), textcoords="offset points")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{sample}: {n} spots, {len(tris)} triangles, "
                 f"b0={b0}, b1={b1} (red = missing spots inside tissue)")
    ax.axis("off")
    os.makedirs("figures", exist_ok=True)
    out_png = os.path.join("figures", f"real_{sample}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("saved", out_png)

    os.makedirs(os.path.join("data", "complex"), exist_ok=True)

    # small file to share with teammates (no expression data)
    out_npz = os.path.join("data", "complex", f"{sample}_complex.npz")
    save_complex(out_npz, edges, tris, adata.obs_names, sample)
    print("saved", out_npz)

    # optional: slice + complex together as .h5ad (can fail on some
    # anndata versions, so it must not stop the script)
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass
    try:
        adata.uns["complex"] = {"edges": edges.astype(np.int64),
                                "triangles": tris.astype(np.int64)}
        out_h5 = os.path.join("data", "complex", f"{sample}_complex.h5ad")
        adata.write_h5ad(out_h5, compression="gzip")
        print("saved", out_h5)
    except Exception as ex:
        print("skipped .h5ad copy:", type(ex).__name__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/real_slice_complex.py path/to/slice.h5ad")
    main(sys.argv[1])
