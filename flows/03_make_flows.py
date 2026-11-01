"""Step 3 - build the edge flows and save them in the v0.1 layout.

Two kinds of flow, saved to two files:
  * product (baseline): f(i->j) = L(i)R(j) - L(j)R(i) for each pair in lr_pairs.csv
  * COMMOT (main):      run COMMOT, then f(i->j) = P[i,j] - P[j,i] per pathway

Run:  python 03_make_flows.py            (both)
      python 03_make_flows.py product    (only the fast baseline)
"""
import sys

import numpy as np
import pandas as pd
import scanpy as sc

import config as C
from flows import (load_complex, align_to_complex, preprocess, product_flows,
                   commot_flows, save_flows)
from plots import median_edge_length

which = sys.argv[1] if len(sys.argv) > 1 else "both"
C.OUT_DIR.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(C.H5AD_IN)
edges, triangles, spot_ids = load_complex(C.COMPLEX_DIR)
adata, _ = align_to_complex(adata, spot_ids)
adata = preprocess(adata, C.TARGET_SUM, C.SCALE_QUANTILE)

# ---------------------------------------------------------------- product
if which in ("both", "product"):
    pairs = pd.read_csv(C.OUT_DIR / "lr_pairs.csv")
    flows, names = product_flows(adata, edges, list(zip(pairs.ligand, pairs.receptor)))
    out = adata.copy()
    save_flows(out, edges, triangles, flows, names, C.FLOWS_PRODUCT, kind="product")

# ---------------------------------------------------------------- COMMOT
if which in ("both", "commot"):
    import commot as ct

    xy = np.asarray(adata.obsm["spatial"])[:, :2]
    dis_thr = C.COMMOT_RANGE_FACTOR * median_edge_length(xy, edges)   # same units as obsm['spatial']
    print(f"COMMOT distance threshold: {dis_thr:.1f} (coordinate units)")

    df = ct.pp.ligand_receptor_database(species="human", signaling_type=C.COMMOT_SIGNALING_TYPE,
                                        database="CellChat")
    df = ct.pp.filter_lr_database(df, adata, min_cell_pct=0.05)
    print(f"COMMOT will score {len(df)} L-R pairs in {df.iloc[:, 2].nunique()} pathways")

    ad_c = adata.copy()    # COMMOT reads the log-normalised adata.X
    ct.tl.spatial_communication(ad_c, database_name="cellchat", df_ligrec=df,
                                dis_thr=dis_thr, heteromeric=True, pathway_sum=True)

    pathways = sorted(df.iloc[:, 2].unique())
    flows, names, kept = commot_flows(ad_c, edges, pathways, database_name="cellchat")

    # Keep the strongest pathways by flow energy (the architecture doc uses 32;
    # one Visium section usually has fewer that carry real signal).
    energy = (flows ** 2).sum(axis=0)
    order = np.argsort(energy)[::-1][:32]
    flows, names, kept = flows[:, order], [names[k] for k in order], kept[order]
    report = pd.DataFrame({"pathway": names, "energy": energy[order], "share_on_complex_edges": kept})
    report.to_csv(C.OUT_DIR / "commot_pathways.csv", index=False)
    print(report.head(15).to_string(index=False))
    print("share_on_complex_edges = fraction of COMMOT's signal that sits on complex edges;"
          " if it is low, lower COMMOT_RANGE_FACTOR in config.py")

    out = adata.copy()
    save_flows(out, edges, triangles, flows, names, C.FLOWS_COMMOT, kind="commot")
