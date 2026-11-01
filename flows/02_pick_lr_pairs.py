"""Step 2 - choose the ligand-receptor pairs to turn into flows.

Uses LIANA+ spatial bivariate scoring: for every L-R pair in the consensus
resource it measures how strongly ligand and receptor co-occur in
neighbouring spots (global Moran's R), with a permutation p-value.

Run:  python 02_pick_lr_pairs.py
Writes outputs/lr_ranking.csv (all pairs) and outputs/lr_pairs.csv (the chosen ones).
Look at lr_ranking.csv yourself and edit lr_pairs.csv if a better-known
breast-cancer pair is just below the cut - the choice is yours to justify.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import liana as li

import config as C
from flows import load_complex, align_to_complex, preprocess
from hodgecc.plots import median_edge_length

C.OUT_DIR.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(C.H5AD_IN)
edges, triangles, spot_ids = load_complex(C.COMPLEX_DIR)
adata, _ = align_to_complex(adata, spot_ids)
adata = preprocess(adata, C.TARGET_SUM, C.SCALE_QUANTILE)   # adata.X is now log-normalised

# Spatial neighbours: Gaussian kernel whose width matches the complex edges
xy = np.asarray(adata.obsm["spatial"])[:, :2]
bandwidth = C.COMMOT_RANGE_FACTOR * median_edge_length(xy, edges)
spatial_neighbors = li.pp.spatial_neighbors if hasattr(li, "pp") and hasattr(li.pp, "spatial_neighbors") \
    else li.ut.spatial_neighbors          # older LIANA versions keep it in li.ut
spatial_neighbors(adata, bandwidth=bandwidth, cutoff=0.1, kernel="gaussian", set_diag=True)

adata.raw = adata    # older LIANA versions read .raw by default; this works for both
lrdata = li.mt.bivariate(
    adata,
    resource_name="consensus",
    local_name="cosine",
    global_name="morans",
    n_perms=100,
    nz_prop=C.MIN_EXPR_FRAC,     # every subunit expressed in >= 10% of spots
    mask_negatives=True,
    add_categories=False,
    verbose=True,
)

rank = lrdata.var.copy()
rank["pair"] = rank.index
rank = rank.sort_values("morans", ascending=False)
rank.to_csv(C.OUT_DIR / "lr_ranking.csv")
print(f"{len(rank)} pairs pass the {C.MIN_EXPR_FRAC:.0%} expression filter")

# 1) Known breast-cancer pairs from config.CURATED_PAIRS, if they pass the
#    expression filter. 2) Fill up with the best remaining significant pairs,
#    at most one pair per ligand AND per receptor, so the channels are not
#    several copies of the same signal (e.g. four HLA class II -> CD4 pairs).
lig_col = "ligand" if "ligand" in rank.columns else "ligand_complex"
rec_col = "receptor" if "receptor" in rank.columns else "receptor_complex"
pval_col = "morans_pvals" if "morans_pvals" in rank.columns else None
pool = rank if pval_col is None else rank[rank[pval_col] < 0.05]

picked, used_l, used_r = [], set(), set()
for lig, rec in C.CURATED_PAIRS:
    hit = rank[(rank[lig_col] == lig) & (rank[rec_col] == rec)]
    if hit.empty:
        print(f"  curated {lig}->{rec}: not in this sample (gene missing or below the expression filter)")
        continue
    picked.append(hit.index[0]); used_l.add(lig); used_r.add(rec)
for idx, row in pool.iterrows():
    if len(picked) >= C.N_PAIRS:
        break
    if row[lig_col] in used_l or row[rec_col] in used_r:
        continue
    picked.append(idx); used_l.add(row[lig_col]); used_r.add(row[rec_col])

chosen = rank.loc[picked]
out = chosen[[lig_col, rec_col, "morans"] + ([pval_col] if pval_col else [])].copy()
out.columns = ["ligand", "receptor", "morans"] + (["pval"] if pval_col else [])
out.insert(2, "source", ["curated" if (l, r) in set(C.CURATED_PAIRS) else "liana"
                         for l, r in zip(out.ligand, out.receptor)])
out.to_csv(C.OUT_DIR / "lr_pairs.csv", index=False)
print(out.to_string(index=False))
print("saved", C.OUT_DIR / "lr_pairs.csv")
