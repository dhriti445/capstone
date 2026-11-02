"""Step 7 - COMMOT on spot-shuffled expression (null flows for Dhriti's shuffle test).

For each shuffle k: randomly reassign whole expression profiles to spots
(positions and the complex stay the same, so the tissue layout is destroyed),
run COMMOT with exactly the settings of step 3, and save the flows for the
SAME pathways in the SAME column order as the real COMMOT file.

Each shuffle is saved in the v0.1 layout:
  outputs/shuffles/<SAMPLE>_flows_commot_shuf<k>.h5ad
  (spots + positions only, no gene matrix) uns['complex'], uns['flows'] (E, C), uns['flow_names'], uns['flow_kind'] = 'commot_shuffle',
  uns['shuffle_seed'], uns['shuffle_perm'] (perm[i] = the spot whose expression spot i now has)

Run after step 3 (needs outputs/<SAMPLE>_flows_commot.h5ad):
  python 07_commot_shuffles.py          # shuffles 0-9
  python 07_commot_shuffles.py 10       # a different number of shuffles
Each shuffle takes about as long as one COMMOT run (10-30 min on a laptop),
so 10 shuffles take a few hours. Finished shuffles are skipped on a rerun,
so you can stop with Ctrl+C and continue later.
"""
import sys
import time

import anndata as ad
import numpy as np
import scanpy as sc

import config as C
from flows import load_complex, align_to_complex, preprocess, commot_flows, save_flows
from hodgecc.plots import median_edge_length

N_SHUFFLES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SHUF_DIR = C.OUT_DIR / "shuffles"
SHUF_DIR.mkdir(parents=True, exist_ok=True)

import commot as ct

# the real COMMOT file fixes the channel list and order
real = sc.read_h5ad(C.FLOWS_COMMOT)
names = [str(n) for n in real.uns["flow_names"]]
print(f"matching {len(names)} pathways of {C.FLOWS_COMMOT.name}")

adata = sc.read_h5ad(C.H5AD_IN)
edges, triangles, spot_ids = load_complex(C.COMPLEX_DIR)
adata, _ = align_to_complex(adata, spot_ids)
adata = preprocess(adata, C.TARGET_SUM, C.SCALE_QUANTILE)
assert np.array_equal(edges, np.asarray(real.uns["complex"]["edges"])), "complex differs from the real COMMOT file"

xy = np.asarray(adata.obsm["spatial"])[:, :2]
dis_thr = C.COMMOT_RANGE_FACTOR * median_edge_length(xy, edges)
df = ct.pp.ligand_receptor_database(species="human", signaling_type=C.COMMOT_SIGNALING_TYPE, database="CellChat")
df = ct.pp.filter_lr_database(df, adata, min_cell_pct=0.05)   # unchanged by shuffling (same genes, same counts)
print(f"COMMOT distance threshold {dis_thr:.1f}; {len(df)} L-R pairs")

for k in range(N_SHUFFLES):
    path = SHUF_DIR / f"{C.SAMPLE}_flows_commot_shuf{k}.h5ad"
    if path.exists():
        print(f"shuffle {k}: already done, skipping")
        continue
    seed = C.SEED + 1000 + k
    perm = np.random.default_rng(seed).permutation(adata.n_obs)
    ad_s = adata.copy()
    ad_s.X = adata.X[perm]                       # expression moves, positions stay
    ad_s.layers["counts"] = adata.layers["counts"][perm]
    t0 = time.time()
    ct.tl.spatial_communication(ad_s, database_name="cellchat", df_ligrec=df,
                                dis_thr=dis_thr, heteromeric=True, pathway_sum=True)
    flows, got, _ = commot_flows(ad_s, edges, names, database_name="cellchat")
    # a pathway with no signal after shuffling is absent from obsp: keep its column as zeros
    full = np.zeros((len(edges), len(names)))
    for j, n in enumerate(got):
        full[:, names.index(n)] = flows[:, j]
    # slim file: spots + positions + flows, no gene matrix (10 full copies would be ~2 GB);
    # the shuffled expression is adata.X[uns['shuffle_perm']] if anyone needs it
    out = ad.AnnData(obs=adata.obs[[]].copy())
    out.obsm["spatial"] = adata.obsm["spatial"]
    out.uns["shuffle_seed"] = seed
    out.uns["shuffle_perm"] = perm
    save_flows(out, edges, triangles, full, names, path, kind="commot_shuffle")
    print(f"shuffle {k}: done in {(time.time() - t0) / 60:.1f} min")

print("all shuffles in", SHUF_DIR)
