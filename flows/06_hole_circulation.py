"""Step 6 - circulation around each hole (the right way to look for harmonic flow).

Why: the harmonic space has only beta1 dimensions (8 on TENX53) out of
~14,000 edges, so the harmonic SHARE of total energy is always ~0% - even
for random flows. That number cannot tell you whether signal circles a duct.

What this does: walks around each hole boundary and adds up the flow along
the way (a line integral). The gradient part always adds up to 0 around a
loop, so a non-zero sum is circulating signal.

  circulation index = sum(flow along loop) / sum(|flow| on loop edges)
     +1  every edge on the loop pushes the same way round (anticlockwise in the image)
     -1  the same, clockwise
      0  no net circulation

For product flows, each index is compared with 100 spot shuffles (z-score).
COMMOT is too slow to rerun 100 times, so its index is reported without a z.

Run:  python 06_hole_circulation.py
Saves outputs/hole_circulation.csv and outputs/figures/06_hole_circulation.png
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from flows import load_complex, align_to_complex, preprocess, gene_values, product_flow
from plots import plot_holes
from holes import ordered_loops, orient_anticlockwise, circ_index, MIN_LOOP_EDGES


adata = sc.read_h5ad(C.H5AD_IN)
edges, triangles, spot_ids = load_complex(C.COMPLEX_DIR)
adata, _ = align_to_complex(adata, spot_ids)
xy = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
loops = orient_anticlockwise(ordered_loops(edges, triangles, adata.n_obs), xy)
print(f"{len(loops)} hole loops with >= {MIN_LOOP_EDGES} edges")

C.FIG_DIR.mkdir(parents=True, exist_ok=True)
rows = []
# --- COMMOT flows (no null)
cm = sc.read_h5ad(C.FLOWS_COMMOT)
cf, cnames = np.asarray(cm.uns["flows"], float), [str(n) for n in cm.uns["flow_names"]]
for k, name in enumerate(cnames):
    for h, L in enumerate(loops):
        rows.append({"kind": "commot", "channel": name, "hole": h + 1, "n_edges": len(L["edges"]),
                     "index": circ_index(cf[:, k], L), "z": np.nan})

# --- product flows with a shuffle null
adata = preprocess(adata, C.TARGET_SUM, C.SCALE_QUANTILE)
pairs = pd.read_csv(C.OUT_DIR / "lr_pairs.csv")
rng = np.random.default_rng(C.SEED)
perms = [rng.permutation(adata.n_obs) for _ in range(C.N_PERMUTATIONS)]
for lig, rec in zip(pairs.ligand, pairs.receptor):
    Lg, Rc = gene_values(adata, lig), gene_values(adata, rec)
    if Lg is None or Rc is None:
        continue
    f = product_flow(Lg, Rc, edges)
    null = np.array([[circ_index(product_flow(Lg[p], Rc[p], edges), L) for L in loops] for p in perms])
    for h, L in enumerate(loops):
        obs = circ_index(f, L)
        z = (obs - null[:, h].mean()) / (null[:, h].std() + 1e-12)
        rows.append({"kind": "product", "channel": f"{lig}->{rec}", "hole": h + 1,
                     "n_edges": len(L["edges"]), "index": obs, "z": z})

res = pd.DataFrame(rows)
res.to_csv(C.OUT_DIR / "hole_circulation.csv", index=False)
strong = res[(res["kind"] == "product") & (res["z"].abs() > 3)].sort_values("z", key=abs, ascending=False)
print("\nproduct-flow circulations with |z| > 3:")
print(strong.head(15).to_string(index=False) if len(strong) else "  none")
top_c = res[res["kind"] == "commot"].sort_values("index", key=abs, ascending=False)
print("\nstrongest COMMOT circulations (no null - treat as leads, not results):")
print(top_c.head(10).to_string(index=False))

# --- figure: loops on tissue + heatmap of indices
fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1.4]})
plot_holes(adata, edges, triangles, ax=axes[0], title="Hole loops (numbers match the heatmap)")
for t in list(axes[0].texts):
    t.remove()   # replace plot_holes' own numbering with labels that match the heatmap
for h, L in enumerate(loops):
    c = xy[L["nodes"]].mean(axis=0)
    axes[0].text(c[0], c[1], str(h + 1), color="black", fontsize=13, weight="bold", ha="center",
                 va="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
M = res.pivot_table(index="channel", columns="hole", values="index", sort=False)
im = axes[1].imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axes[1].set_yticks(range(len(M.index)), M.index, fontsize=7)
axes[1].set_xticks(range(len(M.columns)), [f"hole {c}" for c in M.columns])
axes[1].set_title("Circulation index (+1 anticlockwise, -1 clockwise)")
plt.colorbar(im, ax=axes[1], fraction=0.03)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "06_hole_circulation.png", dpi=180)
print("\nsaved", C.OUT_DIR / "hole_circulation.csv")
