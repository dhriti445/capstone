"""Step 5 - shuffled-expression control (is the flow more than chance?).

For each L-R pair, shuffle which spot gets which expression profile
(same shuffle for ligand and receptor, so each spot keeps its own L/R
levels but the tissue layout is destroyed), recompute the product flow,
decompose it, and repeat N_PERMUTATIONS times. The real flow's energy in
each Hodge part (its SHARE of the total) is then reported as a z-score
against that null.

z > 2 : that part of the signal depends on where the spots are in the tissue.
z ~ 0 : it is no different from randomly placed spots.

Run:  python 05_shuffle_control.py
Saves outputs/shuffle_zscores.csv and outputs/figures/05_shuffle_zscores.png
(Uses product flows: rerunning COMMOT 100 times is too slow for a laptop.)
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from flows import load_complex, align_to_complex, preprocess, build_incidence, gene_values, product_flow
from hodge_quick import HodgeSolver
from plots import PART_COLORS

adata = sc.read_h5ad(C.H5AD_IN)
edges, triangles, spot_ids = load_complex(C.COMPLEX_DIR)
adata, _ = align_to_complex(adata, spot_ids)
adata = preprocess(adata, C.TARGET_SUM, C.SCALE_QUANTILE)
B1, B2 = build_incidence(adata.n_obs, edges, triangles)
solve = HodgeSolver(B1, B2)
pairs = pd.read_csv(C.OUT_DIR / "lr_pairs.csv")
rng = np.random.default_rng(C.SEED)
parts = ("gradient", "curl", "harmonic")


def energies(L, R):
    """Share of the flow's energy in gradient, curl and harmonic parts.

    Shares, not raw energy: shuffling spots always makes the flow noisier
    (bigger), so raw energy would look 'significant' for any smooth gene.
    """
    f = product_flow(L, R, edges)
    g, c, h = solve(f)
    return np.array([g @ g, c @ c, h @ h]) / max(f @ f, 1e-30)


rows = []
for lig, rec in zip(pairs.ligand, pairs.receptor):
    L, R = gene_values(adata, lig), gene_values(adata, rec)
    if L is None or R is None:
        continue
    obs = energies(L, R)
    null = np.array([energies(L[p], R[p]) for p in (rng.permutation(adata.n_obs) for _ in range(C.N_PERMUTATIONS))])
    z = (obs - null.mean(axis=0)) / (null.std(axis=0) + 1e-12)
    rows.append({"pair": f"{lig}->{rec}", **{f"z_{p}": z[k] for k, p in enumerate(parts)},
                 **{f"share_{p}": obs[k] for k, p in enumerate(parts)},
                 **{f"null_share_{p}": null[:, k].mean() for k, p in enumerate(parts)}})
    print(f"{lig}->{rec:25s} z: gradient {z[0]:6.1f}  curl {z[1]:6.1f}  harmonic {z[2]:6.1f}")

res = pd.DataFrame(rows)
res.to_csv(C.OUT_DIR / "shuffle_zscores.csv", index=False)

fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(res) + 2), 4.5))
x = np.arange(len(res))
for k, p in enumerate(parts):
    ax.bar(x + (k - 1) * 0.27, res[f"z_{p}"], width=0.27, color=PART_COLORS[p], label=p)
ax.axhline(2, color="grey", ls="--", lw=1)
ax.axhline(-2, color="grey", ls="--", lw=1)
ax.set_xticks(x, res["pair"], rotation=45, ha="right")
ax.set_ylabel("z-score vs shuffled spots")
ax.set_title(f"{C.SAMPLE}: Hodge energy share vs {C.N_PERMUTATIONS} spot shuffles")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "05_shuffle_zscores.png", dpi=200)
print("saved", C.OUT_DIR / "shuffle_zscores.csv")
