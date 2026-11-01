"""Step 4 - decompose the flows and draw the figures.

Run:  python 04_figures.py commot     (or: product)
Saves, in outputs/figures/:
  04_<kind>_energy.png              stacked bars: gradient / curl / harmonic share per channel
  04_<kind>_<channel>_hodge.png     four panels for each channel
and outputs/<kind>_energy.csv with the numbers.
"""
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")

import config as C
from flows import build_incidence
from hodge_quick import decompose, energy_shares, check_decomposition
from plots import plot_hodge_panels, plot_energy_bars

kind = sys.argv[1] if len(sys.argv) > 1 else "commot"
path = C.FLOWS_COMMOT if kind == "commot" else C.FLOWS_PRODUCT
C.FIG_DIR.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(path)
edges = np.asarray(adata.uns["complex"]["edges"])
triangles = np.asarray(adata.uns["complex"]["triangles"])
flows = np.asarray(adata.uns["flows"], dtype=np.float64)
names = [str(n) for n in adata.uns["flow_names"]]
B1, B2 = build_incidence(adata.n_obs, edges, triangles)

rows, shares = [], []
for k, name in enumerate(names):
    f = flows[:, k]
    g, c, h = decompose(f, B1, B2)
    errs = check_decomposition(f, g, c, h, B1, B2)
    if max(errs.values()) > 1e-4:
        print(f"!! {name}: decomposition checks are loose: {errs}")
    s = energy_shares(f, g, c, h)
    shares.append(s)
    rows.append({"channel": name, "energy": float(f @ f), **s})
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    fig = plot_hodge_panels(adata, edges, f, g, c, h, title=f"{C.SAMPLE} · {name} ({kind} flow)",
                            path=C.FIG_DIR / f"04_{kind}_{safe}_hodge.png")
    matplotlib.pyplot.close(fig)
    print(f"{name:30s} gradient {s['gradient']:.0%}  curl {s['curl']:.0%}  harmonic {s['harmonic']:.0%}")

pd.DataFrame(rows).to_csv(C.OUT_DIR / f"{kind}_energy.csv", index=False)
plot_energy_bars(names, shares, title=f"{C.SAMPLE}: Hodge energy split ({kind} flows)",
                 path=C.FIG_DIR / f"04_{kind}_energy.png")
print("figures in", C.FIG_DIR)
