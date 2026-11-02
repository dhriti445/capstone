"""Step 0 (optional) - make a fake tissue to test every script before the real data.

Run:  python 00_make_demo_data.py
Then run any step in demo mode, e.g.  P2_DEMO=1 python 03_make_flows.py
(Windows PowerShell:  $env:P2_DEMO="1"; python 03_make_flows.py)

The fake section has three planted patterns, so you know what the figures
SHOULD show: TGFb = gradient, SPP1 = harmonic around the left hole,
CXCL = curl (a local swirl).
"""
import os
os.environ["P2_DEMO"] = "1"

import numpy as np
import config as C
from synthetic import make_demo_tissue

adata, edges, triangles = make_demo_tissue(seed=C.SEED)
C.COMPLEX_DIR.mkdir(parents=True, exist_ok=True)
np.save(C.COMPLEX_DIR / "edges.npy", edges)
np.save(C.COMPLEX_DIR / "triangles.npy", triangles)
np.save(C.COMPLEX_DIR / "spot_ids.npy", np.array(adata.obs_names, dtype=str))
adata.write_h5ad(C.H5AD_IN)
print(f"demo tissue: {adata.n_obs} spots, {len(edges)} edges, {len(triangles)} triangles -> {C.H5AD_IN.parent}")

# a ready-made pair list so you can skip LIANA in the demo
import pandas as pd
pd.DataFrame({"ligand": ["TGFB1", "SPP1", "CXCL12"],
              "receptor": ["TGFBR1_TGFBR2", "CD44", "CXCR4"]}).to_csv(C.OUT_DIR / "lr_pairs.csv", index=False)
print("wrote", C.OUT_DIR / "lr_pairs.csv")
