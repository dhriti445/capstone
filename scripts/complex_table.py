"""Table of every saved cell complex: spots, edges, triangles, components, holes.

Usage:  python scripts/complex_table.py [folder]      (default: data/complex)

Reads the *_complex.npz files written by scripts/real_slice_complex.py.
For a planar 2-complex there are no enclosed voids, so
    holes (b1) = components (b0) - (N - E + T)
which is the Euler identity we already check. Writes results/complex_table.csv.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def main(folder):
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*_complex.npz"))):
        z = np.load(path, allow_pickle=False)
        edges, tris = z["edges"], z["triangles"]
        n = len(z["spot_ids"])
        A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        b0 = connected_components(A, directed=False)[0]
        chi = n - len(edges) + len(tris)
        rows.append({"sample": str(z["sample"]), "spots": n, "edges": len(edges),
                     "triangles": len(tris), "components_b0": b0,
                     "holes_b1": b0 - chi, "euler": chi})
    if not rows:
        sys.exit(f"no *_complex.npz files in {folder}")
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/complex_table.csv", index=False)
    print(df.to_string(index=False))
    print("\nsaved results/complex_table.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "complex"))
