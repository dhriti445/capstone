"""Build a complex from a Visium .h5ad and print its statistics.

Usage:  python scripts/check_visium.py [path/to/slice.h5ad]
With no path it downloads a small built-in squidpy Visium slice.
"""
import sys
import gudhi
import scanpy as sc
import squidpy as sq
from hodgecc.complex import build_complex


def main():
    if len(sys.argv) > 1:
        adata = sc.read_h5ad(sys.argv[1])
    else:
        adata = sq.datasets.visium_hne_adata()
    coords = adata.obsm["spatial"]
    edges, tris, B1, B2 = build_complex(coords, method="grid", adata=adata)
    n = adata.n_obs
    print("spots:", n, " edges:", len(edges), " triangles:", len(tris))
    print("Euler N-E+T:", n - len(edges) + len(tris))

    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i])
    for a in edges:
        st.insert([int(a[0]), int(a[1])])
    for t in tris:
        st.insert([int(x) for x in t])
    st.compute_persistence()
    print("Betti numbers (b0, b1):", st.betti_numbers())


if __name__ == "__main__":
    main()
