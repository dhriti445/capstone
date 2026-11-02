"""Run the Hodge decomposition on a teammate's flow file and compare it with
the gradient/curl/harmonic split they reported.

Usage:
  python scripts/check_flows.py flows.h5ad energy.csv [complex.npz]

flows.h5ad must have uns['complex'] (edges, triangles) and uns['flows'] (E, C),
and may have uns['flow_names']. energy.csv needs one row per channel with
columns containing 'grad', 'curl' and 'harm' (fractions or percents).
complex.npz (optional) is your own saved complex; if given, the edges and
triangles in the flow file must be identical to it.
"""
import sys
import numpy as np
import pandas as pd
import anndata as ad
from hodgecc.complex import build_incidence, check_boundary
from hodgecc.hodge import decompose, energy_fractions


def find_col(cols, key):
    for c in cols:
        if key in str(c).lower():
            return c
    return None


def main(h5, csv, npz=None):
    a = ad.read_h5ad(h5)
    cx = a.uns["complex"]
    edges = np.asarray(cx["edges"], dtype=np.int64)
    tris = np.asarray(cx["triangles"], dtype=np.int64)
    F = np.asarray(a.uns["flows"], dtype=float)
    if "flow_names" in a.uns:
        names = [str(x) for x in a.uns["flow_names"]]
    else:
        names = [f"ch{k}" for k in range(F.shape[1])]
    n = a.n_obs
    print(f"file: {h5}")
    print(f"spots {n}, edges {len(edges)}, triangles {len(tris)}, "
          f"channels {F.shape[1]}, flows shape {F.shape}")
    assert F.shape[0] == len(edges), "flows rows != number of edges"
    assert len(names) == F.shape[1], "flow_names length != channels"

    if npz:
        z = np.load(npz, allow_pickle=False)
        same = (np.array_equal(z["edges"], edges)
                and np.array_equal(z["triangles"], tris))
        print("edges/triangles identical to my complex:", same)
        assert same, "complex in the flow file differs from mine"

    B1, B2 = build_incidence(edges, tris, n)
    print("B1 @ B2 == 0:", check_boundary(B1, B2))
    g, c, h = decompose(B1, B2, F)
    fr = energy_fractions(g, c, h)
    mine = pd.DataFrame({"gradient": fr[0], "curl": fr[1], "harmonic": fr[2]},
                        index=names)

    t = pd.read_csv(csv)
    print("\nyour CSV columns:", list(t.columns))
    name_col = None
    for key in ("channel", "pair", "name", "pathway", "lr"):
        name_col = find_col(t.columns, key)
        if name_col:
            break
    if name_col is None and not np.issubdtype(t.iloc[:, 0].dtype, np.number):
        name_col = t.columns[0]
    cols = {k: find_col(t.columns, s) for k, s in
            (("gradient", "grad"), ("curl", "curl"), ("harmonic", "harm"))}
    if None in cols.values():
        print("could not find gradient/curl/harmonic columns:", cols)
        return
    theirs = t.set_index(name_col)[list(cols.values())] if name_col else \
        t[list(cols.values())].set_axis(names[:len(t)])
    theirs.columns = ["gradient", "curl", "harmonic"]
    theirs.index = [str(x).strip() for x in theirs.index]
    if theirs.to_numpy().max() > 1.5:        # given as percents
        theirs = theirs / 100.0

    common = [x for x in mine.index if x in theirs.index]
    print(f"channels matched by name: {len(common)} of {len(mine)}")
    if not common:
        print("no names matched; first names mine:", list(mine.index[:3]),
              "theirs:", list(theirs.index[:3]))
        return
    d = (mine.loc[common] - theirs.loc[common]).abs()
    print("\nmax absolute difference in energy fraction:")
    print(d.max().to_string(float_format=lambda v: f"{v:.2e}"))
    worst = d.max(axis=1).sort_values(ascending=False).head(5)
    print("\nlargest differences by channel:")
    print(worst.to_string(float_format=lambda v: f"{v:.2e}"))
    print("\nmy split (%), first rows:")
    print((100 * mine.loc[common]).round(2).head(10).to_string())

    gb = (n - 1) / len(edges)
    cb = B2.shape[1] / len(edges)
    print(f"\nrandom-flow baseline: gradient {100*gb:.1f}%, curl {100*cb:.1f}%, "
          f"harmonic {100*(1-gb-cb):.3f}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
