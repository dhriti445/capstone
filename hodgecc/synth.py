"""Toy complexes and planted flows for validating the decomposition."""
import numpy as np
from hodgecc.complex import canonical_edges, build_incidence
from hodgecc.hodge import harmonic_basis_dense


def grid_complex(nx, ny, holes=()):
    """Triangulated nx-by-ny grid. holes: list of (cx, cy, r) in grid units.

    Triangles whose centroid lies inside a hole disk are removed, and
    edges are taken only from the remaining triangles.
    Returns coords, edges, triangles, B1, B2.
    """
    coords = np.array([(i, j) for i in range(nx) for j in range(ny)], float)

    def idx(i, j):
        return i * ny + j

    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i, j + 1), idx(i + 1, j + 1)
            tris.append(sorted((a, b, d)))
            tris.append(sorted((a, c, d)))
    tris = np.array(tris, dtype=np.int64)
    keep = np.ones(len(tris), bool)
    cent = coords[tris].mean(axis=1)
    for cx, cy, r in holes:
        keep &= np.hypot(cent[:, 0] - cx, cent[:, 1] - cy) > r
    tris = np.unique(tris[keep], axis=0)
    pairs = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]])
    edges = canonical_edges(pairs)
    B1, B2 = build_incidence(edges, tris, len(coords))
    return coords, edges, tris, B1, B2


def planted_flows(B1, B2, seed=0, harm_scale=1.0):
    """Return (f, g, c, h): a mixed flow and its three planted parts."""
    rng = np.random.default_rng(seed)
    N, E = B1.shape
    T = B2.shape[1]
    g = B1.T @ rng.normal(size=N)
    c = B2 @ rng.normal(size=T) if T > 0 else np.zeros(E)
    H = harmonic_basis_dense(B1, B2)
    h = H @ rng.normal(size=H.shape[1]) * harm_scale if H.shape[1] else np.zeros(E)
    return g + c + h, g, c, h
