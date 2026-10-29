"""Cell-complex builder: points -> graph -> triangles -> incidence matrices.

Conventions (v0.1):
  edges     (E, 2) int, i < j, lexicographically sorted
  triangles (T, 3) int, a < b < c, lexicographically sorted
  B1        (N, E) CSR: edge (i, j) -> -1 at i, +1 at j
  B2        (E, T) CSR: triangle (a, b, c) -> +(a,b) +(b,c) -(a,c)
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay


def canonical_edges(edges):
    """Sort each edge so i < j, drop duplicates/self-loops, sort rows."""
    e = np.sort(np.asarray(edges, dtype=np.int64), axis=1)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(e, axis=0)  # unique also sorts lexicographically
    return e


def triangles_from_edges(edges, n_nodes):
    """All 3-cliques of the graph, as sorted (a<b<c) rows.

    Only correct when every 3-clique is a real face (e.g. Visium grid).
    """
    edges = canonical_edges(edges)
    A = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    A = A + A.T
    tris = []
    for a, b in edges:
        common = np.intersect1d(A[a].indices, A[b].indices)
        for c in common[common > b]:
            tris.append((a, b, c))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(np.array(tris, dtype=np.int64), axis=0)


def build_incidence(edges, triangles, n_nodes):
    """Return B1 (N x E) and B2 (E x T) as CSR matrices."""
    edges = np.asarray(edges, dtype=np.int64)
    triangles = np.asarray(triangles, dtype=np.int64)
    E, T = len(edges), len(triangles)

    cols = np.arange(E)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    vals = np.concatenate([-np.ones(E), np.ones(E)])
    B1 = sp.coo_matrix(
        (vals, (rows, np.concatenate([cols, cols]))), shape=(n_nodes, E)
    ).tocsr()

    eidx = {(int(i), int(j)): k for k, (i, j) in enumerate(edges)}
    r, c, v = [], [], []
    for t, (a, b, cc) in enumerate(triangles):
        for (u, w), s in (((a, b), 1.0), ((b, cc), 1.0), ((a, cc), -1.0)):
            r.append(eidx[(int(u), int(w))])
            c.append(t)
            v.append(s)
    B2 = sp.coo_matrix((v, (r, c)), shape=(E, T)).tocsr()
    return B1, B2


def check_boundary(B1, B2, tol=1e-12):
    """B1 @ B2 must be exactly zero."""
    if B2.shape[1] == 0:
        return True
    return abs(B1 @ B2).max() <= tol


def build_complex(coords, method="delaunay", max_edge_quantile=0.95,
                  n_neighs=6, adata=None):
    """Build (edges, triangles, B1, B2) from (N, 2) coordinates.

    method="delaunay": irregular cells. Edges longer than the
        `max_edge_quantile` quantile of edge lengths are pruned, and any
        triangle with a pruned edge is dropped.
    method="grid": Visium hex grid. Needs `adata` with obsm['spatial'];
        uses squidpy and then takes all 3-cliques.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)

    if method == "delaunay":
        simp = np.sort(Delaunay(coords).simplices, axis=1)
        e = canonical_edges(
            np.vstack([simp[:, [0, 1]], simp[:, [1, 2]], simp[:, [0, 2]]])
        )

        def elen(a, b):
            return np.linalg.norm(coords[a] - coords[b], axis=1)

        thr = np.quantile(elen(e[:, 0], e[:, 1]), max_edge_quantile)
        edges = e[elen(e[:, 0], e[:, 1]) <= thr]

        ok = (
            (elen(simp[:, 0], simp[:, 1]) <= thr)
            & (elen(simp[:, 1], simp[:, 2]) <= thr)
            & (elen(simp[:, 0], simp[:, 2]) <= thr)
        )
        triangles = np.unique(simp[ok], axis=0)
    elif method == "grid":
        import squidpy as sq
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=n_neighs)
        A = sp.triu(adata.obsp["spatial_connectivities"], k=1).tocoo()
        edges = canonical_edges(np.column_stack([A.row, A.col]))
        triangles = triangles_from_edges(edges, n)
    else:
        raise ValueError(method)

    B1, B2 = build_incidence(edges, triangles, n)
    assert check_boundary(B1, B2), "B1 @ B2 != 0: orientation bug"
    return edges, triangles, B1, B2