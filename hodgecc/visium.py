"""Exact Visium hexagonal-lattice complex from array_row / array_col.

Visium spots sit on a staggered lattice: a spot at (row, col) has six
neighbours at (row, col+-2) and (row+-1, col+-1). Building the graph from
these integer indices needs no distance threshold and no squidpy call.
Every 3-clique of this graph is a real triangular face of the lattice.
"""
import numpy as np
from hodgecc.complex import (canonical_edges, triangles_from_edges,
                             build_incidence, check_boundary)

# one half of the six neighbour offsets; the other half are their mirrors
FORWARD_OFFSETS = [(0, 2), (1, 1), (1, -1)]


def build_visium_complex(array_row, array_col, fill_max_missing=0):
    """Return (edges, triangles, B1, B2) for spots with given array indices.

    fill_max_missing > 0 caps every enclosed gap of at most that many
    missing spots with one extra 2-cell, so dropped spots no longer count
    as holes. Edges and triangles are unchanged, so edge-flow arrays stay
    valid; only B2 gets extra columns (B2 then has more columns than
    len(triangles)).
    """
    rows = np.asarray(array_row, dtype=int)
    cols = np.asarray(array_col, dtype=int)
    n = len(rows)
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    if len(lookup) != n:
        raise ValueError("duplicate (array_row, array_col) pairs")
    pairs = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in FORWARD_OFFSETS:
            j = lookup.get((int(r) + dr, int(c) + dc))
            if j is not None:
                pairs.append((i, j))
    edges = canonical_edges(pairs)
    triangles = triangles_from_edges(edges, n)
    B1, B2 = build_incidence(edges, triangles, n)
    if fill_max_missing > 0:
        import scipy.sparse as sp
        loops = gap_loops(rows, cols, edges, fill_max_missing)
        if loops:
            r_, c_, v_ = [], [], []
            for k, (idx, sgn) in enumerate(loops):
                r_ += idx
                c_ += [k] * len(idx)
                v_ += sgn
            extra = sp.coo_matrix((v_, (r_, c_)),
                                  shape=(len(edges), len(loops))).tocsr()
            B2 = sp.hstack([B2, extra]).tocsr()
    assert check_boundary(B1, B2), "B1 @ B2 != 0: orientation bug"
    return edges, triangles, B1, B2


def enclosed_gaps(array_row, array_col):
    """Find holes in the spot lattice: groups of missing lattice sites that
    are completely surrounded by present spots.

    Returns a list of lists of (row, col) missing sites, largest first.
    Each group is one hole of the complex (beta1 counts them).
    """
    from collections import deque
    rows = np.asarray(array_row, dtype=int)
    cols = np.asarray(array_col, dtype=int)
    present = set(zip(rows.tolist(), cols.tolist()))
    r0, r1 = rows.min() - 1, rows.max() + 1
    c0, c1 = cols.min() - 2, cols.max() + 2
    sites = {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)
             if (i % 2) == (j % 2)}
    absent = sites - present
    nbrs = [(0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen, comps = set(), []
    for s in absent:
        if s in seen:
            continue
        q, comp = deque([s]), [s]
        seen.add(s)
        while q:
            u = q.popleft()
            for dr, dc in nbrs:
                v = (u[0] + dr, u[1] + dc)
                if v in absent and v not in seen:
                    seen.add(v)
                    q.append(v)
                    comp.append(v)
        comps.append(comp)

    def touches_border(comp):
        return any(x[0] in (r0, r1) or x[1] in (c0, c0 + 1, c1 - 1, c1)
                   for x in comp)

    inner = [cp for cp in comps if not touches_border(cp)]
    return sorted(inner, key=len, reverse=True)


# neighbour offsets in cyclic order around a lattice site
RING = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]


def gap_loops(array_row, array_col, edges, max_missing=2):
    """Boundary loops of small enclosed gaps, as signed edge chains.

    A gap is a group of missing lattice sites fully surrounded by present
    spots. Its boundary is a closed loop of existing edges. Returns a list of
    (edge_indices, signs) pairs, one per gap with at most `max_missing`
    missing sites. Each pair is one extra 2-cell (polygon) column for B2:
    it fills the hole without adding or changing any edge.
    """
    rows = np.asarray(array_row, dtype=int)
    cols = np.asarray(array_col, dtype=int)
    node = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    eidx = {(int(a), int(b)): k for k, (a, b) in enumerate(edges)}
    out = []
    for gap in enclosed_gaps(rows, cols):
        if len(gap) > max_missing:
            continue
        count = {}
        for (r, c) in gap:
            ring = [(r + dr, c + dc) for dr, dc in RING]
            for k in range(6):
                tri = (( r, c), ring[k], ring[(k + 1) % 6])
                for u, v in ((tri[0], tri[1]), (tri[1], tri[2]),
                             (tri[0], tri[2])):
                    key = frozenset((u, v))
                    count[key] = count.get(key, 0) + 1
        bnd = [tuple(k) for k, n in count.items()
               if n == 1 and all(x in node for x in k)]
        nbrs = {}
        for u, v in bnd:
            nbrs.setdefault(u, []).append(v)
            nbrs.setdefault(v, []).append(u)
        if not bnd or any(len(x) != 2 for x in nbrs.values()):
            continue                      # not a simple loop: leave it alone
        start = bnd[0][0]
        loop, prev, cur = [start], None, start
        while True:
            a, b = nbrs[cur]
            nxt = b if a == prev else a
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) != len(nbrs):
            continue
        idx, sgn = [], []
        for k in range(len(loop)):
            i = node[loop[k]]
            j = node[loop[(k + 1) % len(loop)]]
            lo, hi = (i, j) if i < j else (j, i)
            idx.append(eidx[(lo, hi)])
            sgn.append(1.0 if i < j else -1.0)
        out.append((idx, sgn))
    return out
