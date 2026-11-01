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


def build_visium_complex(array_row, array_col):
    """Return (edges, triangles, B1, B2) for spots with given array indices."""
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
