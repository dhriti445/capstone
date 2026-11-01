"""Hole loops of a complex and the circulation of a flow around them.

The gradient part of any flow adds up to 0 around a closed loop, so a
non-zero sum around a hole is signal that circulates (harmonic or curl).
"""
import numpy as np

from hodgecc.plots import boundary_loops

MIN_LOOP_EDGES = 4


def ordered_loops(edges, triangles, n_nodes):
    """Hole boundaries as ordered node cycles, with each edge's travel sign."""
    bnd, loop, outer = boundary_loops(edges, triangles, n_nodes)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(edges)}
    loops = []
    for lab in np.unique(loop):
        if lab == outer:
            continue
        sel = edges[bnd[loop == lab]]
        if len(sel) < MIN_LOOP_EDGES:
            continue
        nbrs = {}
        for i, j in sel:
            nbrs.setdefault(int(i), []).append(int(j))
            nbrs.setdefault(int(j), []).append(int(i))
        if any(len(v) != 2 for v in nbrs.values()):
            print(f"  skip a hole boundary with {len(sel)} edges: it is not a simple loop "
                  "(touches itself or the tissue edge)")
            continue
        start = min(nbrs)
        path, prev, cur = [start], None, start
        while True:
            nxt = [v for v in nbrs[cur] if v != prev][0]
            if nxt == start:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        cyc = path + [start]
        eidx = np.array([index[(min(a, b), max(a, b))] for a, b in zip(cyc[:-1], cyc[1:])])
        sign = np.array([1.0 if a < b else -1.0 for a, b in zip(cyc[:-1], cyc[1:])])
        loops.append({"nodes": np.array(path), "edges": eidx, "sign": sign})
    return loops


def orient_anticlockwise(loops, xy):
    """Flip loops so +1 always means anticlockwise as seen in the image."""
    for L in loops:
        p = xy[L["nodes"]]
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        # image y points down, so a positive shoelace area is clockwise on screen
        if area > 0:
            L["sign"] = -L["sign"]
    return loops


def circ_index(f, L):
    num = float(np.sum(L["sign"] * f[L["edges"]]))
    den = float(np.sum(np.abs(f[L["edges"]])))
    return num / den if den > 0 else 0.0
