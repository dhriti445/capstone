"""A small fake Visium section for testing the code without downloaded data.

Hex grid with 100 um spacing (in pixels, like HEST), two round holes
(a 'duct' and a 'vessel'), a Delaunay complex built with the same rule as
the project (prune edges > 1.5x the median nearest-neighbour distance,
keep triangles whose three edges survive), and Poisson counts for a few genes.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree

PX_PER_UM = 1 / 0.368   # TENX53 pixel size is about 0.368 um per pixel


def build_complex(xy, factor=1.5):
    tri = Delaunay(xy).simplices
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    max_len = factor * np.median(nn)
    edges = set()
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
            a, b = sorted((int(a), int(b)))
            if np.linalg.norm(xy[a] - xy[b]) <= max_len:
                edges.add((a, b))
    triangles = []
    for t in tri:
        a, b, c = sorted(int(v) for v in t)
        if (a, b) in edges and (b, c) in edges and (a, c) in edges:
            triangles.append((a, b, c))
    edges = np.array(sorted(edges), dtype=np.int64)
    triangles = np.array(sorted(triangles), dtype=np.int64)
    return edges, triangles


def make_tissue(n_cols=30, n_rows=26, seed=0):
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    step = 100 * PX_PER_UM
    pts = []
    for r in range(n_rows):
        for c in range(n_cols):
            pts.append(((c + 0.5 * (r % 2)) * step, r * step * np.sqrt(3) / 2))
    xy = np.array(pts) + 1000
    w, h = xy.max(axis=0) - xy.min(axis=0)
    holes = [(xy.min(axis=0) + [0.3 * w, 0.5 * h], 2.6 * step), (xy.min(axis=0) + [0.72 * w, 0.45 * h], 2.2 * step)]
    keep = np.ones(len(xy), bool)
    for centre, rad in holes:
        keep &= np.linalg.norm(xy - centre, axis=1) > rad
    xy = xy[keep]
    edges, triangles = build_complex(xy)

    # expression: ligand high on the left (a source), receptors high on the right
    x01 = (xy[:, 0] - xy[:, 0].min()) / np.ptp(xy[:, 0])
    rate = {
        "LIGA": 20 * (1 - x01) + 0.5,
        "RECA": 20 * x01 + 0.5,
        "RECB": 15 * x01 + 1.0,
    }
    for k in range(20):   # background genes
        rate[f"GENE{k}"] = rng.uniform(1, 10) * np.ones(len(xy))
    genes = list(rate)
    counts = np.column_stack([rng.poisson(rate[g]) for g in genes]).astype(np.float32)
    counts[:, -1] += rng.poisson(200, len(xy))   # pad library size
    obs = pd.DataFrame(index=[f"SPOT{i:05d}-1" for i in range(len(xy))])
    adata = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = xy
    adata.uns["demo_holes"] = np.array([c for c, _ in holes])
    return adata, edges, triangles


def make_demo_tissue(seed=0):
    """Fake section with real gene names and three planted patterns:

    TGFB1 -> TGFBR1_TGFBR2   one-way: ligand on the left, receptor on the right (gradient)
    SPP1  -> CD44            expression rotates around the left hole (harmonic)
    CXCL12 -> CXCR4          expression rotates around a point in solid tissue (curl)
    """
    import anndata as ad
    import pandas as pd

    base, edges, triangles = make_tissue(seed=seed)
    rng = np.random.default_rng(seed + 1)
    xy = base.obsm["spatial"]
    step = 100 * PX_PER_UM
    duct = base.uns["demo_holes"][0]
    x01 = (xy[:, 0] - xy[:, 0].min()) / np.ptp(xy[:, 0])
    vortex = np.array([xy[:, 0].min() + 0.55 * np.ptp(xy[:, 0]), xy[:, 1].min() + 0.15 * np.ptp(xy[:, 1])])

    def ring(centre, radius):
        d = xy - centre
        theta = np.arctan2(d[:, 1], d[:, 0])
        near = np.exp(-(np.linalg.norm(d, axis=1) / radius) ** 2)
        return near, theta

    near_d, th_d = ring(duct, 6 * step)
    near_v, th_v = ring(vortex, 3 * step)
    rate = {
        "TGFB1": 12 * (1 - x01) + 0.5, "TGFBR1": 12 * x01 + 0.5, "TGFBR2": 10 * x01 + 1.0,
        "SPP1": 1 + 10 * near_d * (1 + np.cos(th_d)), "CD44": 1 + 10 * near_d * (1 + np.sin(th_d)),
        "CXCL12": 1 + 10 * near_v * (1 + np.cos(th_v)), "CXCR4": 1 + 10 * near_v * (1 + np.sin(th_v)),
    }
    for k in range(30):
        rate[f"BG{k}"] = rng.uniform(1, 10) * np.ones(len(xy))
    genes = list(rate)
    counts = np.column_stack([rng.poisson(rate[g]) for g in genes]).astype(np.float32)
    counts[:, -1] += rng.poisson(300, len(xy))
    adata = ad.AnnData(X=counts, obs=base.obs.copy(), var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = xy
    return adata, edges, triangles
