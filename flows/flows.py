"""Turn ligand-receptor expression into signed edge flows (data format v0.1).

Format reminder (agreed with Dhriti, v0.1):
  edges     (E, 2) int, row (i, j) with i < j, sorted lexicographically
  triangles (T, 3) int, a < b < c, sorted lexicographically
  B1        (N, E): column for edge (i, j) has -1 at row i, +1 at row j
  B2        (E, T): triangle (a, b, c) -> +(a,b) +(b,c) -(a,c)
  flows     (E, C) float, positive = signal goes i -> j
  Saved in one .h5ad: uns['complex'] = {edges, triangles}, uns['flows'] = (E, C)
  Extra (to agree with Dhriti): uns['flow_names'] = list of C channel names
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Complex
# ---------------------------------------------------------------------------
def load_complex(complex_dir):
    """Load Dhriti's complex files. Returns edges, triangles, spot_ids."""
    from pathlib import Path
    d = Path(complex_dir)
    edges = np.load(d / "edges.npy").astype(np.int64)
    triangles = np.load(d / "triangles.npy").astype(np.int64)
    spot_ids = np.load(d / "spot_ids.npy", allow_pickle=True).astype(str)
    return edges, triangles, spot_ids


def build_incidence(n_nodes, edges, triangles):
    """B1 (N, E) and B2 (E, T) in the v0.1 sign convention.

    Same output as hodgecc.complex.build_incidence; swap to Dhriti's version
    once her package is installed so there is only one implementation.
    """
    E = len(edges)
    cols = np.arange(E)
    B1 = sp.csr_matrix(
        (np.r_[-np.ones(E), np.ones(E)], (np.r_[edges[:, 0], edges[:, 1]], np.r_[cols, cols])),
        shape=(n_nodes, E),
    )
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(edges)}
    rows, tcols, vals = [], [], []
    for t, (a, b, c) in enumerate(triangles):
        for (u, v), sign in (((a, b), 1.0), ((b, c), 1.0), ((a, c), -1.0)):
            rows.append(index[(int(u), int(v))])
            tcols.append(t)
            vals.append(sign)
    B2 = sp.csr_matrix((vals, (rows, tcols)), shape=(E, len(triangles)))
    assert abs(B1 @ B2).max() == 0, "B1 @ B2 != 0: complex is inconsistent"
    return B1, B2


def align_to_complex(adata, spot_ids):
    """Reorder (and subset) adata so that row k is spot k of the complex."""
    missing = set(spot_ids) - set(adata.obs_names)
    if missing:
        raise ValueError(f"{len(missing)} complex spots are not in the .h5ad, e.g. {list(missing)[:3]}")
    dropped = adata.n_obs - len(spot_ids)
    adata = adata[list(spot_ids)].copy()
    return adata, dropped


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------
def preprocess(adata, target_sum=1e4, quantile=0.99, min_spots=10):
    """Counts -> drop rare genes -> normalise -> log1p, in place on adata.X (kept sparse).

    Raw counts are kept in layers['counts']. The 0-1 scaling used for flows
    is done per gene on demand (see scaled_gene), so no dense copy of the
    whole matrix is ever made.
    """
    import scanpy as sc
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=min_spots)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.uns["scale_quantile"] = quantile
    return adata


def scaled_gene(adata, gene):
    """One gene's log-normalised values scaled to 0-1 by its 99th percentile."""
    x = adata[:, gene].X
    x = np.asarray(x.toarray() if sp.issparse(x) else x).ravel().astype(np.float64)
    q = np.quantile(x, adata.uns.get("scale_quantile", 0.99))
    if q <= 0:
        q = x.max() if x.max() > 0 else 1.0
    return np.clip(x / q, 0.0, 1.0)


def gene_values(adata, name):
    """Per-spot scaled value of a gene or a multi-subunit complex.

    Complexes are written with '_' (e.g. 'ITGA1_ITGB1'), as in LIANA's
    consensus resource and CellChatDB. A complex takes the MINIMUM over
    its subunits. Returns None if any subunit is missing from the data.
    """
    subunits = name.split("_")
    if any(g not in adata.var_names for g in subunits):
        return None
    return np.min(np.column_stack([scaled_gene(adata, g) for g in subunits]), axis=1)


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------
def product_flow(L, R, edges):
    """Baseline flow for one L-R pair: f(i->j) = L(i)R(j) - L(j)R(i).

    Antisymmetric by construction, so it is a valid oriented edge flow.
    """
    i, j = edges[:, 0], edges[:, 1]
    return L[i] * R[j] - L[j] * R[i]


def product_flows(adata, edges, pairs):
    """Stack product flows for a list of (ligand, receptor) names -> (E, C)."""
    cols, names = [], []
    for lig, rec in pairs:
        L, R = gene_values(adata, lig), gene_values(adata, rec)
        if L is None or R is None:
            print(f"  skip {lig}->{rec}: a subunit is missing from the gene list")
            continue
        cols.append(product_flow(L, R, edges))
        names.append(f"{lig}->{rec}")
    if not cols:
        raise ValueError("No usable L-R pairs")
    return np.column_stack(cols).astype(np.float64), names


def matrix_to_edge_flow(P, edges):
    """Sender x receiver matrix (e.g. a COMMOT obsp entry) -> edge flow.

    f(i->j) = P[i, j] - P[j, i]. Signal between spots that are not joined by
    an edge of the complex is dropped (it has nowhere to live on the complex).
    """
    P = sp.csr_matrix(P)
    i, j = edges[:, 0], edges[:, 1]
    fwd = np.asarray(P[i, j]).ravel()
    bwd = np.asarray(P[j, i]).ravel()
    # share of the spot-to-other-spot signal that lands on complex edges
    # (a spot signalling to itself has no edge, so the diagonal is left out)
    between = P.sum() - P.diagonal().sum()
    kept = (fwd.sum() + bwd.sum()) / between if between > 0 else 0.0
    return fwd - bwd, kept


def commot_flows(adata, edges, labels, database_name="cellchat"):
    """Read COMMOT results from adata.obsp and convert each to an edge flow.

    labels: the names to read, e.g. pathway names ['TGFb', 'MHC-II'] or pair
    names ['TGFB1-TGFBR1_TGFBR2']. COMMOT stores each one in
    adata.obsp['commot-<database_name>-<label>'].
    Run commot.tl.spatial_communication first (see 03_make_flows.py).
    """
    prefix = f"commot-{database_name}-"
    cols, names, kept = [], [], []
    for label in labels:
        k = prefix + label
        if k not in adata.obsp:
            print(f"  skip {label}: not in adata.obsp")
            continue
        f, frac = matrix_to_edge_flow(adata.obsp[k], edges)
        if not np.any(f):
            continue
        cols.append(f)
        names.append(label)
        kept.append(frac)
    if not cols:
        raise ValueError("No COMMOT results found in adata.obsp - did spatial_communication run?")
    return np.column_stack(cols), names, np.array(kept)


# ---------------------------------------------------------------------------
# Saving and checks
# ---------------------------------------------------------------------------
def check_flows(flows, edges):
    """The checks Dhriti's code relies on. Raises if anything is off."""
    assert flows.ndim == 2 and flows.shape[0] == len(edges), f"flows must be (E, C), got {flows.shape}"
    assert np.isfinite(flows).all(), "flows contain NaN or inf"
    assert (edges[:, 0] < edges[:, 1]).all(), "edges must satisfy i < j"


def save_flows(adata, edges, triangles, flows, names, path, kind):
    """Write the v0.1 layout. kind = 'product' or 'commot' (recorded in uns)."""
    check_flows(flows, edges)
    adata.uns["complex"] = {"edges": edges, "triangles": triangles}
    adata.uns["flows"] = flows.astype(np.float32)
    adata.uns["flow_names"] = np.array(names, dtype=str)
    adata.uns["flow_kind"] = kind
    adata.write_h5ad(path)
    print(f"saved {flows.shape[1]} flow channels on {flows.shape[0]} edges -> {path}")
