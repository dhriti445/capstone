"""Hodge-SCNN: a simplicial convolutional network on edge flows.

Owner: Deeptha S (P1)

Input is one complex (B1, B2 from hodgecc.complex) and edge features
X of shape (E, C), one column per ligand-receptor pair.

Edge layer (Yang, Isufi and Leus, "Simplicial Convolutional Neural
Networks", 2021; same maths as topomodelx SCNNLayer):

    Y = tanh( X W0 + sum_k Ld^k X Wd_k + sum_k Lu^k X Wu_k )

    Ld = B1^T B1   (lower Laplacian, acts on the gradient part)
    Lu = B2 B2^T   (upper Laplacian, acts on the curl part)

Both are scaled by their largest eigenvalue so powers stay bounded.
We do not call topomodelx.SCNNLayer because it builds a dense
(E x E) identity in every forward pass: about 0.8 GB for a 14k-edge
Visium slice. This version keeps everything sparse.

Modes (used for the ablations):
    "hodge"   three branches: gradient part -> Ld filters,
              curl part -> Lu filters, harmonic part -> pointwise
    "grad"    gradient branch only
    "curl"    curl branch only
    "harm"    harmonic branch only
    "nosplit" one plain SCNN branch on the raw flow (no Hodge split)

The harmonic part h satisfies Ld h = 0 and Lu h = 0, so any Laplacian
filter reduces to its k = 0 term on it. The harmonic branch is
therefore pointwise (channel mixing only).

Orientation: edge layers use tanh and no bias, so flipping the
direction of an edge flips the sign of its features and nothing else.
Node features are built from B1 Y (net outflow) and |B1| |Y| (mean
flow size), which do not change when edge directions are flipped.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

MODES = ("hodge", "grad", "curl", "harm", "nosplit")


# ----------------------------------------------------------------------
# Sparse helpers
# ----------------------------------------------------------------------
def to_torch_sparse(M, dtype=torch.float32):
    """scipy sparse -> torch sparse COO tensor."""
    M = sp.coo_matrix(M)
    idx = torch.tensor(np.vstack([M.row, M.col]), dtype=torch.long)
    val = torch.tensor(M.data, dtype=dtype)
    return torch.sparse_coo_tensor(idx, val, M.shape,
                                   check_invariants=False).coalesce()


def _lambda_max(L):
    """Largest eigenvalue of a symmetric PSD sparse matrix.

    Must be exact: a rough estimate would change with edge orientation
    and break the orientation property of the model.
    """
    n = L.shape[0]
    if n == 0 or L.nnz == 0:
        return 1.0
    if n <= 200:
        lam = np.linalg.eigvalsh(L.toarray()).max()
    else:
        from scipy.sparse.linalg import eigsh
        v0 = np.random.default_rng(0).normal(size=n)
        lam = eigsh(L.astype(float), k=1, which="LM", tol=0, v0=v0,
                    return_eigenvectors=False)[0]
    return max(float(lam), 1e-12)


def hodge_operators(B1, B2):
    """Build every operator the model needs from one complex.

    Returns a dict of torch sparse tensors:
        Ld, Lu   scaled lower / upper Laplacians (E x E)
        B1       signed incidence (N x E)
        B1abs_n  |B1| with each row divided by node degree (N x E)
    """
    B1 = sp.csr_matrix(B1, dtype=float)
    B2 = sp.csr_matrix(B2, dtype=float)
    Ld = (B1.T @ B1).tocsr()
    Lu = (B2 @ B2.T).tocsr() if B2.shape[1] > 0 else sp.csr_matrix(Ld.shape)
    Ld = Ld / _lambda_max(Ld)
    Lu = Lu / _lambda_max(Lu) if Lu.nnz else Lu
    absB1 = abs(B1)
    deg = np.asarray(absB1.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    B1abs_n = sp.diags(1.0 / deg) @ absB1
    return {
        "Ld": to_torch_sparse(Ld),
        "Lu": to_torch_sparse(Lu),
        "B1": to_torch_sparse(B1),
        "B1abs_n": to_torch_sparse(B1abs_n),
    }


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------
class SCNNLayer(nn.Module):
    """Sparse SCNN layer: identity + K_down powers of Ld + K_up powers of Lu."""

    def __init__(self, in_ch, out_ch, k_down=2, k_up=2):
        super().__init__()
        self.k_down, self.k_up = k_down, k_up
        n_terms = 1 + k_down + k_up
        self.weight = nn.Parameter(torch.empty(n_terms, in_ch, out_ch))
        nn.init.xavier_uniform_(self.weight.view(n_terms * in_ch, out_ch))
        self.weight.data /= n_terms ** 0.5

    def forward(self, x, Ld, Lu):
        terms = [x]
        z = x
        for _ in range(self.k_down):
            z = torch.sparse.mm(Ld, z)
            terms.append(z)
        z = x
        for _ in range(self.k_up):
            z = torch.sparse.mm(Lu, z)
            terms.append(z)
        out = sum(t @ w for t, w in zip(terms, self.weight))
        return torch.tanh(out)


class Branch(nn.Module):
    """A stack of SCNN layers for one part of the flow."""

    def __init__(self, in_ch, hidden, n_layers, k_down, k_up):
        super().__init__()
        dims = [in_ch] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            SCNNLayer(a, b, k_down, k_up) for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x, Ld, Lu):
        for layer in self.layers:
            x = layer(x, Ld, Lu)
        return x


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class HodgeSCNN(nn.Module):
    """Edge flows -> edge, node and graph embeddings (+ optional head).

    forward(parts, ops) where
        parts: dict with keys "grad", "curl", "harm" (for Hodge modes)
               or "flow" (for "nosplit"); each a float tensor (E, C)
        ops:   output of hodge_operators(B1, B2)
    returns dict with "edge" (E, H*), "node_agg" (N, 2H*) = edge
    embeddings pooled onto spots (net outflow and mean size), "node"
    (N, D) = node_agg after an MLP, "graph" (2D,), and "out" (n_out,)
    if n_out > 0.
    """

    def __init__(self, in_ch, hidden=16, n_layers=2, k=2, mode="hodge",
                 node_dim=32, n_out=0):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        use = {
            "hodge": ("grad", "curl", "harm"),
            "grad": ("grad",),
            "curl": ("curl",),
            "harm": ("harm",),
            "nosplit": ("flow",),
        }[mode]
        # which Laplacian each branch is allowed to use
        orders = {"grad": (k, 0), "curl": (0, k), "harm": (0, 0), "flow": (k, k)}
        self.parts = use
        self.branches = nn.ModuleDict(
            {p: Branch(in_ch, hidden, n_layers, *orders[p]) for p in use}
        )
        edge_dim = hidden * len(use)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * edge_dim, node_dim), nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.head = nn.Linear(2 * node_dim, n_out) if n_out > 0 else None

    def forward(self, parts, ops):
        Ld, Lu = ops["Ld"], ops["Lu"]
        edge = torch.cat(
            [self.branches[p](parts[p], Ld, Lu) for p in self.parts], dim=1
        )
        div = torch.sparse.mm(ops["B1"], edge)              # net outflow
        size = torch.sparse.mm(ops["B1abs_n"], edge.abs())  # mean flow size
        node_agg = torch.cat([div, size], dim=1)
        node = self.node_mlp(node_agg)
        graph = torch.cat([node.mean(0), node.max(0).values])
        out = {"edge": edge, "node_agg": node_agg, "node": node,
               "graph": graph}
        if self.head is not None:
            out["out"] = self.head(graph)
        return out


def make_parts(B1, B2, F, mode):
    """Numpy flows (E, C) -> dict of torch tensors for HodgeSCNN.forward."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    if mode == "nosplit":
        return {"flow": torch.tensor(F, dtype=torch.float32)}
    from hodgecc.hodge import decompose
    g, c, h = decompose(B1, B2, F)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {"grad": t(g), "curl": t(c), "harm": t(h)}
