"""Quick Hodge decomposition so the plots have something to show.

TEMPORARY: replace with Dhriti's hodgecc decomposition when it is ready.
Same maths as the architecture doc:
  f = B1^T s (gradient) + B2 c (curl) + h (harmonic)
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import lsqr


def decompose(f, B1, B2, tol=1e-10):
    """Split one edge flow f (E,) into gradient, curl and harmonic parts."""
    s = lsqr(B1.T, f, atol=tol, btol=tol, iter_lim=20000)[0]
    grad = B1.T @ s
    c = lsqr(B2, f, atol=tol, btol=tol, iter_lim=20000)[0]
    curl = B2 @ c
    harm = f - grad - curl
    return grad, curl, harm


class HodgeSolver:
    """Exact decomposition that factorises the Laplacians once.

    Use it when you decompose many flows on the same complex (e.g. 100
    permutations x 10 channels). Gradient: solve L0 s = B1 f with one node
    per connected piece fixed to 0 (L0 is singular only by constants).
    Curl: solve L2 c = B2^T f; L2 = B2^T B2 is invertible on a 2-D tissue
    complex (no closed surfaces).
    """

    def __init__(self, B1, B2):
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse.linalg import splu
        import scipy.sparse as sp

        self.B1, self.B2 = B1.tocsr(), B2.tocsr()
        L0 = (self.B1 @ self.B1.T).tocsc()
        _, lab = connected_components(L0, directed=False)
        grounded = np.array([np.where(lab == k)[0][0] for k in np.unique(lab)])
        self.free = np.setdiff1d(np.arange(L0.shape[0]), grounded)
        self.lu0 = splu(L0[self.free][:, self.free].tocsc())
        self.lu2 = splu((self.B2.T @ self.B2).tocsc()) if self.B2.shape[1] else None

    def __call__(self, f):
        s = np.zeros(self.B1.shape[0])
        s[self.free] = self.lu0.solve((self.B1 @ f)[self.free])
        grad = self.B1.T @ s
        curl = self.B2 @ self.lu2.solve(self.B2.T @ f) if self.lu2 is not None else np.zeros_like(f)
        return grad, curl, f - grad - curl


def decompose_all(flows, B1, B2):
    """flows (E, C) -> dict of three (E, C) arrays."""
    parts = [decompose(flows[:, k], B1, B2) for k in range(flows.shape[1])]
    return {name: np.column_stack([p[i] for p in parts]) for i, name in enumerate(("gradient", "curl", "harmonic"))}


def energy_shares(f, grad, curl, harm):
    """Share of flow energy (sum of squares) in each part. Adds to 1."""
    total = float(f @ f)
    if total == 0:
        return {"gradient": 0.0, "curl": 0.0, "harmonic": 0.0}
    return {"gradient": float(grad @ grad) / total, "curl": float(curl @ curl) / total, "harmonic": float(harm @ harm) / total}


def check_decomposition(f, grad, curl, harm, B1, B2, rtol=1e-6):
    """The done-checks from the architecture doc. Returns a dict of errors."""
    total = max(float(f @ f), 1e-30)
    return {
        "grad.curl": abs(float(grad @ curl)) / total,
        "grad.harm": abs(float(grad @ harm)) / total,
        "curl.harm": abs(float(curl @ harm)) / total,
        "energy_gap": abs(float(grad @ grad + curl @ curl + harm @ harm) - total) / total,
        "div(h)": float(np.linalg.norm(B1 @ harm)) / np.sqrt(total),
        "curl(h)": float(np.linalg.norm(B2.T @ harm)) / np.sqrt(total),
    }
