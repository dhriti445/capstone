"""Hodge decomposition of edge flows on a 2-complex.

Given B1 (N x E), B2 (E x T) and flows F (E x C):
  gradient  = projection onto im(B1^T)
  curl      = projection onto im(B2)
  harmonic  = remainder = ker(B1) intersect ker(B2^T)
The three parts are mutually orthogonal because B1 @ B2 = 0.
"""
import numpy as np
from scipy.sparse.linalg import lsqr


def _project(M, f, tol, iter_lim):
    """Return M @ x where x minimises ||M x - f||."""
    x = lsqr(M, f, atol=tol, btol=tol, iter_lim=iter_lim)[0]
    return M @ x


def decompose(B1, B2, F, tol=1e-12, iter_lim=50000):
    """Return (grad, curl, harm), each the same shape as F."""
    F = np.asarray(F, dtype=float)
    squeeze = F.ndim == 1
    if squeeze:
        F = F[:, None]
    B1t = B1.T.tocsr()
    grad = np.zeros_like(F)
    curl = np.zeros_like(F)
    has_tri = B2.shape[1] > 0
    for k in range(F.shape[1]):
        f = F[:, k]
        grad[:, k] = _project(B1t, f, tol, iter_lim)
        if has_tri:
            curl[:, k] = _project(B2, f, tol, iter_lim)
    harm = F - grad - curl
    if squeeze:
        return grad[:, 0], curl[:, 0], harm[:, 0]
    return grad, curl, harm


def energy_fractions(grad, curl, harm):
    """Fraction of squared norm in each part, per channel: (3, C)."""
    g = np.atleast_2d(grad.T).T
    c = np.atleast_2d(curl.T).T
    h = np.atleast_2d(harm.T).T
    e = np.stack([(g**2).sum(0), (c**2).sum(0), (h**2).sum(0)])
    tot = e.sum(0)
    tot[tot == 0] = 1.0
    return e / tot


def harmonic_residual(B1, B2, harm):
    """Max-abs of B1 @ h and B2^T @ h. Both should be ~0."""
    r1 = abs(B1 @ harm).max()
    r2 = abs(B2.T @ harm).max() if B2.shape[1] > 0 else 0.0
    return max(r1, r2)


def _dense_stack(B1, B2):
    top = B1.toarray()
    if B2.shape[1] == 0:
        return top
    return np.vstack([top, B2.T.toarray()])


def nullity_L1_dense(B1, B2):
    """dim of harmonic space = beta_1. Dense, small complexes only."""
    A = _dense_stack(B1, B2)
    E = A.shape[1]
    return E - np.linalg.matrix_rank(A)


def harmonic_basis_dense(B1, B2):
    """Orthonormal basis (E x beta_1) of the harmonic space. Small only."""
    from scipy.linalg import null_space
    return null_space(_dense_stack(B1, B2))
