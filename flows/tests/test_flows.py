"""Tests for the P2 code. Run from the p2_flows folder:  python -m pytest -q tests

They use a small synthetic tissue, so they need no downloaded data.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flows import build_incidence, product_flow, matrix_to_edge_flow, gene_values, preprocess, check_flows  # noqa: E402
from hodge_quick import decompose, check_decomposition  # noqa: E402
from synthetic import make_tissue  # noqa: E402


@pytest.fixture(scope="module")
def tissue():
    return make_tissue(seed=0)


def test_complex_is_consistent(tissue):
    adata, edges, triangles = tissue
    B1, B2 = build_incidence(adata.n_obs, edges, triangles)
    assert abs(B1 @ B2).max() == 0
    assert (edges[:, 0] < edges[:, 1]).all()


def test_product_flow_flips_sign_when_edge_flips(tissue):
    adata, edges, _ = tissue
    rng = np.random.default_rng(1)
    L, R = rng.random(adata.n_obs), rng.random(adata.n_obs)
    f = product_flow(L, R, edges)
    f_flipped = product_flow(L, R, edges[:, ::-1])
    assert np.allclose(f, -f_flipped)


def test_no_expression_no_flow(tissue):
    adata, edges, _ = tissue
    L = np.zeros(adata.n_obs)
    R = np.random.default_rng(2).random(adata.n_obs)
    assert np.allclose(product_flow(L, R, edges), 0)


def test_sender_receiver_matrix_to_flow(tissue):
    adata, edges, _ = tissue
    i, j = edges[0]
    P = sp.lil_matrix((adata.n_obs, adata.n_obs))
    P[i, j] = 3.0   # i sends to j
    P[j, i] = 1.0   # j sends back less
    f, kept = matrix_to_edge_flow(P, edges)
    assert f[0] == pytest.approx(2.0) and np.count_nonzero(f) == 1 and kept == pytest.approx(1.0)


def test_complex_uses_min_subunit(tissue):
    adata, _, _ = tissue
    ad = preprocess(adata.copy())
    a = gene_values(ad, "RECA")
    b = gene_values(ad, "RECB")
    both = gene_values(ad, "RECA_RECB")
    assert np.allclose(both, np.minimum(a, b))
    assert gene_values(ad, "NOT_A_GENE") is None


def test_hodge_split_is_exact(tissue):
    adata, edges, triangles = tissue
    B1, B2 = build_incidence(adata.n_obs, edges, triangles)
    f = np.random.default_rng(3).standard_normal(len(edges))
    g, c, h = decompose(f, B1, B2)
    errs = check_decomposition(f, g, c, h, B1, B2)
    assert max(errs.values()) < 1e-5, errs


def test_check_flows_rejects_bad_shape(tissue):
    _, edges, _ = tissue
    with pytest.raises(AssertionError):
        check_flows(np.zeros((len(edges) - 1, 2)), edges)


def test_fast_solver_matches_lsqr(tissue):
    from hodge_quick import HodgeSolver
    adata, edges, triangles = tissue
    B1, B2 = build_incidence(adata.n_obs, edges, triangles)
    f = np.random.default_rng(4).standard_normal(len(edges))
    fast = HodgeSolver(B1, B2)(f)
    slow = decompose(f, B1, B2)
    for a, b in zip(fast, slow):
        assert np.allclose(a, b, atol=1e-6)


def test_circulation_around_hole(tissue):
    from holes import ordered_loops, orient_anticlockwise, circ_index
    adata, edges, triangles = tissue
    xy = adata.obsm["spatial"]
    loops = orient_anticlockwise(ordered_loops(edges, triangles, adata.n_obs), xy)
    assert len(loops) == 2
    hole = adata.uns["demo_holes"][0]
    mid = (xy[edges[:, 0]] + xy[edges[:, 1]]) / 2
    d = mid - hole
    tang = np.c_[d[:, 1], -d[:, 0]] / np.linalg.norm(d, axis=1, keepdims=True) ** 2  # anticlockwise on screen
    f = np.sum(tang * (xy[edges[:, 1]] - xy[edges[:, 0]]), axis=1)
    idx = sorted(abs(circ_index(f, L)) for L in loops)
    assert idx[1] > 0.99 and idx[0] < 0.01            # circles the planted hole, not the other one
    grad = adata.obsm["spatial"][:, 0]                 # any node potential
    g = grad[edges[:, 1]] - grad[edges[:, 0]]
    assert all(abs(circ_index(g, L)) < 1e-9 for L in loops)   # gradients never circulate
