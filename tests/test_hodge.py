import numpy as np
import pytest
from hodgecc.synth import grid_complex, planted_flows
from hodgecc.complex import check_boundary
from hodgecc.hodge import decompose, nullity_L1_dense, harmonic_residual

HOLES = [
    [],
    [(9.5, 9.5, 2.2)],
    [(5.5, 5.5, 2.0), (14.5, 14.5, 2.0)],
]


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("holes", HOLES)
def test_boundary_and_beta1(holes):
    _, _, _, B1, B2 = grid_complex(20, 20, holes)
    assert check_boundary(B1, B2)
    assert nullity_L1_dense(B1, B2) == len(holes)


@pytest.mark.parametrize("holes", HOLES[1:])
def test_recovery_of_planted_parts(holes):
    _, _, _, B1, B2 = grid_complex(20, 20, holes)
    f, g, c, h = planted_flows(B1, B2, seed=1)
    gg, cc, hh = decompose(B1, B2, f)
    assert rel(gg, g) < 1e-6
    assert rel(cc, c) < 1e-6
    assert rel(hh, h) < 1e-6
    assert np.allclose(gg + cc + hh, f)


@pytest.mark.parametrize("holes", HOLES)
def test_orthogonality_and_harmonic_residual(holes):
    _, _, _, B1, B2 = grid_complex(20, 20, holes)
    rng = np.random.default_rng(3)
    f = rng.normal(size=B1.shape[1])
    gg, cc, hh = decompose(B1, B2, f)
    n = np.linalg.norm(f)
    assert abs(gg @ cc) / n**2 < 1e-8
    assert abs(gg @ hh) / n**2 < 1e-8
    assert abs(cc @ hh) / n**2 < 1e-8
    assert harmonic_residual(B1, B2, hh) < 1e-6


def test_no_holes_means_no_harmonic():
    _, _, _, B1, B2 = grid_complex(20, 20, [])
    f = np.random.default_rng(4).normal(size=B1.shape[1])
    _, _, hh = decompose(B1, B2, f)
    assert np.linalg.norm(hh) / np.linalg.norm(f) < 1e-6


def test_multichannel_matches_single():
    _, _, _, B1, B2 = grid_complex(12, 12, [(5.5, 5.5, 1.5)])
    F = np.random.default_rng(5).normal(size=(B1.shape[1], 3))
    g, c, h = decompose(B1, B2, F)
    g0, c0, h0 = decompose(B1, B2, F[:, 1])
    assert np.allclose(g[:, 1], g0) and np.allclose(h[:, 1], h0)
