import numpy as np
from hodgecc.complex import build_complex, check_boundary


def test_boundary_zero_random_points():
    rng = np.random.default_rng(0)
    pts = rng.random((200, 2))
    edges, tris, B1, B2 = build_complex(pts)
    assert check_boundary(B1, B2)
    assert B1.shape == (200, len(edges))
    assert B2.shape == (len(edges), len(tris))


def test_canonical_order():
    rng = np.random.default_rng(1)
    edges, tris, _, _ = build_complex(rng.random((100, 2)))
    assert (edges[:, 0] < edges[:, 1]).all()
    assert (tris[:, 0] < tris[:, 1]).all() and (tris[:, 1] < tris[:, 2]).all()


def test_euler_characteristic_no_pruning():
    # Delaunay of a point cloud without pruning is a filled disk: N - E + T = 1
    rng = np.random.default_rng(2)
    pts = rng.random((150, 2))
    edges, tris, _, _ = build_complex(pts, max_edge_quantile=1.0)
    assert 150 - len(edges) + len(tris) == 1