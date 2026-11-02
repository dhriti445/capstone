import numpy as np
from hodgecc.visium import build_visium_complex, enclosed_gaps
from hodgecc.complex import check_boundary
from hodgecc.hodge import nullity_L1_dense


def lattice(nrows=14, ncols=30, remove=()):
    sites = [(r, c) for r in range(nrows) for c in range(ncols)
             if r % 2 == c % 2 and (r, c) not in set(remove)]
    r = np.array([s[0] for s in sites])
    c = np.array([s[1] for s in sites])
    return r, c


def test_gaps_found_and_filled():
    # one single missing spot and one adjacent pair, both well inside
    remove = [(5, 7), (7, 17), (7, 19)]
    r, c = lattice(remove=remove)
    assert sorted(len(g) for g in enclosed_gaps(r, c)) == [1, 2]

    _, _, B1, B2 = build_visium_complex(r, c)
    assert nullity_L1_dense(B1, B2) == 2

    _, _, B1f, B2f = build_visium_complex(r, c, fill_max_missing=2)
    assert check_boundary(B1f, B2f)
    assert B2f.shape[1] == B2.shape[1] + 2
    assert nullity_L1_dense(B1f, B2f) == 0


def test_only_small_gaps_filled():
    # a gap of 3 sites stays a hole when the limit is 2
    remove = [(5, 7), (5, 9), (6, 8)]
    r, c = lattice(remove=remove)
    assert [len(g) for g in enclosed_gaps(r, c)] == [3]
    _, _, B1, B2 = build_visium_complex(r, c, fill_max_missing=2)
    assert nullity_L1_dense(B1, B2) == 1
    _, _, B1, B2 = build_visium_complex(r, c, fill_max_missing=3)
    assert nullity_L1_dense(B1, B2) == 0


def test_edges_unchanged_by_filling():
    r, c = lattice(remove=[(5, 7)])
    e0, t0, _, _ = build_visium_complex(r, c)
    e1, t1, _, _ = build_visium_complex(r, c, fill_max_missing=2)
    assert (e0 == e1).all() and (t0 == t1).all()
