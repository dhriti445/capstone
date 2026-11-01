"""Tests for model/hodge_scnn.py (owner: Deeptha S, P1)."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from hodgecc.synth import grid_complex, planted_flows
from model.hodge_scnn import HodgeSCNN, hodge_operators, make_parts, MODES


@pytest.fixture(scope="module")
def toy():
    _, edges, tris, B1, B2 = grid_complex(10, 10, [(4.5, 4.5, 1.6)])
    f, _, _, _ = planted_flows(B1, B2, seed=3)
    F = np.column_stack([f, -0.5 * f + 0.1])  # two channels
    return edges, tris, B1, B2, F


@pytest.mark.parametrize("mode", MODES)
def test_shapes_and_backward(toy, mode):
    _, _, B1, B2, F = toy
    N, E = B1.shape
    torch.manual_seed(0)
    model = HodgeSCNN(in_ch=2, hidden=8, n_layers=2, k=2, mode=mode,
                      node_dim=12, n_out=3)
    out = model(make_parts(B1, B2, F, mode), hodge_operators(B1, B2))
    n_branch = 3 if mode == "hodge" else 1
    assert out["edge"].shape == (E, 8 * n_branch)
    assert out["node"].shape == (N, 12)
    assert out["graph"].shape == (24,)
    assert out["out"].shape == (3,)
    out["out"].sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_laplacians_scaled(toy):
    _, _, B1, B2, _ = toy
    ops = hodge_operators(B1, B2)
    for key in ("Ld", "Lu"):
        L = ops[key].to_dense().numpy()
        lam = np.linalg.eigvalsh(L).max()
        assert lam == pytest.approx(1.0, abs=1e-3)


def test_harmonic_branch_ignores_laplacians(toy):
    """Harmonic flows are in the kernel of Ld and Lu."""
    _, _, B1, B2, F = toy
    parts = make_parts(B1, B2, F, "hodge")
    ops = hodge_operators(B1, B2)
    h = parts["harm"]
    assert torch.sparse.mm(ops["Ld"], h).abs().max() < 1e-4
    assert torch.sparse.mm(ops["Lu"], h).abs().max() < 1e-4


@pytest.mark.parametrize("mode", MODES)
def test_orientation_invariance(toy, mode):
    """Flipping edge directions flips edge features, nodes unchanged."""
    edges, tris, B1, B2, F = toy
    E = B1.shape[1]
    rng = np.random.default_rng(0)
    s = rng.choice([-1.0, 1.0], size=E)
    D = sp.diags(s)
    B1f, B2f, Ff = B1 @ D, D @ B2, s[:, None] * F

    torch.manual_seed(0)
    model = HodgeSCNN(in_ch=2, hidden=8, mode=mode, n_out=3).eval()
    with torch.no_grad():
        a = model(make_parts(B1, B2, F, mode), hodge_operators(B1, B2))
        b = model(make_parts(B1f, B2f, Ff, mode), hodge_operators(B1f, B2f))
    n_branch = 3 if mode == "hodge" else 1
    sign = torch.tensor(np.tile(s[:, None], (1, 8 * n_branch)),
                        dtype=torch.float32)
    assert torch.allclose(b["edge"], sign * a["edge"], atol=1e-4)
    assert torch.allclose(b["node"], a["node"], atol=1e-4)
    assert torch.allclose(b["out"], a["out"], atol=1e-4)


def test_loss_goes_down(toy):
    _, _, B1, B2, F = toy
    torch.manual_seed(0)
    model = HodgeSCNN(in_ch=2, hidden=8, mode="hodge", n_out=3)
    parts = make_parts(B1, B2, F, "hodge")
    ops = hodge_operators(B1, B2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = torch.tensor([2])
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(parts, ops)["out"][None], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
