"""Tests for model/pretrain.py and eval/niches.py (owner: Deeptha S, P1)."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from hodgecc.synth import grid_complex, planted_flows
from model.hodge_scnn import hodge_operators, make_parts
from model.pretrain import make_masks, make_views, MaskedFlowModel, pretrain, embed, run
from eval.niches import score, evaluate, n_label_groups, results_rows


@pytest.fixture(scope="module")
def toy():
    _, _, _, B1, B2 = grid_complex(10, 10, [(4.5, 4.5, 1.6)])
    f, _, _, _ = planted_flows(B1, B2, seed=5)
    F = np.column_stack([f, 0.3 * f + 0.2 * np.sin(np.arange(len(f)))])
    return B1, B2, F


def test_masks():
    m = make_masks(200, n_masks=4, frac=0.15, seed=0)
    assert m.shape == (4, 200)
    assert (m.sum(axis=1) == 30).all()
    assert not (m[0] == m[1]).all()


def test_no_leak_from_hidden_edges(toy):
    """Parts are decomposed from the masked flow, so they add up to it."""
    B1, B2, F = toy
    masks = make_masks(F.shape[0], n_masks=2, seed=1)
    for mask, parts in make_views(B1, B2, F, "hodge", masks):
        total = (parts["grad"] + parts["curl"] + parts["harm"]).numpy()
        Fm = F.copy()
        Fm[mask.numpy()] = 0.0
        assert np.allclose(total, Fm, atol=1e-5)


def test_pretrain_loss_goes_down(toy):
    B1, B2, F = toy
    ops = hodge_operators(B1, B2)
    views = make_views(B1, B2, F, "hodge", make_masks(F.shape[0], 4, seed=0))
    torch.manual_seed(0)
    model = MaskedFlowModel(F.shape[1], hidden=8)
    hist = pretrain(model, ops, views, F, epochs=40, lr=1e-2)
    assert hist[-1] < 0.7 * hist[0]


@pytest.mark.parametrize("mode", ["hodge", "nosplit"])
def test_embeddings_shape_and_orientation(toy, mode):
    B1, B2, F = toy
    N, E = B1.shape
    torch.manual_seed(0)
    model = MaskedFlowModel(F.shape[1], hidden=8, mode=mode).eval()
    Z = embed(model, make_parts(B1, B2, F, mode), hodge_operators(B1, B2))
    n_branch = 3 if mode == "hodge" else 1
    assert Z.shape == (N, 2 * 8 * n_branch)

    s = np.random.default_rng(0).choice([-1.0, 1.0], size=E)
    D = sp.diags(s)
    B1f, B2f, Ff = B1 @ D, D @ B2, s[:, None] * F
    Zf = embed(model, make_parts(B1f, B2f, Ff, mode), hodge_operators(B1f, B2f))
    assert np.allclose(Z, Zf, atol=1e-4)


def test_run_end_to_end(toy):
    B1, B2, F = toy
    Z, hist = run(B1, B2, F, mode="hodge", hidden=8, epochs=5, n_masks=2)
    assert Z.shape[0] == B1.shape[0]
    assert np.isfinite(Z).all() and len(hist) == 5


def test_score_perfect_and_relabelled():
    t = np.array(["a", "a", "b", "b", "c", "c"])
    assert score(t, [0, 0, 1, 1, 2, 2])["ari"] == pytest.approx(1.0)
    assert score(t, [2, 2, 0, 0, 1, 1])["nmi"] == pytest.approx(1.0)


def test_missing_labels_not_scored():
    t = np.array(["a", "a", "b", "b", None, "", np.nan, "NA"], dtype=object)
    p = [0, 0, 1, 1, 0, 1, 0, 1]
    out = score(t, p)
    assert out["n_scored"] == 4 and out["ari"] == pytest.approx(1.0)
    assert n_label_groups(t) == 2


def test_evaluate_finds_clear_clusters():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 0.1, (30, 4)), rng.normal(5, 0.1, (30, 4))])
    t = np.array(["x"] * 30 + ["y"] * 30)
    out, pred = evaluate(Z, t)
    assert out["k"] == 2 and out["ari"] == pytest.approx(1.0)


def test_results_rows_format():
    rows = results_rows("E3", "DLPFC", "151673", "hodge-scnn", "hodge", 0, 1,
                        {"ari": 0.5, "nmi": 0.6})
    assert [r["metric"] for r in rows] == ["ari", "nmi"]
    assert set(rows[0]) == {"experiment_id", "dataset", "sample_id", "method",
                            "mode", "seed", "fold", "metric", "value"}
