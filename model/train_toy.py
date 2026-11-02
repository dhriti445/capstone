"""Train Hodge-SCNN on toy complexes and run the ablations.

Owner: Deeptha S (P1)

Toy task: each sample is a triangulated grid with 1-2 holes and one
edge flow made of a gradient, a curl and a harmonic part. One part is
dominant (weight 1, the other two weight 0.35). The label is which part
dominates (0 = gradient, 1 = curl, 2 = harmonic). Flows are scaled to
RMS 1, so the model has to use the shape of the flow, not its size.

A model that only sees one Hodge part can tell "my part dominates" from
"it doesn't", but cannot separate the other two classes, so it should
top out near 67% accuracy. The full Hodge model and the plain SCNN
should get close to 100%. This is a sanity check of the ablation logic,
not a result for the paper.

Run from the repo root:
    python -m model.train_toy
    python -m model.train_toy --epochs 40 --n-train 120 --n-test 60
"""
import argparse
import time

import numpy as np
import torch

from hodgecc.synth import grid_complex
from hodgecc.hodge import harmonic_basis_dense
from model.hodge_scnn import HodgeSCNN, hodge_operators, make_parts, MODES

CLASS_NAMES = ("gradient", "curl", "harmonic")


def random_holes(rng, n, size):
    """1-2 non-overlapping holes away from the border."""
    for _ in range(1000):
        pts = rng.uniform(2.5, size - 3.5, size=(n, 2))
        if n == 1 or np.hypot(*(pts[0] - pts[1])) > 4.5:
            return [(cx, cy, 1.6) for cx, cy in pts]
    raise RuntimeError("could not place holes; use a larger grid")


def make_sample(rng, size=12, weak=0.35, noise=0.05):
    holes = random_holes(rng, int(rng.integers(1, 3)), size)
    _, _, _, B1, B2 = grid_complex(size, size, holes)
    N, E = B1.shape
    g = B1.T @ rng.normal(size=N)
    c = B2 @ rng.normal(size=B2.shape[1])
    H = harmonic_basis_dense(B1, B2)
    h = H @ rng.normal(size=H.shape[1])
    unit = [v / np.linalg.norm(v) for v in (g, c, h)]
    y = int(rng.integers(0, 3))
    w = np.full(3, weak)
    w[y] = 1.0
    f = sum(wi * ui for wi, ui in zip(w, unit))
    f = f + noise * rng.normal(size=E) / np.sqrt(E)
    f = f / np.sqrt(np.mean(f ** 2))
    return B1, B2, f, y


def build_dataset(n, seed):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        B1, B2, f, y = make_sample(rng)
        ops = hodge_operators(B1, B2)
        parts = make_parts(B1, B2, f, "hodge")
        parts.update(make_parts(B1, B2, f, "nosplit"))
        data.append((ops, parts, y))
    return data


def accuracy(model, data):
    model.eval()
    with torch.no_grad():
        hits = [int(model(p, o)["out"].argmax()) == y for o, p, y in data]
    return float(np.mean(hits))


def train_one(mode, train, test, epochs, lr, seed, verbose):
    torch.manual_seed(seed)
    model = HodgeSCNN(in_ch=1, hidden=16, n_layers=2, k=2, mode=mode, n_out=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for i in rng.permutation(len(train)):
            ops, parts, y = train[i]
            logits = model(parts, ops)["out"]
            loss = loss_fn(logits[None], torch.tensor([y]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        history.append(total / len(train))
        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(f"  [{mode}] epoch {ep:3d}  train loss {history[-1]:.4f}")
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=90)
    ap.add_argument("--n-test", type=int, default=45)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    args = ap.parse_args()

    t0 = time.time()
    train = build_dataset(args.n_train, seed=args.seed)
    test = build_dataset(args.n_test, seed=args.seed + 1000)
    o, p, _ = train[0]
    print(f"toy data: {len(train)} train, {len(test)} test graphs; "
          f"first graph has {o['B1'].shape[0]} nodes, {o['B1'].shape[1]} edges "
          f"({time.time() - t0:.1f}s)")
    counts = np.bincount([y for _, _, y in train], minlength=3)
    print("train class counts (grad, curl, harm):", counts.tolist())

    results = {}
    for mode in args.modes:
        t1 = time.time()
        model, hist = train_one(mode, train, test, args.epochs, args.lr,
                                args.seed, verbose=True)
        n_par = sum(p.numel() for p in model.parameters())
        results[mode] = (hist[0], hist[-1], accuracy(model, train),
                         accuracy(model, test), n_par, time.time() - t1)

    print("\nmode      params  loss(first->last)  train acc  test acc  time")
    for mode, (l0, l1, tr, te, n_par, dt) in results.items():
        print(f"{mode:8s}  {n_par:6d}  {l0:.3f} -> {l1:.3f}     "
              f"{tr:6.2f}     {te:6.2f}   {dt:5.1f}s")


if __name__ == "__main__":
    main()
