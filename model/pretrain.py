"""Self-supervised pretraining for Hodge-SCNN: masked edge-flow reconstruction.

Owner: Deeptha S (P1)

Why: for niches (experiments E3-E5) the model must learn without labels.
Labels are only used afterwards, to score the clusters.

How:
  1. Hide a random 15% of edges: set their flow to 0 in every channel.
  2. Run the Hodge decomposition on the masked flow, so hidden values
     cannot leak into the gradient / curl / harmonic parts.
  3. The model sees the masked parts and predicts the true flow on the
     hidden edges from its edge embeddings (a linear decoder, no bias).
  4. Loss = mean squared error on hidden edges only.

The decomposition is slow compared with one training step (about 1 s
for 10 channels on a 14k-edge slice), so a small set of masks is made
once and reused ("views").

The decoder has no bias, so flipping an edge's direction flips the
prediction and the target together; the orientation property of the
model is kept.

After pretraining, embed() returns spot (node) embeddings for the full,
unmasked flow; eval/niches.py clusters and scores them.
"""
import numpy as np
import torch
from torch import nn

from model.hodge_scnn import HodgeSCNN, hodge_operators, make_parts


def make_masks(n_edges, n_masks=8, frac=0.15, seed=0):
    """Boolean masks (n_masks, E); True = hidden edge."""
    rng = np.random.default_rng(seed)
    k = max(1, int(round(frac * n_edges)))
    masks = np.zeros((n_masks, n_edges), dtype=bool)
    for m in range(n_masks):
        masks[m, rng.choice(n_edges, size=k, replace=False)] = True
    return masks


def make_views(B1, B2, F, mode, masks):
    """One (mask, parts) pair per mask, decomposed AFTER masking."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    views = []
    for m in masks:
        Fm = F.copy()
        Fm[m] = 0.0
        views.append((torch.tensor(m), make_parts(B1, B2, Fm, mode)))
    return views


class MaskedFlowModel(nn.Module):
    """Hodge-SCNN encoder + linear decoder back to the C flow channels."""

    def __init__(self, in_ch, hidden=16, n_layers=2, k=2, mode="hodge",
                 node_dim=32):
        super().__init__()
        self.encoder = HodgeSCNN(in_ch, hidden=hidden, n_layers=n_layers,
                                 k=k, mode=mode, node_dim=node_dim, n_out=0)
        n_branch = 3 if mode == "hodge" else 1
        self.decoder = nn.Linear(hidden * n_branch, in_ch, bias=False)

    def forward(self, parts, ops):
        out = self.encoder(parts, ops)
        out["recon"] = self.decoder(out["edge"])
        return out


def pretrain(model, ops, views, F, epochs=100, lr=5e-3, seed=0, verbose=False):
    """Train on the masked views; returns the list of mean losses per epoch."""
    torch.manual_seed(seed)
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    target = torch.tensor(F, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for i in rng.permutation(len(views)):
            mask, parts = views[i]
            pred = model(parts, ops)["recon"]
            loss = ((pred[mask] - target[mask]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        history.append(total / len(views))
        if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
            print(f"    epoch {ep:4d}  masked MSE {history[-1]:.4f}")
    return history


def embed(model, parts, ops):
    """Spot embeddings as numpy, for the full unmasked flow.

    Uses "node_agg" (trained edge embeddings pooled onto spots), not
    "node": the reconstruction loss only trains the edge layers, so the
    node MLP would still be at its random initial weights here.
    """
    model.eval()
    with torch.no_grad():
        return model(parts, ops)["node_agg"].numpy()


def run(B1, B2, F, mode="hodge", hidden=16, n_layers=2, k=2, node_dim=32,
        epochs=100, lr=5e-3, n_masks=8, frac=0.15, seed=0, verbose=False):
    """Full pipeline for one slice: masks -> pretrain -> node embeddings.

    Returns (Z, history) with Z of shape (N, 2 * hidden * n_branches).
    """
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    ops = hodge_operators(B1, B2)
    masks = make_masks(F.shape[0], n_masks=n_masks, frac=frac, seed=seed)
    views = make_views(B1, B2, F, mode, masks)
    torch.manual_seed(seed)
    model = MaskedFlowModel(F.shape[1], hidden=hidden, n_layers=n_layers,
                            k=k, mode=mode, node_dim=node_dim)
    history = pretrain(model, ops, views, F, epochs=epochs, lr=lr, seed=seed,
                       verbose=verbose)
    Z = embed(model, make_parts(B1, B2, F, mode), ops)
    return Z, history
