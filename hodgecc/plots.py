"""Plotting functions reused in every slide and figure.

All functions take spot coordinates from adata.obsm['spatial'] (HEST stores
them in full-resolution image pixels) and draw in that pixel space, so the
H&E image lines up when one is available.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

PART_COLORS = {"flow": "#1f2a44", "gradient": "#2166ac", "curl": "#c0392b", "harmonic": "#1a9a80"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def coords(adata):
    return np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]


def background(adata):
    """Find a low-res H&E image stored in the .h5ad, if any.

    Returns (image, scale) where image pixel = full-res pixel * scale, or (None, None).
    """
    spatial = adata.uns.get("spatial", {})
    for lib in spatial.values():
        images, sf = lib.get("images", {}), lib.get("scalefactors", {})
        for key in ("downscaled_fullres", "hires", "lowres"):
            if key in images:
                for sk in (f"tissue_{key}_scalef", f"tissue_{key}_scale"):
                    if sk in sf:
                        return np.asarray(images[key]), float(sf[sk])
    return None, None


def _setup(ax, adata, show_he=True, title=None):
    xy = coords(adata)
    img, scale = background(adata) if show_he else (None, None)
    if img is not None:
        h, w = img.shape[:2]
        ax.imshow(img, extent=[0, w / scale, h / scale, 0], alpha=0.45)
    pad = 0.03 * np.ptp(xy, axis=0).max()
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)   # image y points down
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    return xy


def median_edge_length(xy, edges):
    return float(np.median(np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1)))


# ---------------------------------------------------------------------------
# 1. Values on spots
# ---------------------------------------------------------------------------
def plot_spots(adata, values, ax=None, title=None, cmap="viridis", show_he=True, size=None, colorbar=True):
    """One value per spot (gene expression, niche label, divergence...)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    xy = _setup(ax, adata, show_he, title)
    size = size or max(2.0, 30000 / len(xy))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=size, cmap=cmap, linewidths=0)
    if colorbar:
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    return ax


# ---------------------------------------------------------------------------
# 2. Edge flow as arrows
# ---------------------------------------------------------------------------
def plot_edge_flow(adata, edges, flow, ax=None, title=None, top_k=1500, color=None,
                   vmax=None, show_he=True, show_spots=True):
    """One arrow per edge, from its midpoint, pointing the way the signal goes.

    Positive flow on edge (i, j) points i -> j; negative points j -> i.
    Arrow length is proportional to |flow| (shared scale if vmax is given).
    Only the top_k strongest edges are drawn; 14k arrows are unreadable.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    xy = _setup(ax, adata, show_he, title)
    if show_spots:
        ax.scatter(xy[:, 0], xy[:, 1], s=1, c="#b0b0b0", linewidths=0)
    flow = np.asarray(flow, dtype=float)
    mag = np.abs(flow)
    vmax = vmax or (mag.max() if mag.max() > 0 else 1.0)
    keep = np.argsort(mag)[::-1][: min(top_k, len(mag))]
    keep = keep[mag[keep] > 0]
    a, b = xy[edges[keep, 0]], xy[edges[keep, 1]]
    mid = (a + b) / 2
    unit = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
    length = 0.9 * median_edge_length(xy, edges) * (mag[keep] / vmax)
    vec = unit * (np.sign(flow[keep]) * length)[:, None]
    ax.quiver(mid[:, 0] - vec[:, 0] / 2, mid[:, 1] - vec[:, 1] / 2, vec[:, 0], vec[:, 1],
              angles="xy", scale_units="xy", scale=1, color=color or PART_COLORS["flow"],
              width=0.0028, headwidth=4, headlength=4.5, headaxislength=4)
    return ax


# ---------------------------------------------------------------------------
# 3. Four Hodge panels
# ---------------------------------------------------------------------------
def plot_hodge_panels(adata, edges, f, grad, curl, harm, title="", top_k=1500, path=None, show_he=True):
    """Raw flow | gradient | curl | harmonic, on one shared arrow scale."""
    shares = [float(x @ x) / max(float(f @ f), 1e-30) for x in (grad, curl, harm)]
    vmax = np.abs(f).max() or 1.0
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    panels = [("flow", f, "Observed flow f"),
              ("gradient", grad, f"Gradient B1ᵀs  ({shares[0]:.0%})"),
              ("curl", curl, f"Curl B2c  ({shares[1]:.0%})"),
              ("harmonic", harm, f"Harmonic h  ({shares[2]:.0%})")]
    for ax, (key, x, t) in zip(axes, panels):
        plot_edge_flow(adata, edges, x, ax=ax, title=t, top_k=top_k, color=PART_COLORS[key],
                       vmax=vmax, show_he=show_he)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Holes of the complex
# ---------------------------------------------------------------------------
def boundary_loops(edges, triangles, n_nodes):
    """Edges on the boundary of the complex, grouped into loops.

    A boundary edge belongs to fewer than 2 triangles. The boundary edges
    split into connected loops: the biggest is the outer edge of the tissue,
    the others go around holes (that is where harmonic flow can live).
    Returns (boundary_edge_index, loop_label_per_boundary_edge, outer_label).
    """
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(edges)}
    count = np.zeros(len(edges), dtype=int)
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (a, c)):
            count[index[(int(u), int(v))]] += 1
    bnd = np.where(count < 2)[0]
    e = edges[bnd]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_nodes, n_nodes))
    _, node_label = connected_components(A, directed=False)
    loop = node_label[e[:, 0]]
    sizes = np.bincount(loop)
    return bnd, loop, int(np.argmax(sizes))


def plot_holes(adata, edges, triangles, ax=None, title="Boundary of the complex", show_he=True, min_edges=3):
    """Outer boundary in grey, hole boundaries in red, numbered."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    xy = _setup(ax, adata, show_he, title)
    ax.scatter(xy[:, 0], xy[:, 1], s=1, c="#c8c8c8", linewidths=0)
    bnd, loop, outer = boundary_loops(edges, triangles, len(xy))
    segs = np.stack([xy[edges[bnd, 0]], xy[edges[bnd, 1]]], axis=1)
    is_outer = loop == outer
    ax.add_collection(LineCollection(segs[is_outer], colors="#808080", linewidths=1.0))
    ax.add_collection(LineCollection(segs[~is_outer], colors="#d62728", linewidths=1.8))
    n = 0
    for lab in np.unique(loop[~is_outer]):
        sel = loop == lab
        if sel.sum() < min_edges:
            continue
        n += 1
        c = segs[sel].reshape(-1, 2).mean(axis=0)
        ax.text(c[0], c[1], str(n), color="#d62728", fontsize=10, ha="center", va="center", weight="bold")
    return ax, n


# ---------------------------------------------------------------------------
# 5. Energy split per channel
# ---------------------------------------------------------------------------
def plot_energy_bars(names, shares, ax=None, title="Share of flow energy in each Hodge part", path=None):
    """shares: list of dicts {'gradient','curl','harmonic'} -> stacked bars."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.6 * len(names) + 2), 4.5))
    bottom = np.zeros(len(names))
    for part in ("gradient", "curl", "harmonic"):
        vals = np.array([s[part] for s in shares])
        ax.bar(names, vals, bottom=bottom, color=PART_COLORS[part], label=part)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of energy")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    if path:
        ax.figure.tight_layout()
        ax.figure.savefig(path, dpi=200, bbox_inches="tight")
    return ax
