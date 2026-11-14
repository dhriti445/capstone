"""Can the Hodge engine find hidden feedback loops? (simulated tissue, RQ3)

Usage:  python scripts/circuit_recovery.py data/starter.h5ad

On the REAL TENX53 mesh we plant known local feedback loops (a 6-edge ring of
circulating flow around each of 40 random spots), hide them under a strong
smooth source-to-sink (gradient) background, and add noise. Then we ask: which
edges belong to a planted loop? Two scorers are compared with AUROC
(1.0 = perfect, 0.5 = guessing):
  raw     score = size of the flow on the edge (what you see without Hodge)
  hodge   score = size of the CURL part of the flow on the edge
Output: figures/circuit_recovery.png and a printed table.
"""
import sys
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hodgecc.visium import build_visium_complex
from hodgecc.hodge import decompose


def auroc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    npos = label.sum()
    nneg = len(label) - npos
    return (ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main(path):
    a = ad.read_h5ad(path)
    r, c = a.obs["array_row"].values, a.obs["array_col"].values
    xy = np.asarray(a.obsm["spatial"], float)
    edges, tris, B1, B2 = build_visium_complex(r, c)
    N, E = B1.shape
    T = B2.shape[1]

    # orientation (+1 counter-clockwise) of each stored triangle, so that the
    # triangles around a spot add up to one coherent loop
    P = xy[tris]
    cross = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
             - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    orient = np.sign(cross)
    tri_of = [[] for _ in range(N)]
    for k, tr in enumerate(tris):
        for v in tr:
            tri_of[v].append(k)
    interior = np.array([i for i in range(N) if len(tri_of[i]) == 6])

    def make(seed, grad_scale, noise_sd, n_loops=40):
        rng = np.random.default_rng(seed)
        s = np.zeros(N)
        for _ in range(6):                       # smooth sources and sinks
            c0 = xy[rng.integers(N)]
            w = rng.uniform(1500, 3500)
            s += rng.choice([-1, 1]) * np.exp(-((xy - c0) ** 2).sum(1) / (2 * w ** 2))
        g = B1.T @ s
        g = g / np.abs(g).max() * grad_scale
        t = np.zeros(T)
        for i in rng.choice(interior, n_loops, replace=False):
            t[tri_of[i]] = orient[tri_of[i]]
        loops = B2 @ t                           # unit-strength loop flow
        truth = np.abs(loops) > 1e-9
        f = g + loops + noise_sd * rng.normal(size=E)
        return f, truth

    noise_levels = [0.0, 0.3, 0.6, 1.0]
    backgrounds = [1.0, 3.0]
    res = {}
    print("loop strength 1.0; truth = %d loop edges of %d" % (make(0, 1, 0)[1].sum(), E))
    print("background  noise |  raw    hodge-curl")
    for gs in backgrounds:
        for ns in noise_levels:
            vals = []
            for seed in range(3):
                f, truth = make(seed, gs, ns)
                _, curl, _ = decompose(B1, B2, f)
                vals.append((auroc(np.abs(f), truth), auroc(np.abs(curl), truth)))
            res[(gs, ns)] = np.mean(vals, axis=0)
            print("  %5.1fx   %5.1f | %6.3f   %6.3f" % (gs, ns, *res[(gs, ns)]))

    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    for ax, gs in zip(axs, backgrounds):
        raw = [res[(gs, n)][0] for n in noise_levels]
        hod = [res[(gs, n)][1] for n in noise_levels]
        ax.plot(noise_levels, raw, "o-", color="#A9B2CB", lw=2, label="flow size (no Hodge)")
        ax.plot(noise_levels, hod, "o-", color="#1F6FB5", lw=2, label="Hodge curl part")
        ax.axhline(0.5, color="0.6", ls=":", lw=1)
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("noise level")
        ax.set_title("downhill background %.0f× loop strength" % gs, fontsize=10)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("AUROC for finding planted loops")
    axs[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Finding hidden feedback loops on the real TENX53 mesh", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/circuit_recovery.png", dpi=200)
    print("saved figures/circuit_recovery.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
