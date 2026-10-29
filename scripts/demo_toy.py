"""Visual check: plant gradient + curl + harmonic flows on a toy mesh with
two holes, decompose, print recovery errors, save a 4-panel figure.

Usage:  python scripts/demo_toy.py
Output: figures/toy_decomposition.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hodgecc.synth import grid_complex, planted_flows
from hodgecc.hodge import decompose, nullity_L1_dense, harmonic_residual


def draw(ax, coords, edges, flow, title):
    p = coords[edges[:, 0]]
    q = coords[edges[:, 1]]
    mid = (p + q) / 2
    d = (q - p) * np.sign(flow)[:, None]
    scale = np.abs(flow) / (np.abs(flow).max() + 1e-12)
    ax.quiver(mid[:, 0], mid[:, 1], d[:, 0], d[:, 1], scale,
              cmap="viridis", angles="xy", scale_units="xy",
              scale=1.4, width=0.003,
              headwidth=3.5, headlength=4)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    holes = [(5.5, 5.5, 2.0), (14.5, 14.5, 2.0)]
    coords, edges, tris, B1, B2 = grid_complex(20, 20, holes)
    f, g, c, h = planted_flows(B1, B2, seed=1)
    gg, cc, hh = decompose(B1, B2, f)

    def rel(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    print("nodes", len(coords), "edges", len(edges), "triangles", len(tris))
    print("beta1 (should be 2):", nullity_L1_dense(B1, B2))
    print("relative error gradient :", rel(gg, g))
    print("relative error curl     :", rel(cc, c))
    print("relative error harmonic :", rel(hh, h))
    print("harmonic residual       :", harmonic_residual(B1, B2, hh))

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    draw(axs[0], coords, edges, f, "Mixed flow (input)")
    draw(axs[1], coords, edges, gg, "Gradient (source to sink)")
    draw(axs[2], coords, edges, cc, "Curl (small loops)")
    draw(axs[3], coords, edges, hh, "Harmonic (around holes)")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/toy_decomposition.png", dpi=130)
    print("saved figures/toy_decomposition.png")


if __name__ == "__main__":
    main()
