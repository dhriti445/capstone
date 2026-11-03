"""Real vs shuffled control for Disha's COMMOT flows.

Usage:
  python scripts/shuffle_control.py REAL.h5ad SHUFFLE_DIR

REAL.h5ad     TENX53_flows_commot.h5ad (needs obs array_row/array_col)
SHUFFLE_DIR   folder with the TENX53_flows_commot_shuf*.h5ad files

For every pathway it decomposes the real flow and every shuffled flow into
gradient / curl / harmonic and compares the energy shares. Shuffled flows come
from COMMOT run on spot-shuffled expression, so they keep how much is
expressed but destroy where. Outputs:
  results/shuffle_control.csv
  figures/shuffle_control.png
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hodgecc.visium import build_visium_complex
from hodgecc.hodge import decompose, energy_fractions


def shares(B1, B2, F):
    return energy_fractions(*decompose(B1, B2, F))      # (3, C)


def main(real_path, shuf_dir):
    real = ad.read_h5ad(real_path)
    names = [str(x) for x in real.uns["flow_names"]]
    r, c = real.obs["array_row"].values, real.obs["array_col"].values
    Fr = np.asarray(real.uns["flows"], float)
    files = sorted(glob.glob(os.path.join(shuf_dir, "*.h5ad")))
    if not files:
        sys.exit("no .h5ad files found in " + shuf_dir)
    Fs = []
    for f in files:
        a = ad.read_h5ad(f)
        assert [str(x) for x in a.uns["flow_names"]] == names, f"{f}: names differ"
        assert np.array_equal(a.uns["complex"]["edges"], real.uns["complex"]["edges"]), \
            f"{f}: edges differ from the real file"
        Fs.append(np.asarray(a.uns["flows"], float))
    print(f"real: {Fr.shape}, shuffles: {len(Fs)}")

    res = {}
    for fill in (0, 2):
        _, _, B1, B2 = build_visium_complex(r, c, fill_max_missing=fill)
        res[fill] = (shares(B1, B2, Fr), np.array([shares(B1, B2, F) for F in Fs]))
    fr, fs = res[0]
    fr2, fs2 = res[2]

    rows = []
    for k, n in enumerate(names):
        g = fs[:, 0, k]
        rows.append({
            "pathway": n,
            "gradient_real_%": 100 * fr[0, k],
            "gradient_shuf_mean_%": 100 * g.mean(),
            "gradient_shuf_min_%": 100 * g.min(),
            "gradient_shuf_max_%": 100 * g.max(),
            "curl_real_%": 100 * fr[1, k],
            "curl_shuf_mean_%": 100 * fs[:, 1, k].mean(),
            "harmonic_real_%": 100 * fr[2, k],
            "harmonic_shuf_mean_%": 100 * fs[:, 2, k].mean(),
            "harmonic_real_filled_%": 100 * fr2[2, k],
            "harmonic_shuf_max_filled_%": 100 * fs2[:, 2, k].max(),
            "gradient_vs_shuffles": ("above all" if fr[0, k] > g.max()
                                     else "below all" if fr[0, k] < g.min()
                                     else "within range"),
        })
    df = pd.DataFrame(rows).set_index("pathway")
    os.makedirs("results", exist_ok=True)
    df.round(4).to_csv("results/shuffle_control.csv")

    counts = df["gradient_vs_shuffles"].value_counts()
    print("\ngradient share, real vs the", len(Fs), "shuffles:")
    print(counts.to_string())
    print("expected by chance with this many shuffles, per direction: about "
          f"{len(df) / (len(Fs) + 1):.1f} of {len(df)}")
    print("\nmedian gradient share: real %.1f%%  shuffled %.1f%%" %
          (df["gradient_real_%"].median(), df["gradient_shuf_mean_%"].median()))
    print("median curl share:     real %.1f%%  shuffled %.1f%%" %
          (df["curl_real_%"].median(), df["curl_shuf_mean_%"].median()))
    print("harmonic share (gaps filled): real max %.4f%%, shuffled max %.4f%%" %
          (df["harmonic_real_filled_%"].max(), df["harmonic_shuf_max_filled_%"].max()))

    # figure
    order = (df["gradient_real_%"] - df["gradient_shuf_mean_%"]).sort_values().index
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [3, 2]})
    ax = axs[0]
    for i, n in enumerate(order):
        d = df.loc[n]
        ax.plot([d["gradient_shuf_min_%"], d["gradient_shuf_max_%"]], [i, i],
                color="0.75", lw=6, solid_capstyle="round", zorder=1)
        ax.plot(d["gradient_shuf_mean_%"], i, "|", color="0.4", ms=10, zorder=2)
        col = {"above all": "tab:blue", "below all": "tab:orange",
               "within range": "0.2"}[d["gradient_vs_shuffles"]]
        ax.plot(d["gradient_real_%"], i, "o", color=col, ms=7, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("gradient share of flow energy (%)")
    ax.set_title("Gradient share: real (dot) vs 10 shuffles (grey bar = range)\n"
                 "blue = above all shuffles, orange = below all, black = within")
    ax.grid(axis="x", alpha=0.3)
    ax = axs[1]
    x = np.arange(len(order))
    ax.barh(x - 0.2, [df.loc[n, "harmonic_real_filled_%"] for n in order], 0.4,
            label="real", color="tab:blue")
    ax.barh(x + 0.2, [df.loc[n, "harmonic_shuf_max_filled_%"] for n in order], 0.4,
            label="shuffles (max)", color="0.7")
    ax.set_yticks(x)
    ax.set_yticklabels([])
    ax.set_xlabel("harmonic share (%), small gaps filled")
    ax.set_title("Harmonic share\n(loops around the one remaining hole)")
    ax.legend()
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/shuffle_control.png", dpi=130)
    print("\nsaved results/shuffle_control.csv and figures/shuffle_control.png")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
