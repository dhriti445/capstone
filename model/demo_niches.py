"""Toy niche test: can pretraining + clustering find regions without labels?

Owner: Deeptha S (P1)

Toy tissue: a 20 x 20 triangulated grid with one hole, cut into three
vertical bands (the "true niches"):
  left band    gradient-rich flows (local sources and sinks)
  middle band  quiet (weak noise only)
  right band   curl-rich flows (local whirlpools)
There are 4 flow channels, like 4 ligand-receptor pairs, with different
random mixtures. Labels are never shown to the model; they are used only
to score the clusters (ARI, NMI).

Baseline: k-means on simple per-spot flow summaries (net outflow and
mean flow size for each channel), with no learning and no Hodge split.

--tissues N makes N toy "sections": the same grid with different random
flows (tissue seeds 0..N-1). With --out, every score is written in the
shared results format (experiment E3toy, dataset TOY) so that
eval/compare.py can build a scoreboard from it.

Run from the repo root:
    python -m model.demo_niches
    python -m model.demo_niches --epochs 150 --seeds 0 1 2
    python -m model.demo_niches --tissues 8 --modes hodge nosplit --seeds 0 \
        --out results/toy_E3.csv
"""
import argparse
import time

import numpy as np
import pandas as pd

from hodgecc.synth import grid_complex
from model.hodge_scnn import MODES
from model.pretrain import run
from eval.niches import evaluate, results_rows

EXPERIMENT, DATASET = "E3toy", "TOY"
OURS, BASELINE = "hodge-scnn", "flowsummary-kmeans"


def toy_tissue(seed=0, size=20, n_ch=4, noise=0.05):
    rng = np.random.default_rng(seed)
    coords, edges, tris, B1, B2 = grid_complex(size, size, [(9.5, 9.5, 1.8)])
    x = coords[:, 0]
    band = np.where(x < size / 3, 0, np.where(x < 2 * size / 3, 1, 2))
    labels = np.array(["gradient-rich", "quiet", "curl-rich"])[band].astype(object)
    isolated = np.asarray(abs(B1).sum(axis=1)).ravel() == 0  # grid points in the hole
    labels[isolated] = ""  # not tissue: not scored

    tri_band = band[tris].min(axis=1)  # triangle fully inside a band
    E = B1.shape[1]
    F = np.zeros((E, n_ch))
    for ch in range(n_ch):
        pot = rng.normal(size=B1.shape[0]) * (band == 0)       # left only
        rot = rng.normal(size=B2.shape[1]) * (tri_band == 2)    # right only
        a, b = rng.uniform(0.5, 1.5, size=2)
        F[:, ch] = a * (B1.T @ pot) + b * (B2 @ rot)
    F += noise * rng.normal(size=F.shape)
    F /= np.sqrt(np.mean(F ** 2))
    return B1, B2, F, labels


def baseline_features(B1, F):
    absB1 = abs(B1)
    deg = np.asarray(absB1.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    div = B1 @ F
    size = (absB1 @ np.abs(F)) / deg[:, None]
    return np.hstack([div, size])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--tissues", type=int, default=1,
                    help="number of toy sections (different random flows)")
    ap.add_argument("--out", help="write scores in the shared results format (CSV)")
    args = ap.parse_args()

    # name -> list of score dicts (one per tissue x seed)
    scores = {BASELINE: []}
    rows = []
    t_start = time.time()
    for tissue in range(args.tissues):
        B1, B2, F, labels = toy_tissue(seed=tissue)
        sample_id = f"toy{tissue}"
        if tissue == 0:
            print(f"toy tissue: {int((labels != '').sum())} spots, "
                  f"{B1.shape[1]} edges, {B2.shape[1]} triangles, "
                  f"{F.shape[1]} channels, 3 true niches, "
                  f"{args.tissues} section(s)")
        for s in args.seeds:
            sc, _ = evaluate(baseline_features(B1, F), labels, seed=s)
            scores[BASELINE].append(sc)
            rows += results_rows(EXPERIMENT, DATASET, sample_id, BASELINE, "",
                                 s, 0, sc)
        for mode in args.modes:
            for s in args.seeds:
                Z, hist = run(B1, B2, F, mode=mode, epochs=args.epochs, seed=s)
                sc, _ = evaluate(Z, labels, seed=s)
                sc["loss0"], sc["loss1"] = hist[0], hist[-1]
                scores.setdefault(mode, []).append(sc)
                rows += results_rows(EXPERIMENT, DATASET, sample_id, OURS, mode,
                                     s, 0, sc)
        print(f"  section {tissue + 1}/{args.tissues} done "
              f"({time.time() - t_start:.0f}s elapsed)")

    print("\nmethod     ARI (mean ± sd)   NMI (mean ± sd)   masked MSE first -> last")
    for name, res in scores.items():
        ari = np.array([r["ari"] for r in res])
        nmi = np.array([r["nmi"] for r in res])
        loss = (f"{np.mean([r['loss0'] for r in res]):.3f} -> "
                f"{np.mean([r['loss1'] for r in res]):.3f}") if "loss0" in res[0] else "-"
        label = "baseline" if name == BASELINE else name
        print(f"{label:9s}  {ari.mean():.3f} ± {ari.std():.3f}     "
              f"{nmi.mean():.3f} ± {nmi.std():.3f}     {loss}")

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {len(rows)} rows to {args.out}")
        print(f'next: python -m eval.compare {args.out} --experiment {EXPERIMENT} '
              f'--dataset {DATASET} --ours "{OURS}[hodge]"')


if __name__ == "__main__":
    main()
