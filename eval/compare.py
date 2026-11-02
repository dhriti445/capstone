"""Compare our model with baselines using the shared results files.

Owner: Deeptha S (P1). Reads results/<experiment_id>.csv, written by
eval/niches.py results_rows(), with columns

    experiment_id, dataset, sample_id, method, mode, seed, fold, metric, value

A method is named  method  or  method[mode]  (for example
"hodge-scnn[hodge]", "hodge-scnn[nosplit]", "leiden", "banksy").

How the comparison works (fixed in advance, see Experiment Plan section 7):
  * The unit is a section (sample_id). Seeds and folds are averaged
    within a section first, so seeds are not counted as extra samples.
  * Ours is compared with every other method on the same sections only
    (paired). Difference = ours - other, so positive means we are ahead.
  * 95% CI of the mean difference: bootstrap over sections (10,000 resamples).
  * p-value: two-sided Wilcoxon signed-rank on the paired differences,
    Holm-corrected within each group (baselines, ablations).
  * Groups: "ablation" = same method as ours with a different mode
    (hodge-scnn[grad], [nosplit], ...); "baseline" = any other method.

Verdict for one comparison:
  BEATS   CI of the difference is above zero and Holm p < 0.05
  LEADS   CI is above zero but Holm p >= 0.05
  PARITY  CI includes zero (no clear difference)
  TRAILS  CI is below zero (ours is behind)

Usage:
  python -m eval.compare results/ --experiment E3 --dataset DLPFC \
      --ours "hodge-scnn[hodge]" --metric ari
  python -m eval.compare results/E3.csv --experiment E3 --dataset HER2ST \
      --ours "hodge-scnn[hodge]" --md-out docs/SCOREBOARD_E3_HER2ST.md
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REQUIRED = ["experiment_id", "dataset", "sample_id", "method", "mode", "seed",
            "fold", "metric", "value"]
NO_MODE = {"", "-", "nan", "none"}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_results(path):
    """Read one results CSV, or every *.csv in a folder. Adds a 'name' column."""
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        paths = [path]
    if not paths:
        raise FileNotFoundError(f"no .csv files found at {path}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"results file is missing columns: {missing}")
    df["mode"] = df["mode"].fillna("").astype(str)
    no_mode = df["mode"].str.strip().str.lower().isin(NO_MODE)
    df["name"] = np.where(no_mode, df["method"].astype(str),
                          df["method"].astype(str) + "[" + df["mode"] + "]")
    return df


def method_of(name):
    """'hodge-scnn[grad]' -> 'hodge-scnn'."""
    return name.split("[")[0]


def sample_table(df, experiment_id, dataset, metric="ari"):
    """Sections x methods table of seed-averaged scores."""
    d = df[(df["experiment_id"] == experiment_id) & (df["dataset"] == dataset)
           & (df["metric"] == metric)]
    if d.empty:
        raise ValueError(f"no rows for experiment={experiment_id} "
                         f"dataset={dataset} metric={metric}")
    return d.groupby(["sample_id", "name"])["value"].mean().unstack("name")


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def paired_bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    """CI of the mean of `diff`, resampling sections with replacement."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def _verdict(ci_lo, ci_hi, p_holm):
    if ci_lo > 0:
        return "BEATS" if p_holm < 0.05 else "LEADS"
    if ci_hi < 0:
        return "TRAILS"
    return "PARITY"


def compare(table, ours, n_boot=10000, seed=0):
    """One row per other method: paired difference, CI, Wilcoxon, Holm, verdict."""
    if ours not in table.columns:
        raise KeyError(f"'{ours}' not in results; have: {list(table.columns)}")
    rows = []
    for name in table.columns:
        if name == ours:
            continue
        pair = table[[ours, name]].dropna()
        n = len(pair)
        row = {"method": name, "n": n,
               "group": "ablation" if method_of(name) == method_of(ours)
               else "baseline"}
        if n < 2:
            row.update(mean=np.nan, diff=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                       wins=0, p=1.0)
        else:
            diff = (pair[ours] - pair[name]).to_numpy()
            lo, hi = paired_bootstrap_ci(diff, n_boot=n_boot, seed=seed)
            if np.all(diff == 0):  # identical scores: Wilcoxon is undefined
                p = 1.0
            else:
                p = float(wilcoxon(diff).pvalue)
                p = p if np.isfinite(p) else 1.0
            row.update(mean=float(pair[name].mean()), diff=float(diff.mean()),
                       ci_lo=lo, ci_hi=hi, wins=int((diff > 0).sum()), p=p)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for g in ("baseline", "ablation"):
        m = out["group"] == g
        if m.any():
            out.loc[m, "p_holm"] = holm(out.loc[m, "p"].fillna(1.0).to_numpy())
    out["verdict"] = [
        "n/a" if r.n < 2 else _verdict(r.ci_lo, r.ci_hi, r.p_holm)
        for r in out.itertuples()
    ]
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def headline(table, comp, ours, metric="ari"):
    """One sentence on where we stand against the best external baseline."""
    ours_mean = table[ours].mean()
    ext = comp[(comp["group"] == "baseline") & (comp["n"] >= 2)]
    if ext.empty:
        return f"No external baselines with results yet (ours: {ours_mean:.3f})."
    b = ext.sort_values("mean", ascending=False).iloc[0]
    return (f"Best baseline: {b['method']} ({metric.upper()} {b['mean']:.3f}). "
            f"Ours: {ours_mean:.3f}. Difference {b['diff']:+.3f} "
            f"[95% CI {b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]; ours is ahead on "
            f"{b['wins']} of {b['n']} sections. Verdict: {b['verdict']}.")


def scoreboard_rows(table, comp, ours):
    cmp_by_name = comp.set_index("method")
    order = table.mean().sort_values(ascending=False).index
    rows = []
    for name in order:
        col = table[name].dropna()
        sd = col.std(ddof=1) if len(col) > 1 else np.nan
        mean_sd = f"{col.mean():.3f} ± {sd:.3f}" if len(col) > 1 else f"{col.mean():.3f}"
        if name == ours:
            rows.append([name + " (ours)", "ours", mean_sd, str(len(col)),
                         "-", "-", "-", "-"])
            continue
        c = cmp_by_name.loc[name]
        rows.append([
            name, c["group"], mean_sd, str(int(c["n"])),
            f"{c['diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]",
            f"{int(c['wins'])}/{int(c['n'])}",
            f"{c['p_holm']:.3f}", c["verdict"],
        ])
    return rows


HEADER = ["method", "group", "mean ± sd", "n", "ours − this [95% CI]",
          "ours ahead", "Holm p", "verdict"]


def format_scoreboard(table, comp, ours, metric="ari", title="", markdown=False):
    rows = scoreboard_rows(table, comp, ours)
    lines = []
    if title:
        lines.append(f"## {title}" if markdown else title)
        lines.append("")
    if markdown:
        lines.append("| " + " | ".join(HEADER) + " |")
        lines.append("|" + "|".join(["---"] * len(HEADER)) + "|")
        lines += ["| " + " | ".join(r) + " |" for r in rows]
    else:
        widths = [max(len(HEADER[i]), *(len(r[i]) for r in rows))
                  for i in range(len(HEADER))]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        lines.append(fmt.format(*HEADER))
        lines.append(fmt.format(*["-" * w for w in widths]))
        lines += [fmt.format(*r) for r in rows]
    lines += ["", headline(table, comp, ours, metric), ""]
    notes = [
        f"{metric.upper()} per section, averaged over seeds; mean ± sd is across sections.",
        "'ours − this' is paired by section; CI is a 95% bootstrap over sections.",
        "Holm p: Wilcoxon signed-rank, corrected within baselines and within ablations.",
        "BEATS = CI above 0 and Holm p < 0.05; LEADS = CI above 0 only; "
        "PARITY = CI includes 0; TRAILS = CI below 0.",
    ]
    if (comp["n"] < 6).any():
        notes.append("Some comparisons have n < 6 sections: Wilcoxon cannot reach "
                     "p < 0.05 at that size, so treat those as indicative only.")
    lines += [("- " + n) if markdown else n for n in notes]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", help="results CSV or a folder of CSVs")
    ap.add_argument("--experiment", required=True, help="e.g. E3")
    ap.add_argument("--dataset", required=True, help="e.g. DLPFC")
    ap.add_argument("--ours", required=True, help='e.g. "hodge-scnn[hodge]"')
    ap.add_argument("--metric", default="ari", choices=["ari", "nmi"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--md-out", help="also write the scoreboard as Markdown here")
    args = ap.parse_args()

    df = load_results(args.results)
    table = sample_table(df, args.experiment, args.dataset, args.metric)
    comp = compare(table, args.ours, n_boot=args.n_boot)
    title = f"{args.experiment} on {args.dataset}: {args.metric.upper()}"
    print(format_scoreboard(table, comp, args.ours, args.metric, title))
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(format_scoreboard(table, comp, args.ours, args.metric,
                                      title, markdown=True) + "\n")
        print(f"\nwrote {args.md_out}")


if __name__ == "__main__":
    main()
