"""Tests for eval/compare.py (owner: Deeptha S, P1)."""
import numpy as np
import pandas as pd
import pytest

from eval.compare import (holm, paired_bootstrap_ci, load_results, sample_table,
                          compare, format_scoreboard, headline, method_of)
from eval.niches import results_rows

OURS = "hodge-scnn[hodge]"


def make_df(effects, n=10, seed=0, noise=0.02, dataset="DLPFC"):
    """Results in the shared format. effects: name -> (method, mode, mean ARI)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 0.6, size=n)  # section difficulty, shared by methods
    rows = []
    for name, (method, mode, shift) in effects.items():
        for i in range(n):
            for s in (0, 1):
                ari = base[i] + shift + noise * rng.normal()
                rows += results_rows("E3", dataset, f"sec{i}", method, mode, s, 0,
                                     {"ari": ari, "nmi": ari})
    return pd.DataFrame(rows)


def prep(df):
    """Same name handling as load_results, without touching disk."""
    df = df.copy()
    df["mode"] = df["mode"].fillna("").astype(str)
    df["name"] = np.where(df["mode"] == "", df["method"],
                          df["method"] + "[" + df["mode"] + "]")
    return df


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == pytest.approx([0.5])
    assert (holm([0.4, 0.9]) <= 1.0).all()


def test_bootstrap_ci_contains_mean_and_is_ordered():
    diff = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
    lo, hi = paired_bootstrap_ci(diff, n_boot=2000)
    assert lo < diff.mean() < hi and lo > 0


def test_method_of():
    assert method_of("hodge-scnn[grad]") == "hodge-scnn"
    assert method_of("leiden") == "leiden"


def test_clear_win_and_clear_loss_and_parity():
    df = prep(make_df({
        OURS: ("hodge-scnn", "hodge", 0.20),
        "weak": ("weak", "", 0.00),          # we are 0.20 ahead
        "strong": ("strong", "", 0.40),      # we are 0.20 behind
        "same": ("same", "", 0.20),          # no real difference
    }))
    comp = compare(sample_table(df, "E3", "DLPFC"), OURS).set_index("method")
    assert comp.loc["weak", "verdict"] == "BEATS"
    assert comp.loc["strong", "verdict"] == "TRAILS"
    assert comp.loc["same", "verdict"] == "PARITY"
    assert comp.loc["weak", "wins"] == 10
    assert comp.loc["weak", "diff"] == pytest.approx(0.20, abs=0.03)


def test_groups_baseline_vs_ablation():
    df = prep(make_df({
        OURS: ("hodge-scnn", "hodge", 0.20),
        "hodge-scnn[nosplit]": ("hodge-scnn", "nosplit", 0.10),
        "leiden": ("leiden", "", 0.00),
    }))
    comp = compare(sample_table(df, "E3", "DLPFC"), OURS).set_index("method")
    assert comp.loc["hodge-scnn[nosplit]", "group"] == "ablation"
    assert comp.loc["leiden", "group"] == "baseline"


def test_seeds_are_not_counted_as_samples():
    df = prep(make_df({OURS: ("hodge-scnn", "hodge", 0.2),
                       "leiden": ("leiden", "", 0.0)}, n=7))
    table = sample_table(df, "E3", "DLPFC")
    assert table.shape == (7, 2)          # 7 sections, not 14 seed rows
    assert compare(table, OURS).loc[0, "n"] == 7


def test_pairing_drops_sections_a_method_did_not_run():
    df = prep(make_df({OURS: ("hodge-scnn", "hodge", 0.2),
                       "leiden": ("leiden", "", 0.0)}, n=8))
    df = df[~((df["name"] == "leiden") & df["sample_id"].isin(["sec0", "sec1"]))]
    comp = compare(sample_table(df, "E3", "DLPFC"), OURS)
    assert comp.loc[0, "n"] == 6


def test_identical_scores_do_not_crash():
    df = prep(make_df({OURS: ("hodge-scnn", "hodge", 0.2),
                       "copy": ("copy", "", 0.2)}, noise=0.0))
    comp = compare(sample_table(df, "E3", "DLPFC"), OURS).set_index("method")
    assert comp.loc["copy", "verdict"] == "PARITY"
    assert comp.loc["copy", "p"] == 1.0


def test_small_n_cannot_be_beats():
    """With 4 sections Wilcoxon cannot reach p < 0.05, so at best LEADS."""
    df = prep(make_df({OURS: ("hodge-scnn", "hodge", 0.3),
                       "weak": ("weak", "", 0.0)}, n=4))
    comp = compare(sample_table(df, "E3", "DLPFC"), OURS)
    assert comp.loc[0, "verdict"] in ("LEADS", "PARITY")
    assert comp.loc[0, "verdict"] != "BEATS"
    text = format_scoreboard(sample_table(df, "E3", "DLPFC"), comp, OURS)
    assert "n < 6" in text


def test_missing_ours_raises():
    df = prep(make_df({"leiden": ("leiden", "", 0.0)}))
    with pytest.raises(KeyError):
        compare(sample_table(df, "E3", "DLPFC"), OURS)


def test_no_rows_raises():
    df = prep(make_df({OURS: ("hodge-scnn", "hodge", 0.2)}))
    with pytest.raises(ValueError):
        sample_table(df, "E3", "HER2ST")


def test_scoreboard_text_and_markdown(tmp_path):
    df = prep(make_df({
        OURS: ("hodge-scnn", "hodge", 0.20),
        "hodge-scnn[nosplit]": ("hodge-scnn", "nosplit", 0.10),
        "leiden": ("leiden", "", 0.0),
        "banksy": ("banksy", "", 0.1),
    }))
    table = sample_table(df, "E3", "DLPFC")
    comp = compare(table, OURS)
    txt = format_scoreboard(table, comp, OURS, "ari", "E3 on DLPFC: ARI")
    md = format_scoreboard(table, comp, OURS, "ari", "E3 on DLPFC: ARI", markdown=True)
    assert "(ours)" in txt and "banksy" in txt and "verdict" in txt
    assert md.startswith("## E3 on DLPFC") and "| method |" in md
    assert "Best baseline: banksy" in headline(table, comp, OURS)


def test_load_results_roundtrip(tmp_path):
    df = make_df({OURS: ("hodge-scnn", "hodge", 0.2),
                  "leiden": ("leiden", "", 0.0)}, n=3)
    p = tmp_path / "E3.csv"
    df.to_csv(p, index=False)
    back = load_results(str(p))
    assert set(back["name"]) == {OURS, "leiden"}
    assert set(load_results(str(tmp_path))["name"]) == {OURS, "leiden"}  # folder
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_results(str(bad))
