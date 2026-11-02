"""Download the 12 DLPFC (Maynard 2021) sections and save them in our layout.

Usage:
  python scripts/get_dlpfc.py                 # download from Hugging Face
  python scripts/get_dlpfc.py --local DIR     # use files DIR/DLPFC_<id>.h5ad

Output (one file per section) in data/dlpfc/<id>.h5ad with
  obs['layer']   expert label: L1..L6 or WM, NaN if missing
  obs['donor']   g1 / g2 / g3 (the 3 donors; for leave-one-donor-out splits)
  obs['sample_id'], obs['array_row'], obs['array_col'], obs['in_tissue']
  obsm['spatial'], layers['counts'], X = log1p(normalised counts)
and data/dlpfc/dlpfc_inventory.csv summarising all sections.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

IDS = ["151507", "151508", "151509", "151510",
       "151669", "151670", "151671", "151672",
       "151673", "151674", "151675", "151676"]
DONOR = {**{i: "g1" for i in IDS[0:4]},
         **{i: "g2" for i in IDS[4:8]},
         **{i: "g3" for i in IDS[8:12]}}
OUT = os.path.join("data", "dlpfc")
LABEL_CANDIDATES = ["cluster", "layer", "Layer", "layer_guess", "ground_truth",
                    "Ground Truth", "groundtruth", "GroundTruth", "label"]


def norm_label(v):
    """Return 'L1'..'L6', 'WM', np.nan (missing) or None (unrecognised)."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s.lower() in ("", "nan", "none", "na", "unknown"):
        return np.nan
    m = re.fullmatch(r"(?i)l(?:ayer)?[\s_\-]*([1-6])", s)
    if m:
        return "L" + m.group(1)
    if re.fullmatch(r"(?i)(wm|white[\s_\-]*matter)", s):
        return "WM"
    return None


def find_label_column(adata):
    for c in LABEL_CANDIDATES:
        if c in adata.obs.columns:
            return c
    for c in adata.obs.columns:
        if "layer" in c.lower() or "truth" in c.lower():
            return c
    return None


def looks_like_counts(X):
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    data = data[data != 0][:5000]
    return len(data) > 0 and (data >= 0).all() and np.allclose(data, np.round(data))


def standardise(adata, sample_id):
    adata = adata.copy()
    adata.var_names_make_unique()
    if "in_tissue" in adata.obs:
        adata = adata[adata.obs["in_tissue"] == 1].copy()
    else:
        adata.obs["in_tissue"] = 1
    for need in ("array_row", "array_col"):
        if need not in adata.obs:
            raise ValueError(f"{sample_id}: obs has no '{need}'; columns: "
                             f"{list(adata.obs.columns)}")
    if "spatial" not in adata.obsm:
        raise ValueError(f"{sample_id}: obsm has no 'spatial'")

    col = find_label_column(adata)
    unrec = 0
    if col is None:
        raise ValueError(f"{sample_id}: no layer-label column; columns: "
                         f"{list(adata.obs.columns)}")
    vals = adata.obs[col].astype(object).tolist()
    mapped = [norm_label(v) for v in vals]
    unrec = sum(1 for v in mapped if v is None)
    mapped = [np.nan if v is None else v for v in mapped]
    adata.obs["layer"] = pd.Categorical(
        mapped, categories=["L1", "L2", "L3", "L4", "L5", "L6", "WM"])
    adata.obs["donor"] = DONOR[sample_id]
    adata.obs["sample_id"] = sample_id

    if "counts" not in adata.layers:
        if looks_like_counts(adata.X):
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
    return adata, col, unrec


def main():
    local = None
    if "--local" in sys.argv:
        local = sys.argv[sys.argv.index("--local") + 1]
    os.makedirs(OUT, exist_ok=True)
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass

    rows = []
    for sid in IDS:
        try:
            if local:
                path = os.path.join(local, f"DLPFC_{sid}.h5ad")
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id="han-shu/st_datasets",
                                       repo_type="dataset",
                                       filename=f"DLPFC/DLPFC_{sid}.h5ad")
            a, col, unrec = standardise(sc.read_h5ad(path), sid)
            a.write_h5ad(os.path.join(OUT, f"{sid}.h5ad"), compression="gzip")
            lab = a.obs["layer"]
            rows.append({"sample_id": sid, "donor": DONOR[sid],
                         "spots": a.n_obs, "genes": a.n_vars,
                         "label_column_used": col,
                         "labelled": int(lab.notna().sum()),
                         "missing_label": int(lab.isna().sum()),
                         "unrecognised_label_values": unrec,
                         "layers_present": ",".join(sorted(
                             lab.dropna().astype(str).unique())),
                         "has_counts_layer": "counts" in a.layers})
            print("ok", sid, a.n_obs, "spots")
        except Exception as ex:
            print("FAILED", sid, type(ex).__name__, str(ex)[:200])
            rows.append({"sample_id": sid, "donor": DONOR[sid],
                         "error": str(ex)[:200]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "dlpfc_inventory.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
