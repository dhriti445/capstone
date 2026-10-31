
from pathlib import Path
from datetime import datetime
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

try:
    import anndata as ad
    import scanpy as sc
except ImportError:
    print("Install dependencies with:")
    print("python3 -m pip install anndata scanpy pandas numpy scipy matplotlib")
    sys.exit(1)


# ============================================================
# CAPSTONE MASTER PREPROCESSING PIPELINE
# Sample: TENX53 — Human Breast Cancer, Visium
# ============================================================

ROOT = Path.cwd()
DATA_DIR = ROOT / "hest_samples"
OUT = ROOT / "processed"
OUT.mkdir(parents=True, exist_ok=True)

STARTER_PATH = ROOT / "starter.h5ad"
QC_METRICS_PATH = OUT / "TENX53_qc_metrics.csv"
QC_FLAGS_PATH = OUT / "TENX53_qc_flags.csv"
QC_SUMMARY_PATH = OUT / "TENX53_qc_flag_summary.csv"
QC_PLOT_PATH = OUT / "TENX53_spatial_qc.png"
REPORT_PATH = OUT / "TENX53_master_preprocessing_report.txt"
STATUS_PATH = OUT / "capstone_status.json"

# Provisional screening thresholds. These are NOT universal
# biological exclusion criteria.
MIN_COUNTS = 500
MIN_GENES = 200
MAX_MITO_PCT = 15.0

# Conservative default: never remove spots automatically.
REMOVE_ZERO_COUNT_SPOTS = False

INPUT_CANDIDATES = [
    OUT / "TENX53_processed.h5ad",
    OUT / "TENX53_hvg_v3.h5ad",
    DATA_DIR / "st" / "TENX53.h5ad",
]


def record(status, key, done, detail):
    status[key] = {
        "status": "DONE" if done else "PENDING",
        "detail": str(detail),
    }
    print(f"[{'DONE' if done else 'PENDING'}] {key}: {detail}")


def row_sums(matrix):
    return np.asarray(matrix.sum(axis=1)).ravel().astype(float)


def column_sums(matrix):
    return np.asarray(matrix.sum(axis=0)).ravel().astype(float)


def calculate_qc(adata, count_matrix):
    """Calculate spot QC metrics from count-like expression values."""

    total_counts = row_sums(count_matrix)

    if sparse.issparse(count_matrix):
        detected_genes = np.asarray(
            (count_matrix > 0).sum(axis=1)
        ).ravel()
    else:
        detected_genes = np.count_nonzero(
            np.asarray(count_matrix), axis=1
        )

    names = pd.Index(adata.var_names.astype(str))
    mito_mask = names.str.upper().str.startswith("MT-")

    if mito_mask.any():
        mito_matrix = count_matrix[:, mito_mask]
        mito_counts = row_sums(mito_matrix)

        mito_pct = np.divide(
            100.0 * mito_counts,
            total_counts,
            out=np.full(len(total_counts), np.nan),
            where=total_counts > 0,
        )
    else:
        mito_pct = np.full(len(total_counts), np.nan)
        print(
            "WARNING: No MT- genes detected. "
            "Mitochondrial QC cannot be calculated reliably "
            "from these gene names."
        )

    return (
        total_counts,
        detected_genes.astype(int),
        mito_pct,
        mito_mask,
    )


def main():
    print("\nCAPSTONE MASTER PREPROCESSING")
    print("=" * 65)
    print(f"Working directory: {ROOT}")

    status = {}
    notes = []

    # --------------------------------------------------------
    # 1. Locate and load data
    # --------------------------------------------------------
    source = next(
        (p for p in INPUT_CANDIDATES if p.exists()),
        None,
    )

    if source is None:
        raise FileNotFoundError(
            "TENX53 data not found. Expected one of:\n"
            + "\n".join(str(p) for p in INPUT_CANDIDATES)
        )

    print(f"Input: {source}")
    adata = sc.read_h5ad(source)

    record(
        status,
        "Dataset load",
        True,
        f"{adata.n_obs} spots x {adata.n_vars} features",
    )

    # --------------------------------------------------------
    # 2. Determine whether expression is already processed
    # --------------------------------------------------------
    # Existing processed files have a counts layer and normalized
    # log1p expression in X. Preserve them without normalizing twice.
    already_processed = (
        "counts" in adata.layers
        and source.name != "TENX53.h5ad"
    )

    if "counts" in adata.layers:
        count_matrix = adata.layers["counts"]
    else:
        count_matrix = adata.X.copy()

    # For raw input, remove zero-count genes and normalize once.
    # For an existing processed file, retain the existing matrix.
    if not already_processed:
        gene_totals = column_sums(count_matrix)
        keep_genes = gene_totals > 0
        removed_genes = int((~keep_genes).sum())

        if removed_genes:
            adata = adata[:, keep_genes].copy()
            count_matrix = (
                count_matrix[:, keep_genes]
                if sparse.issparse(count_matrix)
                else np.asarray(count_matrix)[:, keep_genes]
            )

        # Preserve the count-like matrix before normalization.
        adata.layers["counts"] = (
            count_matrix.copy()
            if sparse.issparse(count_matrix)
            else np.asarray(count_matrix).copy()
        )

        # Compute metrics before transformation.
        counts_total_before = row_sums(count_matrix)

        # Optional zero-count spot filtering is disabled by default.
        if REMOVE_ZERO_COUNT_SPOTS:
            keep_spots = counts_total_before > 0
            adata = adata[keep_spots].copy()
            count_matrix = adata.layers["counts"]

        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)

        notes.append(
            f"Raw input path: removed {removed_genes} zero-count genes; "
            "normalized to 10,000 counts per spot and applied log1p."
        )
    else:
        notes.append(
            "Existing processed input detected: expression was preserved "
            "to avoid double normalization."
        )

    # Use the retained counts layer for QC.
    count_matrix = adata.layers["counts"]

    # --------------------------------------------------------
    # 3. Validate coordinates
    # --------------------------------------------------------
    spatial_ok = (
        "spatial" in adata.obsm
        and adata.obsm["spatial"].shape == (adata.n_obs, 2)
        and np.isfinite(adata.obsm["spatial"]).all()
    )

    record(
        status,
        "Spatial coordinates",
        spatial_ok,
        (
            f"{adata.n_obs} valid coordinate pairs"
            if spatial_ok
            else "Missing or invalid obsm['spatial']",
        ),
    )

    if not spatial_ok:
        raise ValueError(
            "Valid spatial coordinates are required for starter.h5ad."
        )

    # --------------------------------------------------------
    # 4. Compute QC metrics from counts
    # --------------------------------------------------------
    total_counts, detected_genes, mito_pct, mito_mask = calculate_qc(
        adata, count_matrix
    )

    low_counts = total_counts < MIN_COUNTS
    low_genes = detected_genes < MIN_GENES
    high_mito = np.isfinite(mito_pct) & (mito_pct > MAX_MITO_PCT)
    zero_counts = total_counts <= 0

    # Missing mitochondrial annotation is not treated as a pass.
    mito_unavailable = not mito_mask.any()

    any_flag = low_counts | low_genes | high_mito | zero_counts

    qc = pd.DataFrame({
        "spot_id": adata.obs_names.astype(str),
        "total_counts": total_counts,
        "n_genes_detected": detected_genes,
        "pct_mito": mito_pct,
        "flag_low_counts": low_counts,
        "flag_low_genes": low_genes,
        "flag_high_mito": high_mito,
        "flag_zero_counts": zero_counts,
        "flag_any_qc": any_flag,
    })

    qc.to_csv(QC_METRICS_PATH, index=False)
    qc.loc[any_flag].to_csv(QC_FLAGS_PATH, index=False)

    # Counts of individual flags and their combinations.
    flag_summary = pd.DataFrame([
        {"metric": "Total spots", "count": len(qc)},
        {"metric": f"Counts < {MIN_COUNTS}", "count": int(low_counts.sum())},
        {"metric": f"Detected genes < {MIN_GENES}", "count": int(low_genes.sum())},
        {"metric": f"Mitochondrial % > {MAX_MITO_PCT}", "count": int(high_mito.sum())},
        {"metric": "Zero-count spots", "count": int(zero_counts.sum())},
        {"metric": "Flagged by any available rule", "count": int(any_flag.sum())},
        {"metric": "Not flagged by any available rule", "count": int((~any_flag).sum())},
    ])

    flag_summary.to_csv(QC_SUMMARY_PATH, index=False)

    record(
        status,
        "QC metrics and flag overlap",
        True,
        f"Saved metrics and {int(any_flag.sum())} flagged spots",
    )

    print("\nQC SUMMARY")
    print("-" * 50)
    print(f"Spots: {adata.n_obs}")
    print(f"Genes/features: {adata.n_vars}")
    print(f"Median counts: {np.median(total_counts):.2f}")
    print(f"Median detected genes: {np.median(detected_genes):.0f}")
    print(f"Median mitochondrial percentage: {np.nanmedian(mito_pct) if np.isfinite(mito_pct).any() else float('nan'):.3f}%")
    print(f"Counts below {MIN_COUNTS}: {low_counts.sum()}")
    print(f"Genes below {MIN_GENES}: {low_genes.sum()}")
    print(f"Mitochondrial percentage above {MAX_MITO_PCT}%: {high_mito.sum()}")
    print(f"Zero-count spots: {zero_counts.sum()}")
    print(f"Flagged by at least one rule: {any_flag.sum()}")

    if mito_unavailable:
        notes.append(
            "No MT- genes were identified; mitochondrial fraction "
            "cannot be interpreted as passing the QC threshold."
        )

    # --------------------------------------------------------
    # 5. Add QC metrics and flags to AnnData
    # --------------------------------------------------------
    adata.obs["qc_total_counts_recomputed"] = total_counts
    adata.obs["qc_n_genes_recomputed"] = detected_genes
    adata.obs["qc_pct_mito_recomputed"] = mito_pct
    adata.obs["qc_flag_low_counts"] = low_counts
    adata.obs["qc_flag_low_genes"] = low_genes
    adata.obs["qc_flag_high_mito"] = high_mito
    adata.obs["qc_flag_zero_counts"] = zero_counts
    adata.obs["qc_flag_any"] = any_flag

    adata.uns["master_preprocessing"] = {
        "source_file": str(source),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts_threshold": MIN_COUNTS,
        "detected_genes_threshold": MIN_GENES,
        "mitochondrial_pct_threshold": MAX_MITO_PCT,
        "thresholds_are_provisional_screening_rules": True,
        "spots_removed_by_qc": 0,
        "normalization": (
            "Preserved existing normalized log1p X"
            if already_processed
            else "Total-count normalization to 10000 followed by log1p"
        ),
    }

    # --------------------------------------------------------
    # 6. Save spatial QC plots
    # --------------------------------------------------------
    coords = np.asarray(adata.obsm["spatial"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    plots = [
        (np.log1p(total_counts), "Log1p total counts"),
        (detected_genes, "Detected genes"),
        (mito_pct, "Mitochondrial %"),
    ]

    for ax, (values, title) in zip(axes, plots):
        sc_plot = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=8,
        )
        ax.set_title(title)
        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")
        ax.invert_yaxis()
        fig.colorbar(sc_plot, ax=ax)

    fig.tight_layout()
    fig.savefig(QC_PLOT_PATH, dpi=180)
    plt.close(fig)

    record(status, "Spatial QC plot", True, QC_PLOT_PATH.name)

    # --------------------------------------------------------
    # 7. Save and reopen starter.h5ad
    # --------------------------------------------------------
    # Do not delete flagged spots automatically.
    adata.write_h5ad(STARTER_PATH, compression="gzip")

    check = sc.read_h5ad(STARTER_PATH)

    starter_ok = (
        check.n_obs == adata.n_obs
        and check.n_vars == adata.n_vars
        and "spatial" in check.obsm
        and check.obsm["spatial"].shape == (check.n_obs, 2)
        and "counts" in check.layers
    )

    record(
        status,
        "starter.h5ad handoff",
        starter_ok,
        f"Reopened successfully with shape {check.shape}",
    )

    # --------------------------------------------------------
    # 8. Audit remaining project deliverables
    # --------------------------------------------------------
    patterns = {
        "HER2ST": ["*HER2ST*.h5ad", "*her2st*.h5ad"],
        "TCGA_BRCA": ["*TCGA*BRCA*.h5ad", "*tcga*brca*.h5ad"],
        "DLPFC": ["*DLPFC*.h5ad", "*dlpfc*.h5ad"],
    }

    roots = [ROOT / "data", DATA_DIR, OUT]

    for dataset, file_patterns in patterns.items():
        matches = set()

        for folder in roots:
            if folder.exists():
                for pattern in file_patterns:
                    matches.update(folder.rglob(pattern))

        record(
            status,
            f"Additional dataset: {dataset}",
            bool(matches),
            (
                f"Found {len(matches)} matching file(s)"
                if matches
                else "Not found by this filename audit"
            ),
        )

    splits_file = ROOT / "splits.json"
    record(
        status,
        "Patient-level splits.json",
        splits_file.exists(),
        (
            "File exists; patient-level separation must be checked"
            if splits_file.exists()
            else "Not found; requires valid patient/sample identifiers"
        ),
    )

    cell_fields = [
        col for col in check.obs.columns
        if any(term in col.lower() for term in [
            "cell_type", "celltype", "proportion", "label", "cluster"
        ])
    ]

    record(
        status,
        "Cell-type proportions/labels",
        bool(cell_fields),
        (
            ", ".join(cell_fields)
            if cell_fields
            else "No likely cell-type fields found in obs"
        ),
    )

    # --------------------------------------------------------
    # 9. Write report and machine-readable status
    # --------------------------------------------------------
    report = [
        "TENX53 MASTER PREPROCESSING REPORT",
        "=" * 60,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: {source}",
        f"Starter: {STARTER_PATH}",
        f"Final shape: {check.n_obs} spots x {check.n_vars} features",
        "",
        "QC THRESHOLDS (PROVISIONAL)",
        f"Low counts: < {MIN_COUNTS}",
        f"Low detected genes: < {MIN_GENES}",
        f"High mitochondrial fraction: > {MAX_MITO_PCT}%",
        "Zero counts: flagged",
        "",
        "QC COUNTS",
        f"Low-count spots: {int(low_counts.sum())}",
        f"Low-gene spots: {int(low_genes.sum())}",
        f"High-mito spots: {int(high_mito.sum())}",
        f"Zero-count spots: {int(zero_counts.sum())}",
        f"Flagged by any rule: {int(any_flag.sum())}",
        "Spots removed automatically: 0",
        "",
        "OUTPUT FILES",
        str(STARTER_PATH),
        str(QC_METRICS_PATH),
        str(QC_FLAGS_PATH),
        str(QC_SUMMARY_PATH),
        str(QC_PLOT_PATH),
        "",
        "PROJECT STATUS",
    ]

    for key, item in status.items():
        report.append(
            f"[{item['status']}] {key}: {item['detail']}"
        )

    report.extend(["", "NOTES"])
    report.extend(notes or ["No additional notes."])
    report.extend([
        "",
        "QC interpretation: thresholds are screening rules, not validated",
        "automatic exclusion criteria. Review flagged spots and tissue images",
        "before deciding whether to remove observations.",
    ])

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    with STATUS_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input": str(source),
            "starter": str(STARTER_PATH),
            "shape": [int(check.n_obs), int(check.n_vars)],
            "qc_thresholds": {
                "min_counts": MIN_COUNTS,
                "min_genes": MIN_GENES,
                "max_mito_pct": MAX_MITO_PCT,
                "remove_spots_automatically": REMOVE_ZERO_COUNT_SPOTS,
            },
            "status": status,
            "notes": notes,
        }, f, indent=2)

    print("\n" + "=" * 65)
    print("MASTER PREPROCESSING COMPLETE")
    print("=" * 65)
    print(f"Starter file: {STARTER_PATH}")
    print(f"QC metrics: {QC_METRICS_PATH}")
    print(f"Flagged spots: {QC_FLAGS_PATH}")
    print(f"Flag summary: {QC_SUMMARY_PATH}")
    print(f"QC plot: {QC_PLOT_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Status JSON: {STATUS_PATH}")
    print("\nNo spots were removed automatically.")
    print("Review the report and flagged spots before final QC decisions.")


if __name__ == "__main__":
    main()
