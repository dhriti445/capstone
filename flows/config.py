"""Paths and settings for P2 (ligand-receptor flows + visualization).

Edit the paths in this file only. Every script imports from here.
"""
from pathlib import Path

# ---- Paths (change these to match your laptop) ---------------------------
ROOT = Path(__file__).resolve().parent
HEST_DIR = ROOT.parent.parent / "hest_data"                    # where `hf download` put the data
COMPLEX_DIR = ROOT.parent.parent / "TENX53_complex"            # Dhriti's edges.npy / triangles.npy / spot_ids.npy
SAMPLE = "TENX53"

H5AD_IN = HEST_DIR / "st" / f"{SAMPLE}.h5ad"            # raw HEST file (input)
OUT_DIR = ROOT / "outputs"

# Demo mode: `P2_DEMO=1 python 0X_....py` runs every step on the fake tissue
# made by 00_make_demo_data.py, so you can check the scripts before the real data.
import os
if os.environ.get("P2_DEMO") == "1":
    SAMPLE = "DEMO"
    OUT_DIR = ROOT / "outputs_demo"
    H5AD_IN = OUT_DIR / "input" / "DEMO.h5ad"
    COMPLEX_DIR = OUT_DIR / "input" / "DEMO_complex"

FLOWS_PRODUCT = OUT_DIR / f"{SAMPLE}_flows_product.h5ad"   # baseline flows  -> hand to Dhriti
FLOWS_COMMOT = OUT_DIR / f"{SAMPLE}_flows_commot.h5ad"     # main flows      -> hand to Dhriti
FIG_DIR = OUT_DIR / "figures"

# ---- Expression preprocessing (agreed in the architecture doc) -----------
TARGET_SUM = 1e4         # normalise each spot to 10,000 counts, then log1p
SCALE_QUANTILE = 0.99    # scale each gene to 0-1 by its 99th percentile (flows only)

# ---- Ligand-receptor pair selection --------------------------------------
N_PAIRS = 10             # how many L-R pairs to keep
# Pairs with known roles in breast cancer, always kept if expressed enough.
# Names must match LIANA's consensus resource (complex subunits joined by '_').
CURATED_PAIRS = [
    ("SPP1", "CD44"),          # osteopontin, invasion / metastasis
    ("CXCL12", "CXCR4"),       # chemokine axis, metastasis homing
    ("MDK", "SDC1"),           # midkine (top COMMOT pathway in TENX53)
    ("TGFB1", "TGFBR1_TGFBR2"),
    ("VEGFA", "KDR"),          # angiogenesis
    ("JAG1", "NOTCH1"),
]
MIN_EXPR_FRAC = 0.10     # each subunit expressed in >= 10% of spots

# ---- COMMOT ----------------------------------------------------------------
# Signalling range as a multiple of the median nearest-neighbour spot distance.
# Visium spots are 100 um apart; 1.5x keeps COMMOT close to the complex edges.
COMMOT_RANGE_FACTOR = 1.5
COMMOT_SIGNALING_TYPE = "Secreted Signaling"   # or None for all types (much slower)

# ---- Controls --------------------------------------------------------------
N_PERMUTATIONS = 100
SEED = 0
