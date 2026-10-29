# Hodge-Aware Topological Deep Learning for Cell-Cell Communication (Spatial Omics)

**UE24CS320A Capstone Phase-1 | Project ID PW_27_CN_08 | Guide: Dr. Chaitra N**

We build a cell complex of a tissue (spots/cells, edges, triangles), compute
ligand-receptor signalling as edge flows, and split each flow with Hodge theory
into **gradient** (source to sink), **curl** (local loops) and **harmonic**
(loops around holes) parts. A topology-aware network learns from each part to
find tissue niches and test for prognostic signal.

This README is the team's shared status board. **Update your own section
whenever you finish something**, and add a row to the changelog at the bottom.

---

## Team and ownership

| Person | Role | Owns | Main folders/files |
|---|---|---|---|
| **Deeptha S** (P1) | Model | Data contract, Hodge-SCNN, CPU baselines, evaluation harness | `model/`, `baselines/`, `eval/` |
| **Disha Lokesh** (P2) | L-R flows and visualization | LIANA+/COMMOT scoring, flows on the complex, plotting | `flows/`, `hodgecc/plots.py` |
| **Dhriti Kamani** (P3) | Hodge core | Repo/env, cell-complex builder, Hodge decomposition, synthetic validation | `hodgecc/complex.py`, `hodgecc/hodge.py`, `hodgecc/synth.py`, `tests/` |
| **S Mayur** (P4) | Data | Dataset downloads, QC table, `.h5ad` standardisation | `data/` (not in git), `scripts/data_*` |

Status key: **Done** | **In progress** | **Not started**

---

## Progress board

### Dhriti (P3): Hodge core

| Task | Status | Notes |
|---|---|---|
| Repo skeleton and environment | Done | Python 3.11, CPU torch, scanpy, squidpy, toponetx, topomodelx, gudhi |
| Cell-complex builder (`complex.py`) | Done | Delaunay mode (irregular cells) and grid mode (Visium). `B1 @ B2 = 0` and Euler checks pass |
| Hodge decomposition (`hodge.py`) | In progress | Code written; confirm `pytest` passes locally, then mark Done |
| Synthetic complexes and planted flows (`synth.py`) | In progress | Grid with holes, planted gradient/curl/harmonic flows |
| Real Visium slice into a complex (spots, edges, triangles, beta1) | Not started | `python scripts/check_visium.py <file.h5ad>` |
| Synthetic validation notebook | Not started | Error vs noise, mesh size, masked edges, beta1 vs hole count; 4-panel figure |

### Disha (P2): L-R flows and visualization

| Task | Status | Notes |
|---|---|---|
| Agree edge-flow format with Dhriti | Not started | See "Shared data format" below |
| LIANA+ L-R scoring on one DLPFC slice | Not started | |
| COMMOT spatial communication on one DLPFC slice | Not started | |
| Convert scores to edge flows `(E, C)` on Dhriti's complex | Not started | |
| Plotting: flow arrows, gradient/curl/harmonic panels (`plots.py`) | Not started | |

### Deeptha (P1): model and evaluation

| Task | Status | Notes |
|---|---|---|
| One-page data contract (tensors the model receives) | Not started | |
| Small Hodge-SCNN in TopoModelX | Not started | Train on synthetic data, then one DLPFC slice |
| CPU baselines: Leiden/BANKSY, plain GCN, Cox survival (lifelines) | Not started | |
| Evaluation harness: ARI/NMI (niches), C-index (survival) | Not started | |

### Mayur (P4): data

| Task | Status | Notes |
|---|---|---|
| Download DLPFC (spatialLIBD) and HER2ST | Not started | |
| HEST-1k: breast Visium subset only (filter metadata) | Not started | Do not download the full set |
| TCGA clinical side (TCGA-CDR survival, TCGA-BRCA tables) + 3-5 test WSIs | Not started | |
| Data inventory and QC table (samples, spots, genes, missing labels, endpoints) | Not started | |
| Save everything as `.h5ad` with a consistent layout | Not started | |

### Waiting for GPU (nobody starts these yet)

- H&E foundation-model embeddings (UNI/CONCH) on TCGA and HEST slides
- Full-scale training on all HEST breast and HER2ST samples
- Hyperparameter sweeps and the full ablation grid
- The H&E-only prediction model

---

## Shared data format (v0.1, do not change without telling the group)

| Item | Shape / type | Convention |
|---|---|---|
| `edges` | `(E, 2)` int | each row `(i, j)` with `i < j`, sorted lexicographically |
| `triangles` | `(T, 3)` int | sorted vertices `a < b < c`, sorted lexicographically |
| `B1` | `(N, E)` scipy CSR | edge `(i, j)`: -1 at row i, +1 at row j |
| `B2` | `(E, T)` scipy CSR | triangle `(a, b, c)`: +(a,b) +(b,c) -(a,c) |
| flows | `(E, C)` float | one column per L-R pair, same edge order as `edges`; positive means i to j |
| saved layout | one `.h5ad` per sample | `adata.uns['complex']` holds `edges` and `triangles`; `adata.uns['flows']` holds the `(E, C)` matrix. Rebuild `B1`/`B2` with `hodgecc.complex.build_incidence` |

Invariant: `B1 @ B2 == 0`.

### Using the code

```python
from hodgecc.complex import build_complex
from hodgecc.hodge import decompose

edges, triangles, B1, B2 = build_complex(coords)          # irregular cells
# Visium: build_complex(coords, method="grid", adata=adata)
grad, curl, harm = decompose(B1, B2, flows)                # flows is (E, C)
```

---

## Repo layout


## Setup (Windows / PowerShell)

```powershell
git clone https://github.com/dhriti445/capstone.git
cd capstone
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
pytest -q
```

If you get a pyarrow/NumPy error: `pip install "pyarrow<21"`.

## Team rules

- Never commit data files (`.h5ad`, `.npz`, slides). Share them through Drive.
- `git pull` before you start and before you push.
- Edit files in your own area. If you need a change in someone else's file, message them.
- Keep the data format above fixed. If it must change, announce it in the group first.
- Update your section of this README and the changelog when you finish something.

---

## Changelog

| Date | Who | What |
|---|---|---|
| 2026-10-29 | Dhriti | Repo skeleton, environment, pinned `requirements.txt` |
| 2026-10-29 | Dhriti | `complex.py`: Delaunay and grid builders, `B1`/`B2`, tests (Euler check caught and fixed a fake-triangle bug) |
| 2026-10-29 | Dhriti | `hodge.py`, `synth.py` and `tests/test_hodge.py` added (awaiting local test run) |
