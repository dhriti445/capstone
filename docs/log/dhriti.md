# Log: Dhriti (P3, Hodge core)

Newest entries at the bottom. Format is defined in docs/LOG_PROMPT.md.

### 2026-10-29 - Repo and environment
- **Task:** Repo skeleton and environment
- **What I did:** created `hodgecc` repo and Windows venv (Python 3.11), installed CPU torch, scanpy, squidpy, toponetx, topomodelx, gudhi, pytest
- **Result:** `pytest tests/test_env.py` passes
- **What it means:** the environment can import every library we need
- **Problems and fixes:** `pyarrow requires NumPy 2.0`, fixed with `pip install "pyarrow<21"`
- **Next:** cell-complex builder

### 2026-10-29 - Cell-complex builder
- **Task:** `complex.py`
- **What I did:** wrote Delaunay and grid builders, `B1`/`B2`, tests
- **Result:** 3 tests pass; `B1 @ B2 = 0`; Euler check N-E+T = 1 on a filled disk
- **What it means:** orientation and incidence matrices are consistent
- **Problems and fixes:** Euler test found 291 triangles instead of 288 because 3-cliques include fake faces; fixed by using Delaunay simplices directly
- **Next:** Hodge decomposition (`hodge.py`)
### 2026-10-29 - Hodge decomposition and full test run
- **Task:** `hodge.py`, `synth.py`, tests
- **What I did:** added Hodge decomposition (gradient/curl/harmonic), toy grid complexes with holes, planted-flow generator, and tests; ran `pytest -q`
- **Result:** 14 passed (env 1, complex 3, hodge 10)
- **What it means:** the decomposition recovers planted parts, parts are orthogonal, and beta1 equals the number of holes on the toy meshes
- **Problems and fixes:** none (only harmless docrep SyntaxWarnings from squidpy)
- **Next:** run `scripts/demo_toy.py` for the figure, then `scripts/check_visium.py` on a real slice

### 2026-10-31 - Toy demo and first real Visium slice
- **Task:** Real Visium slice into a complex; toy decomposition figure
- **What I did:** ran `scripts/demo_toy.py` and `scripts/check_visium.py` (squidpy built-in Visium slice, mouse brain, as a stand-in until the DLPFC file arrives)
- **Result:** toy mesh 400 nodes / 1071 edges / 678 triangles, beta1 = 2, errors ~1e-11 to 5e-10. Visium slice: 2688 spots, 7790 edges, 5097 triangles, betti (b0, b1) = (3, 8), Euler -5 = 3 - 8
- **What it means:** the complex builder works on real hex-grid data and the Euler identity holds. beta1 = 8 may include fake holes from tissue gaps, to be separated later by a persistence threshold
- **Problems and fixes:** FutureWarning from squidpy about spatial_neighbors (no action yet)
- **Next:** repeat on DLPFC when Mayur shares the file; notebook for the validation experiments

### 2026-11-01 - Real slice complex and sharing format
- **Task:** Real Visium slice into a complex, ready for Disha
- **What I did:** ran `scripts/real_slice_complex.py` on Mayur's `starter.h5ad`; added `hodgecc/io.py` (`save_complex`, `load_complex`) and a small `.npz` output
- **Result:** TENX53: 4869 spots, 14289 edges, 9413 triangles, b0 = 1, b1 = 8; 126 KB .npz saved
- **What it means:** the real-data deliverable is done, and Disha can load the complex with one call
- **Problems and fixes:** anndata refused to write string index (`allow_write_nullable_strings`); fixed by enabling the setting and making the .h5ad copy optional
- **Next:** wait for Disha's flow array, then run real vs shuffled decomposition

### 2026-11-01 - Real-data notebook (TENX53)
- **Task:** Real-data validation notebook
- **What I did:** built `notebooks/real_slice_TENX53.ipynb`: real mesh, Hodge Laplacian spectrum, planted flows on the real mesh, proxy L-R flows (8 pairs) with a 20-shuffle control
- **Result:** 4869 spots / 14289 edges / 9413 triangles, b0 = 1, b1 = 8 (gudhi and spectrum agree); planted-flow error ~1e-10 without noise; real flows have higher gradient share than shuffled in 7 of 8 pairs; harmonic share < 0.1% everywhere
- **What it means:** the pipeline works on real data and separates real from shuffled; harmonic flow is negligible on this slice because its holes are missing spots, not glands
- **Problems and fixes:** none
- **Next:** swap in Disha's LIANA+/COMMOT flows; test on a tissue with real glands or vessels

### 2026-11-02 - Verified Disha's flows
- **Task:** Check P2 flow outputs against my decomposition
- **What I did:** ran `scripts/check_flows.py` on TENX53_flows_commot.h5ad (32 channels) and TENX53_flows_product.h5ad (10 channels)
- **Result:** edges/triangles identical to my complex; B1@B2 = 0; my gradient/curl/harmonic shares match her CSVs to ~1e-14 on all 42 channels; harmonic share <= 0.27% unfilled and <= 0.02% with gaps <= 2 filled
- **What it means:** the interface between our parts works, and the holes on TENX53 are dropouts, not ducts
- **Problems and fixes:** none
- **Next:** spot-shuffled COMMOT control (needs Disha's reruns); swap COMMOT flows into the real-data notebook
### 2026-11-02 - DLPFC data and complex
- **Task:** Real DLPFC slice into a complex; unblock the team's data
- **What I did:** wrote `scripts/get_dlpfc.py` (downloads 12 sections from Hugging Face han-shu/st_datasets, standardises labels from the `cluster` column to L1-L6/WM, adds donor g1/g2/g3, keeps counts); zipped 12 sections (~485 MB) for teammates
- **Result:** 12 sections, e.g. 151673 has 3639 spots and 33538 genes; donor g2 has only 5 layers (L3-L6, WM)
- **What it means:** the team now has the labelled benchmark data for the niche experiments
- **Problems and fixes:** raw files had no `layer` column, so I identified `cluster` as the expert annotation by its 7/7/5 layer pattern
- **Next:** build the complex for 151673 and all 12 sections; run Disha's flows on DLPFC

### 2026-11-03 - Real vs shuffled COMMOT control
- **Task:** Real-vs-shuffled control on COMMOT flows
- **What I did:** ran `scripts/shuffle_control.py` on TENX53_flows_commot.h5ad vs 10 spot-shuffled COMMOT runs from Disha (32 pathways, same edges)
- **Result:** gradient share above all shuffles for 10 pathways, below all for 10, within range for 12 (about 3 per direction expected by chance); medians 50.6% vs 50.1% (gradient), 49.3% vs 49.8% (curl); harmonic with gaps filled at most 0.019% real vs up to 0.12% shuffled
- **What it means:** real flows have pathway-specific spatial structure (both directions), no overall excess of loops, and no circulation around holes beyond chance
- **Problems and fixes:** none; 10 shuffles is too few for per-pathway significance
- **Next:** more shuffles for curl-heavy pathways; H&E check of the 7-spot gap