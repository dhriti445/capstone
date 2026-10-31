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