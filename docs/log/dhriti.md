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
