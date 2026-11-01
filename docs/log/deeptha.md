# Log: Deeptha

Newest entries at the bottom. Format is defined in docs/LOG_PROMPT.md.
Only edit this file, not other people's logs.


### 2026-11-01 - Local environment setup
- **Task:** Setup for P1 (no README board row yet)
- **What I did:** cloned `dhriti445/capstone`, installed Python 3.11 from python.org, created `.venv`, installed CPU torch, then `pip install --no-deps -r req_local.txt` and `pip install -e .`, ran `python -m pytest -q` and `python scripts/demo_toy.py`
- **Result:** `14 passed, 2 warnings in 68.12s`; demo: nodes 400, edges 1071, triangles 678, beta1 = 2 (expected 2), relative error gradient 1.09e-11, curl 1.48e-11, harmonic 5.35e-10, harmonic residual 1.20e-10
- **What it means:** my environment matches Dhriti's, the Hodge decomposition recovers the planted gradient/curl/harmonic parts on the toy grid, and I can start the data contract and the model on top of it
- **Problems and fixes:** `py -3.11` not found and `winget` missing, so installed Python 3.11 from python.org; `requirements.txt` has a laptop path (`-e c:\users\dhriti\...`) and a `pyg-nightly` pin, so used a local copy `req_local.txt` without them (not committed); `numpy==1.26.4` conflicts with `fast-array-utils` (needs numpy>=2), fixed with `pip install --no-deps`; harmless docrep SyntaxWarnings in tests
- **Next:** write `docs/DATA_CONTRACT.md`, then a small Hodge-SCNN in `model/` trained on `hodgecc.synth` data
