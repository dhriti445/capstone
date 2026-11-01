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

### 2026-11-01 - Hodge-SCNN skeleton and toy ablations
- **Task:** Small Hodge-SCNN (README board, Deeptha P1)
- **What I did:** added `model/hodge_scnn.py` (sparse SCNN layers; gradient, curl and harmonic branches; modes hodge/grad/curl/harm/nosplit), `model/train_toy.py`, `tests/test_model.py`; ran `python -m pytest -q` and `python -m model.train_toy`
- **Result:** `27 passed, 2 warnings in 30.71s`. Toy (90 train / 45 test graphs, 30 epochs) test accuracy: hodge 1.00, grad 0.64, curl 0.62, harm 0.73, nosplit 1.00. Final train loss: hodge 0.000, grad 0.418, curl 0.454, harm 0.518, nosplit 0.000; nosplit loss spiked to 0.525 at epoch 10 before recovering, hodge decreased smoothly. On starter.h5ad (TENX53: 4,869 spots, 14,289 edges, 9,413 triangles, beta1 = 8) one forward+backward pass takes 0.06 s on CPU
- **What it means:** the model runs end to end, the loss goes down, and each single Hodge part tops out near 67% as expected, so the ablation setup works. The toy is too easy to show Hodge beats plain SCNN; that has to come from real data
- **Problems and fixes:** `topomodelx` SCNNLayer builds a dense E x E identity each pass (~0.8 GB on a Visium slice), so I wrote the same layer with sparse ops; an inexact eigenvalue estimate broke orientation invariance, fixed by using exact `eigsh`
- **Next:** experiment plan (research questions -> experiments, baselines, ablations, metrics, incl. multi-seed training stability), then train on one real slice once Disha's LR flows exist
