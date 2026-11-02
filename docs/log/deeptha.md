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

### 2026-11-01 - Masked-flow pretraining and niche evaluation
- **Task:** Small Hodge-SCNN (self-supervised training for niches); Evaluation harness (ARI/NMI)
- **What I did:** added `model/pretrain.py` (hide 15% of edges, decompose the masked flow, reconstruct hidden edges), `eval/niches.py` (k-means on spot embeddings, ARI/NMI against labels, missing labels skipped, rows in the shared results format), `model/demo_niches.py` (toy tissue with 3 bands: gradient-rich, quiet, curl-rich), `tests/test_pretrain_eval.py`; small change to `model/hodge_scnn.py` (forward also returns `node_agg`); ran `python -m pytest -q` and `python -m model.demo_niches`
- **Result:** `37 passed, 2 warnings in 34.11s`. Toy (396 spots, 1100 edges, 704 triangles, 4 channels, 3 seeds) ARI / NMI: hodge 0.841 / 0.825, nosplit 0.529 / 0.601, curl 0.431 / 0.627, grad 0.426 / 0.577, harm 0.010 / 0.079, baseline (no learning) 0.535 / 0.635. Masked MSE: hodge 0.964 -> 0.226, nosplit 0.972 -> 0.240
- **What it means:** the model can find regions without seeing labels, and the full Hodge model beats the plain SCNN and the baseline on this toy. The toy was built with gradient and curl regions, so it favours the split; it checks the pipeline, it is not evidence for the paper. harm is near 0 because no harmonic pattern was planted
- **Problems and fixes:** first run gave hodge ARI 0.09, because the reconstruction loss only trains the edge layers and the node MLP stayed random; embeddings now use the trained edge embeddings pooled onto spots (`node_agg`). Hole grid points had no edges and gave NaN in the baseline; they are now unlabelled and skipped
- **Next:** run E3 on one DLPFC section once Mayur's DLPFC file and Disha's LR flows exist; add C-index to the evaluation harness

### 2026-11-02 - Comparison scoreboard (ours vs baselines)
- **Task:** Evaluation harness (ARI/NMI, comparison against baselines)
- **What I did:** added `eval/compare.py` (reads the shared results CSVs; paired by section; seeds averaged within a section; bootstrap 95% CI of the difference; Wilcoxon signed-rank with Holm correction; verdict BEATS / LEADS / PARITY / TRAILS; baselines and ablations kept separate), `tests/test_compare.py` (13 tests), and extended `model/demo_niches.py` with `--tissues` and `--out` so it writes the shared results format; ran `python -m pytest -q` and `python -m model.demo_niches --tissues 6 --modes hodge nosplit --seeds 0 --out results/toy_E3.csv`, then `python -m eval.compare results/toy_E3.csv --experiment E3toy --dataset TOY --ours "hodge-scnn[hodge]"`
- **Result:** `50 passed, 2 warnings in 11.00s`. Toy scoreboard, 6 sections, ARI: hodge 0.861 ± 0.020; flowsummary-kmeans baseline 0.527 (ours ahead by +0.335, 95% CI [+0.309, +0.360], ahead on 6/6 sections, Holm p 0.031, BEATS); nosplit ablation 0.515 (+0.347, CI [+0.327, +0.376], 6/6, Holm p 0.031, BEATS)
- **What it means:** one command now turns the results files into the table that answers "did we improve?", with a rule fixed in advance for what counts as better. The toy was built with gradient and curl regions, so this checks the tool, it is not a result for the paper. With 6 sections, 0.031 is the smallest p-value Wilcoxon can give
- **Problems and fixes:** `eval/compare.py` first crashed on identical scores (Wilcoxon undefined); now treated as p = 1.0. Pasted the test file late, so the first pytest run showed 37 instead of 50; re-ran after adding it
- **Next:** ask Mayur to save every baseline's scores in the shared results format (add STAGATE, GraphST, SpaGCN to his list); run E3 on one DLPFC section once his file and Disha's LR flows exist
