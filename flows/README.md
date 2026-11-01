# P2 — ligand–receptor flows and visualisation

Your job in the pipeline: **expression → signed edge flows → pictures**.
Dhriti's code decomposes your flows; your plots go on every slide.

## 0. Folder layout

Put the three folders side by side inside your `implementation` folder:

```
implementation/
├── hest_data/          from `hf download` (st/TENX53.h5ad, metadata/, HEST_v1_3_0.csv)
├── TENX53_complex/     edges.npy, triangles.npy, spot_ids.npy (from the project)
└── p2_flows/           this code
```

If yours differ, change the paths at the top of `config.py` — nowhere else.

## 1. One-time setup (Mac terminal)

```bash
cd implementation/p2_flows
conda env create -f environment.yml      # takes a few minutes
conda activate hodge
python -m pytest -q tests                # should print "8 passed"
```

Always run `conda activate hodge` before working. COMMOT does not run on
NumPy 2, which is why this environment pins `numpy<2`; do not install the
project into `(base)`.

## 2. Try everything on fake data first (2 minutes)

```bash
python 00_make_demo_data.py
P2_DEMO=1 python 01_check_data.py
P2_DEMO=1 python 03_make_flows.py
P2_DEMO=1 python 04_figures.py commot
P2_DEMO=1 python 05_shuffle_control.py
P2_DEMO=1 python 06_hole_circulation.py
```

Figures appear in `outputs_demo/figures/`. If these run, your setup is right.

## 3. Real data: TENX53 (breast IDC, Visium)

Run the steps in order. Each one prints what it did; read the output.

| Step | Command | What you get | Time |
|---|---|---|---|
| 1 | `python 01_check_data.py` | Checks genes, spot order, coordinates, holes. Figure `01_spots_and_holes.png` | < 1 min |
| 2 | `python 02_pick_lr_pairs.py` | LIANA+ ranks all L–R pairs by spatial co-expression. `outputs/lr_ranking.csv`, `outputs/lr_pairs.csv` | a few min |
| 3 | `python 03_make_flows.py` | Product flows + COMMOT flows, saved in the agreed v0.1 layout | COMMOT: 10–30 min |
| 4 | `python 04_figures.py commot` and `python 04_figures.py product` | Four-panel Hodge figure per channel + energy-split bars | 1–2 min |
| 5 | `python 05_shuffle_control.py` | z-scores vs 100 spot shuffles | a few min |
| 6 | `python 06_hole_circulation.py` | Circulation around each hole, product flows tested against 100 shuffles. `outputs/hole_circulation.csv`, figure `06_hole_circulation.png` | a few min |

Quick COMMOT-free run while you test: `python 03_make_flows.py product`.

**Type only the command.** zsh does not treat `#` as a comment when you paste it, so text after `#` breaks the command (`>` saves output over a file, `( )` is read as a file pattern).

### What to check after each step

- **Step 1.** `first genes` must be gene symbols such as `TGFB1`. If they start
  with `ENSG`, follow the printed hint. The holes figure should show ~8 red loops;
  note which sit on real gaps in the tissue and which are just the ragged edge.
- **Step 2.** Open `lr_ranking.csv`. Before finalising `lr_pairs.csv`, check each chosen
  pair is known in breast cancer (literature check is your job in the work plan).
  You can edit `lr_pairs.csv` by hand. Keep the format `ligand,receptor`; complexes use `_`
  (e.g. `TGFBR1_TGFBR2`).
- **Step 3.** COMMOT prints `share_on_complex_edges`. It is the fraction of COMMOT's
  spot-to-spot signal that lands on complex edges. Near 1 is good; if it is low,
  lower `COMMOT_RANGE_FACTOR` in `config.py`.
- **Step 4.** Look at the figures. Write down, per pathway, the gradient / curl / harmonic split.

## 4. What to hand to Dhriti

`outputs/TENX53_flows_commot.h5ad` and `outputs/TENX53_flows_product.h5ad`. Each holds:

- `uns['complex']` → `edges`, `triangles` (unchanged from her files)
- `uns['flows']` → `(E, C)` float matrix, positive = signal goes i → j
- `uns['flow_names']` → the C channel names  ← **new; agree this key with her**
- `uns['flow_kind']` → `"commot"` or `"product"`

Rebuild B1, B2 with her `hodgecc.complex.build_incidence`; `flows.build_incidence`
is the same thing and asserts `B1 @ B2 == 0`.

## 5. Things to know before you present results

1. **Product flows are mostly gradient, by maths.** Its non-gradient part is proportional to
   ∇L × ∇R: loops only appear where the ligand and receptor patterns change in
   *different directions*. If one gene is roughly uniform, the flow is pure gradient. So a
   high gradient share for product flows is expected, not a bug, and is a point for the
   theory section (tell Dhriti).
2. **Read the shuffle control carefully.** Spatially smooth expression always gives a high
   gradient share and a low curl share compared with shuffled spots. So a positive gradient
   z-score says "this pathway has spatial structure"; a negative curl z-score does **not** mean
   curl is absent. A null that keeps spatial smoothness is needed to test curl and harmonic;
   that is a team decision.

## Files

| File | Purpose |
|---|---|
| `config.py` | all paths and settings |
| `flows.py` | load complex, preprocess, product / COMMOT → edge flows, save |
| `hodge_quick.py` | temporary Hodge decomposition until Dhriti's is ready |
| `viz.py` | `plot_spots`, `plot_edge_flow`, `plot_hodge_panels`, `plot_holes`, `plot_energy_bars` |
| `synthetic.py`, `00_make_demo_data.py` | fake tissue with planted patterns |
| `tests/` | `python -m pytest -q tests` |
3. **Harmonic share is always ~0%; use step 6 instead.** The harmonic space has only
   beta1 dimensions (8 on TENX53) out of ~14,000 edges, so even a random flow puts ~0.06% of its
   energy there. Step 6 measures circulation around each hole directly (a line integral;
   gradients always sum to 0 around a loop).
4. **COMMOT's curl share (~40-60%) is close to what pure noise would give** (random flow on
   TENX53: ~34% gradient / ~66% curl). So "COMMOT has more curl than product flows" is not yet
   evidence of feedback loops; it needs a COMMOT null (rerun COMMOT on ~10 shuffles). The same
   applies to COMMOT circulation in step 6: in the demo, COMMOT showed a strong circulation
   (index 0.8) around a hole where nothing was planted.
