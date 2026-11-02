# Log: Disha

Newest entries at the bottom. Format is defined in docs/LOG_PROMPT.md.
Only edit this file, not other people's logs.

## 2026-11-01 — TENX53 ligand–receptor flows (P2)

**Done**
- Env `hodge` (Python 3.11, numpy<2 — COMMOT 0.0.3 breaks on NumPy 2), LIANA 1.10, COMMOT 0.0.3.
- Downloaded HEST TENX53 (breast IDC, Visium, 4,898 spots); aligned to Dhriti's complex (4,869 spots, 14,289 edges, 9,413 triangles; B1·B2 = 0; β₁ = 8).
- LR pairs (LIANA+ bivariate, Moran's R, 100 perms; one pair per ligand and per receptor + curated breast pairs): SPP1→CD44, CXCL12→CXCR4, MDK→SDC1, JAG1→NOTCH1, C3→LRP1, HLA-DPB1→CD4, COL11A1→ITGA11_ITGB1, APOE→TREM2, CD14→ITGB2, CDH1→PTPRF. TGFB1→TGFBR and VEGFA→KDR below the expression filter.
- Flows in v0.1 format: product (10 pairs) and COMMOT (32 pathways, CellChatDB secreted, dis_thr = 1.5 × spot spacing; 100% of signal on complex edges). Added `uns['flow_names']` (to confirm with Dhriti).
- Figures: 4-panel Hodge per channel, energy splits, shuffle z-scores, hole circulation.

**Findings**
1. Product flows 74–98% gradient: expected, since the non-gradient part ∝ ∇L × ∇R.
2. COMMOT ~40–60% curl, but a random flow on this complex is ~66% curl, so not yet evidence of loops; needs a COMMOT shuffle null.
3. SPP1→CD44: LIANA Moran's R −0.10 (not co-localised), yet strong directed flow (shuffle z = 19).
4. JAG1→NOTCH1 flat (Moran p = 0.58, z = 2.2) — negative control.
5. All 5 measurable holes are 6-edge loops = single QC-dropped spots, not ducts. No pair circulates beyond chance (no |z| > 3). Harmonic share ≈ 0% by construction (8 of 14,289 dims).

**Handed over**: TENX53_flows_commot.h5ad, TENX53_flows_product.h5ad, commot_energy.csv, product_energy.csv → Dhriti.

**Next**: Dhriti to cross-check energy splits and fill single-spot gaps in the complex; literature table for the 10 pairs; pick Review-2 figures; COMMOT shuffle null (~10 runs); rerun on DLPFC / colon when available.

## 2026-11-02 — COMMOT shuffle nulls (P2)

**Done**
- Dhriti cross-checked my TENX53 flows: edges/triangles identical, gradient/curl/harmonic match on all 42 channels (32 COMMOT + 10 product). `uns['flow_names']` kept.
- Hole correction from Dhriti: 6 of 8 holes are single dropped spots, one is a 2-spot gap, one a 7-spot gap. My step 6 skipped one 24-edge self-touching boundary, likely holding the larger gaps; the 7-spot gap (~250 µm) needs checking on the H&E.
- New `flows/07_commot_shuffles.py`: 10 COMMOT runs on spot-shuffled expression (same settings as step 3), saved as `outputs/shuffles/TENX53_flows_commot_shuf{0..9}.h5ad`. Same 32 pathways and order, same edges, `flow_kind='commot_shuffle'`, plus `shuffle_seed`, `shuffle_perm`; no gene matrix. ~5.9 min per shuffle.
- Demo check: shuffled COMMOT curl share ≈ real (29–46% real vs 30–44% shuffled), so the test is needed before claiming feedback loops.

**Handed over**: TENX53_commot_shuffles.zip → Dhriti.

**Next**: Dhriti runs the COMMOT shuffle test; check the 7-spot gap on H&E; literature table; Review-2 figures; DLPFC when ready.