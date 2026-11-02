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