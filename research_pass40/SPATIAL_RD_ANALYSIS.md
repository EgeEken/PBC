# Spatial RD analysis

The regional analysis reuses the fixed Pass 39 masks: convert source RGB to linear light, use the top 20% Sobel luma magnitude as edge, then use the top 20% 8×8 local luma variance among remaining pixels as texture; all other pixels are smooth. Masks are fixed per source for each comparison. One root qindex sample per 64×64 coded superblock is expanded to pixels for regional qindex summaries. Qindex describes encoder decisions; it is not an attribution of regional bytes.

The mode-6 root maps vary across superblocks and match uninstrumented AVIF outputs byte-for-byte. Same-q and matched-rate evidence differ:

- In same-q component tests, mode 6 reduces MSE while spending more bytes. It particularly helps Cat and Coast, where rate rises by 26–216% across the tested points.
- In matched-rate Cat comparisons, the ordinary boost lowered smooth-region MSE by roughly 3%, while edge and texture MSE rose roughly 5–8%. The region qindex means were lower in smooth regions. This supported a single redistribution test around the existing variance crossover, not a new threshold search.
- The frozen correction moved Cat's matched-rate edge and texture error in the intended direction. Across the five intervention pairs that met the 5% rate rule, edge MSE improved in four pairs (mean −2.18%) and texture MSE improved in four (mean −2.48%); smooth MSE improved in only two (mean +4.85%, i.e. average regression). The broad mix, especially Googlewiki's smooth-region regressions, limits the claim to a small redistribution effect.

The exact regional measurements and qindex means are in `outputs/intervention_analysis.json`. `visuals/intervention_rd.png` shows the seven fixed-q points; `visuals/intervention_matched_mse.png` shows only the five rate-matched pairs.
