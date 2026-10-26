# Frozen allocator matched-rate validation

Weight0.4 is a useful content-dependent alternative. Nearest fresh controls within 5% yield13/16 main and 4/4 holdout MSE wins. Against the best actually measured prior/new control within 5%, there are 16/19 main and 3/4 holdout wins. These direct comparisons retain rate gaps; they are not exact equal-rate claims. Inside released Q-HQ rates, the stronger direct comparison has 11/14 main wins but only one holdout match, essentially neutral. Three candidate images remain dominated somewhere on the measured curve.

## Frozen policy and evaluation procedure

The exact Pass 3 selector at weight 0.4, Quality 500k/35, remains unchanged. Same production model, custom RNG/seed, preparation, initialization, proposal/model policy, stop rule, utility and syntax. No weights, working resolution or candidate parameters were tuned on the holdout. Archived Pass 3 source/model/input hashes are verified; fresh candidate confirmation must reproduce every saved Pass 3 hash. The candidate's exact post-LZMA bpp is the target used to choose ordinary controls; their measured MSE is never used to choose the next configuration.

prior_rate_overlap.json records all 25 candidate points, nearest existing nondominated measured controls on either side, absolute/relative gaps, bracket width, nearest500k control and observed dominance. Prior gaps are often wide; a measured bracket is not proof that its chord is achievable. The new protocol was frozen before its results in PLAN.md: ordinary Quality 500k budgets 10/25 first; then bitrate-only bounded interpolation/bisection in budgets 3..50 using available20/35 controls, and only q 0.7/0.9 at 35 if discrete budget bracketing offers no unused point. At most six fresh settings/image, stopping at a 3% point where possible. Every evaluated ordinary setting is retained, including nonmonotonic/worse results; no prefixes or new allocator adaptation are used.

106 fresh ordinary settings were encoded across the 25 images. Their individual exploratory timings are not repeated speed evidence. The selected nearest fresh setting and the frozen allocator were then re-encoded with complete per-config warmups and three interleaved repeats. Fresh smaller budgets can change search-q interpolation, model step features and the complete trajectory. Their lower compute cost is real and must not be confused with reducing allocator-only overhead at equal35-budget work.

## Direct measured points and remaining gaps

Positive gap means the fresh control uses more bpp than the candidate. Negative MSE% favors the candidate; it is a measured-point difference, not exact equal-rate quality. Close5 means abs(control_bpp/candidate_bpp -1)<=5%, using the candidate as denominator. A5.0021% gap remains outside the 5% set. Strict wins require both no higher MSE and no higher complete bpp, with one strict inequality. Missing matches are not wins.

| Set | Image | Candidate bpp | Candidate MSE | Fresh control | Control bpp | Control MSE | Rate gap % | MSE % | Within5% | Strict RD win | Fresh lower gap% | Fresh upper gap% | Q-HQ region | Best actual close MSE% | Dominated anywhere |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | cat.jpg | 0.0613 | 250.5060 | ordinary_p21_q0.8 | 0.0603 | 274.2866 | -1.7272 | -8.6700 | True | False | 1.7272 | 16.9127 | True | -8.6700 | False |
| hpt_data | cathedral.png | 0.0716 | 306.2071 | ordinary_p16_q0.8 | 0.0742 | 371.1153 | 3.5157 | -17.4900 | True | True | 22.3888 | 3.5157 | True | -15.9556 | False |
| hpt_data | coast.jpg | 0.0157 | 28.6004 | ordinary_p25_q0.8 | 0.0154 | 32.4783 | -2.0131 | -11.9397 | True | False | 2.0131 | — | True | -11.9397 | False |
| hpt_data | ducklings.jpg | 0.0266 | 35.3510 | ordinary_p42_q0.8 | 0.0268 | 34.9762 | 1.0963 | 1.0716 | True | False | 14.6502 | 1.0963 | True | 2.0679 | False |
| hpt_data | eiffel.png | 1.8878 | 27.4426 | ordinary_p50_q0.8 | 1.8528 | 25.5588 | -1.8565 | 7.3705 | True | False | 1.8565 | — | True | 9.5571 | True |
| hpt_data | field.jpg | 0.0369 | 72.3072 | ordinary_p8_q0.8 | 0.0398 | 78.2990 | 7.8690 | -7.6524 | False | True | 31.8402 | 7.8690 | True | — | False |
| hpt_data | flag.jpg | 0.0156 | 23.9407 | ordinary_p10_q0.8 | 0.0154 | 34.8144 | -1.5033 | -31.2334 | True | False | 1.5033 | 40.7311 | False | -31.2334 | False |
| hpt_data | flower.jpg | 0.0309 | 31.5861 | ordinary_p35_q0.7 | 0.0286 | 34.9912 | -7.5651 | -9.7313 | False | False | 7.5651 | 19.9897 | True | -29.0502 | False |
| hpt_data | flowers.jpg | 0.0424 | 140.0138 | ordinary_p6_q0.8 | 0.0457 | 209.5791 | 7.9027 | -33.1928 | False | True | 54.2699 | 7.9027 | True | -13.9945 | False |
| hpt_data | googlewiki.png | 1.1774 | 590.1078 | ordinary_p9_q0.8 | 1.2110 | 729.1595 | 2.8536 | -19.0701 | True | True | 39.7759 | 2.8536 | True | -18.6968 | False |
| hpt_data | mario64.png | 0.3634 | 217.2607 | ordinary_p6_q0.8 | 0.3655 | 397.9062 | 0.5972 | -45.3990 | True | True | 23.1082 | 0.5972 | False | -45.3990 | True |
| hpt_data | monet.jpg | 0.8239 | 88.3818 | ordinary_p23_q0.8 | 0.7927 | 96.7375 | -3.7915 | -8.6374 | True | False | 3.7915 | 9.5535 | True | -8.6374 | False |
| hpt_data | monkey.png | 0.3049 | 35.1971 | ordinary_p4_q0.8 | 0.2850 | 85.1247 | -6.5421 | -58.6523 | False | False | 6.5421 | 10.7262 | False | -1.4007 | False |
| hpt_data | panorama.png | 0.2420 | 99.9333 | ordinary_p9_q0.8 | 0.2477 | 155.7349 | 2.3622 | -35.8312 | True | True | 33.7651 | 2.3622 | True | -35.8312 | False |
| hpt_data | park.jpg | 0.1004 | 236.3723 | ordinary_p35_q0.7 | 0.1031 | 234.5667 | 2.7122 | 0.7698 | True | False | 15.2518 | 2.7122 | True | 0.7698 | False |
| hpt_data | perlman.png | 0.1216 | 171.0971 | ordinary_p9_q0.8 | 0.1217 | 188.9445 | 0.0740 | -9.4458 | True | True | 37.5971 | 0.0740 | True | -5.4471 | False |
| hpt_data | ratedistortion.png | 0.7867 | 56.3980 | ordinary_p10_q0.8 | 0.7753 | 120.1944 | -1.4424 | -53.0777 | True | False | 1.4424 | 28.9042 | False | -53.0777 | False |
| hpt_data | seagull.jpg | 0.1170 | 59.3393 | ordinary_p15_q0.8 | 0.1172 | 82.5147 | 0.2037 | -28.0864 | True | True | 24.6013 | 0.2037 | False | -11.9320 | False |
| hpt_data | stockpoint.jpg | 0.1296 | 25.4944 | ordinary_p9_q0.8 | 0.1308 | 31.1440 | 0.9089 | -18.1400 | True | True | 20.3241 | 0.9089 | True | -17.1687 | False |
| hpt_data | sunset.jpg | 0.0657 | 273.7244 | ordinary_p24_q0.8 | 0.0649 | 278.6293 | -1.1899 | -1.7604 | True | False | 1.1899 | 3.1386 | True | -0.7559 | False |
| hpt_data_val | 15k_kills.png | 1.7125 | 62.4068 | ordinary_p3_q0.8 | 1.6959 | 280.2896 | -0.9700 | -77.7349 | True | False | 0.9700 | 31.3054 | False | -77.7349 | False |
| hpt_data_val | ducks.jpg | 0.2624 | 144.4492 | ordinary_p11_q0.8 | 0.2755 | 195.2351 | 5.0021 | -26.0127 | False | True | 7.1271 | 5.0021 | True | — | False |
| hpt_data_val | old_man_standing.jpg | 1.4898 | 13.9744 | ordinary_p4_q0.8 | 1.4964 | 36.8770 | 0.4434 | -62.1055 | True | True | 7.4936 | 0.4434 | False | -62.1055 | False |
| hpt_data_val | pigeon.JPG | 0.0123 | 53.6133 | ordinary_p12_q0.8 | 0.0120 | 82.0397 | -2.1428 | -34.6496 | True | False | 2.1428 | 58.5631 | False | -34.6496 | True |
| hpt_data_val | pigeons.JPG | 0.0105 | 20.6324 | ordinary_p25_q0.8 | 0.0106 | 20.9738 | 0.9225 | -1.6277 | True | True | 16.3971 | 0.9225 | True | 0.0839 | False |

Candidate500k35 and control rates/raw/final bytes, decoded metrics, configs/dimensions/counts and source/model/input identifiers remain in matched_controls.jsonl, matched_rate_comparisons.json and matched_confirmation.jsonl. New below/above endpoints are full raw rows, not just rounded values. Direct closest-fresh and strongest-actual-close comparisons are separate: strongest-close chooses lowest MSE among all prior/new measured controls within 5%, retaining its signed rate gap. A candidate can beat a poor fresh control yet be dominated by an older point at a different rate; mario64/pigeon demonstrate this.

Aggregate coverage and direct/interpolated results. qhq_bestclose requires a strongest actual close control; its nearest-fresh columns may include a gap outside tolerance and are descriptive, while the best-close columns are direct tolerance comparisons:

| Set | Subset | Fresh cases | Fresh MSE % | Fresh median % | Fresh MSE wins | Fresh strict wins | Abs rate gap % | Best actual close coverage | Best close MSE % | Best close median % | Best close wins | Prior interpolation % | Densified interpolation % | Conservative interpolation % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | all | 20 | -19.4399 | -14.7149 | 17 | 9 | 2.8863 | 19 | -15.6208 | -11.9397 | 16 | -5.1601 | -3.3730 | -3.0544 |
| hpt_data | close3 | 14 | -18.1030 | -15.0399 | 11 | 6 | 1.4672 | 14 | -16.2683 | -11.9359 | 11 | -3.8630 | -2.4468 | -2.1199 |
| hpt_data | close5 | 16 | -17.4731 | -14.7149 | 13 | 7 | 1.7405 | 16 | -15.7719 | -11.9359 | 13 | -3.7589 | -2.5197 | -2.2337 |
| hpt_data | qhq_close5 | 12 | -10.1477 | -9.0579 | 9 | 5 | 2.0084 | 12 | -9.2256 | -8.6537 | 9 | -6.0943 | -4.3438 | -4.0608 |
| hpt_data | qhq_bestclose | 14 | -11.7641 | -9.5886 | 11 | 6 | 2.8264 | 14 | -10.9823 | -10.3048 | 11 | -7.7103 | -5.0072 | -4.7646 |
| hpt_data_val | all | 5 | -40.4261 | -34.6496 | 5 | 3 | 1.8962 | 4 | -43.6015 | -48.3775 | 3 | -9.2590 | -13.6268 | -8.3379 |
| hpt_data_val | close3 | 4 | -44.0294 | -48.3775 | 4 | 2 | 1.1197 | 4 | -43.6015 | -48.3775 | 3 | -9.4902 | -16.0793 | -9.4682 |
| hpt_data_val | close5 | 4 | -44.0294 | -48.3775 | 4 | 2 | 1.1197 | 4 | -43.6015 | -48.3775 | 3 | -9.4902 | -16.0793 | -9.4682 |
| hpt_data_val | qhq_close5 | 1 | -1.6277 | -1.6277 | 1 | 1 | 0.9225 | 1 | 0.0839 | 0.0839 | 0 | -2.1827 | -2.0947 | -2.0947 |
| hpt_data_val | qhq_bestclose | 1 | -1.6277 | -1.6277 | 1 | 1 | 0.9225 | 1 | 0.0839 | 0.0839 | 0 | -2.1827 | -2.0947 | -2.0947 |

The all subset's fresh measured-point average includes gaps outside tolerance and is descriptive only; close3/close5 are the meaningful direct tolerance subsets. qhq_close5 restricts to candidate rates between actual released Quality/HQ endpoints. Strongest-close coverage can exceed fresh-close coverage because an older point provides a match. The best actual close comparison still has rate mismatch and cannot define a continuous RD curve. Main/holdout arithmetic per-image percent means and medians are separate; no image-size or compressed-byte weighting is hidden.

Against the best actual measured control within 5% bpp, mean per-image MSE changes -15.62% on 19 main cases (16 wins, median -11.94%) and -43.60% on four holdout cases (three wins, median -48.38%). Large holdout improvements against low-budget500k controls, especially15k_kills/old_man_standing, occur outside the released Q-HQ interval. Within that interval, direct main MSE is -10.98% over 14 matches (11 wins); holdout is +0.0839% on only pigeons.JPG. Eight main/one holdout cases strictly beat the strongest close control at their actual points, but mario64 is itself dominated by an older lower-rate point; seven main/one holdout such cases remain nondominated in the observed controls. Six of those main cases are in the Q-HQ region. Thus actual-file evidence supports individual frontier extensions, not a universal preset or robust high-quality holdout claim.

## Interpolation and comparison with Pass 3

The prior strongest-envelope estimates reproduce Pass 3 exactly: main -5.1601% over 20, holdout -9.2590% over 5. Densified neighboring nondominated controls and exact weights are retained separately. Adding a measured point above an old interpolation chord can raise the interpolated baseline MSE, even when no old measured point worsens. This can make apparent candidate gains larger without a new codec benefit. It occurs notably for old_man_standing: the sparse prior chord bridges a large quality discontinuity, while new low-budget controls at nearby bpp remain poor. Therefore the conservative comparison uses the lower baseline MSE from the original and densified interpolants per image, never presenting a weakened old chord as a stronger control. These are still estimates, not achievable extra bitstreams or a lower-bound proof. No extrapolation or sparse BD-rate claim is used.

The ordinary500k-only dense curve, full prior/all-current-release envelope and actual close controls answer different questions. Their coverage/signs must not be pooled. A mean gain over all rates does not establish high-quality holdout performance; only one holdout image has an actual close control inside the released Q-HQ rate region, and it is essentially neutral/slightly worse against the best prior nearby ordinary point. The second target-region holdout case remains just beyond the 5% tolerance after the bounded search.

## Named regression mechanisms

| Image | Selected types | Changed retained gain | Changed retained raw bits | Ordinary patches | Integrated patches | Ordinary MSE | Integrated MSE | Fixed-geometry MSE | Ordinary bpp | Integrated bpp | First changed payload |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mario64.png | {'coarser_grid': 30, 'lower_precision': 5} | 0.8812 | 0.3408 | 35 | 35 | 175.4103 | 217.2607 | 257.8631 | 0.7284 | 0.3634 | 3 |
| monet.jpg | {'coarser_grid': 8, 'original': 17, 'lower_precision': 10} | 0.8399 | 0.5110 | 33 | 35 | 85.6660 | 88.3818 | 108.6426 | 0.8770 | 0.8239 | 3 |
| eiffel.png | {'lower_precision': 7, 'original': 11, 'coarser_grid': 2} | 0.9365 | 0.7065 | 23 | 20 | 25.2207 | 27.4426 | 31.4989 | 1.8281 | 1.8878 | 3 |
| pigeon.JPG | {'coarser_grid': 20, 'lower_precision': 15} | 0.9095 | 0.5176 | 35 | 35 | 41.4099 | 53.6133 | 55.4943 | 0.0222 | 0.0123 | 3 |

mario64: every one of 35 actions changes representation (30 coarser, five lower precision); no early stop. Changed actions retain88.1% of local gain at 34.1% raw bits on average, yet final RGB loses detail/efficiency versus 250k Balanced, which already has both lower rate and MSE. The fixed-geometry counterpart is worse still, so later evolving geometry partially recovers quality at additional rate; it does not erase the observed dominance. Coarsening and resolution/allocation interaction are supported mechanisms; an isolated percentage of final loss caused by high-frequency detail is not identified. Monet:18/35 representations change (eight coarser, ten lower precision); the allocator runs35 patches versus ordinary33. Changed actions retain84.0% local gain/51.1% raw bits. This is not premature stopping. Unlike the other named cases, new close ordinary controls favor the allocator(-8.64% measured MSE against the best5%-near point); the prior interpolated envelope still predicts+12.02% loss. It remains an interpolation/control-family issue rather than a newly verified direct loss, and must not be called fixed or assigned one precision-only cause. Eiffel: nine of 20 emitted actions change (seven lower precision, two coarser), retaining 93.65% local gain/70.65% raw bits. The modified trajectory stops at 20 versus ordinary23; both final MSE and bpp worsen (25.22/1.82813 to 27.44/1.88780). A fresh p25 control at 1.81396 bpp/25.04869 MSE dominates it; this is a verified representation/trajectory/stop failure, not merely a sparse interpolation artifact. Pigeon: all 35 actions change (20 coarser, 15 lower precision), without early stopping, retaining 90.95% local gain/51.76% raw bits. It beats a nearby low-budget500k control but is dominated by released Balanced at 0.009133 bpp/51.3971 MSE versus 0.012250/53.6133. Its main failure is extra500k representation cost relative to an effective low-rate250k trajectory, with coarsening/precision and subsequent geometry interacting. No unique model-capacity or entropy-statistic cause is proved for any case.

regression_diagnostics.json records every same-state original/alternative/selected cell size, maximum/actual precision, local clipped working-channel SSE gain, raw bits, utility scale and stop, and reproduces the frozen final file on all four cases. The fixed-geometry Pass 3 counterparts separate a representation-only trajectory ablation from reoptimized later geometry; they do not isolate a unique causal percentage for every final-RGB loss. Coarser grids and lower precision visibly reduce the chosen correction's capacity, but missing fine detail, color/resizing and later overlap cannot be assigned separate causal shares from local SSE alone. Raw-bit proxy calibration is not final entropy coding: eiffel even grows both raw/final size after an altered trajectory, despite every local selector optimizing its own proxy. No per-candidate LZMA is introduced.

## Full latency at the selected actual points

| Set | Configuration | MSE | bpp | Raw bytes | Patches | Encode s | Encode CV | Decode s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | ordinary | 174.8378 | 0.3187 | 106262.8500 | 16.6500 | 0.6426 | 0.0787 | 0.1113 |
| hpt_data | ordinary | 174.8378 | 0.3187 | 106262.8500 | 16.6500 | 0.6426 | 0.0787 | 0.1113 |
| hpt_data | allocator | 138.4631 | 0.3211 | 117060.8500 | 34.2500 | 0.9382 | 0.0869 | 0.1251 |
| hpt_data | allocator | 138.4631 | 0.3211 | 117060.8500 | 34.2500 | 0.9382 | 0.0869 | 0.1251 |
| hpt_data_val | ordinary | 123.0830 | 0.6981 | 61607.4000 | 11 | 0.4538 | 0.1043 | 0.0784 |
| hpt_data_val | ordinary | 123.0830 | 0.6981 | 61607.4000 | 11 | 0.4538 | 0.1043 | 0.0784 |
| hpt_data_val | allocator | 59.0152 | 0.6975 | 79048 | 29.6000 | 0.7018 | 0.0731 | 0.0961 |
| hpt_data_val | allocator | 59.0152 | 0.6975 | 79048 | 29.6000 | 0.7018 | 0.0731 | 0.0961 |

These controls often request far fewer patches than35. Any allocator encode penalty against them includes additional sequential search/model/action work, not just the price of alternative representations. Equal-budget incremental costs and exact cost optimizations are evaluated separately in PBC_ALLOCATOR_EFFICIENCY.md. Runtime improvements are not counted as RD improvements.

## Acceptance and next research

The gate is passed only for a content-dependent alternative with useful observed finite-frontier extensions. It is not passed for a generally better Q-HQ release preset: three measured dominance failures, remaining rate gaps, only one directly matched high-quality holdout case, and material encode cost remain. Ordinary500k35 to allocator remains a rate-for-quality tradeoff (main 128.69 to 138.46 MSE, 0.40329 to 0.32107 bpp; holdout 43.93 to 59.02, 1.01974 to 0.69751). No PBC 3.1 all-metric success is claimed.

The observed finite frontier expands on useful individual images, with actual close-rate wins; this is more evidence than a static local utility or sparse interpolated mean. It remains a content-dependent alternative rather than a universally better preset. Extreme-compression defaults are unchanged and no universal low-bpp preservation claim is made for applying the allocator everywhere. Four known regressions remain explicit, broad high-quality holdout generalization is unresolved, and 25 previously studied images do not justify an adaptive classifier. Another broad fixed-weight sweep is not the next experiment. Retain the frozen setting, direct matched controls and negative cases for a future independent-image, tightly matched validation or a single targeted geometry/representation hypothesis.

RD plots rd_hpt_data/rd_hpt_data_val (PNG/SVG) show every measured prior/new point and the frozen candidate, with 5% tolerance bands. Their raw source is matched_rate_points.json/.csv. Dotted lines are guides; the JSON endpoints/weights are authoritative. No production defaults, decoder, model, RNG or previous result was changed.
