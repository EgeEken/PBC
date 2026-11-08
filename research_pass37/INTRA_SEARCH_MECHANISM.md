# Intra search mechanism

Pinned source: libaom 3.12.1, commit 10aece4157eb79315da205f39e19bf6ab3ee30d0.

av1_rd_pick_intra_sby_mode() in av1/encoder/intra_mode_search.c is active for the AVIF all-intra speed-6 path. Its caller is av1_rd_pick_intra_mode_sb() in rdopt.c; the partition path reaches it through hybrid_intra_mode_search() in partition_search.c.

The current base-mode order is fixed in intra_rd_search_mode_order: DC, H, V, Smooth, Paeth, Smooth-V, Smooth-H, then six directional modes. set_y_mode_and_delta_angle() follows these with directional angle variants. When odd-angle pruning is enabled, even angles are visited before odd angles because prune_luma_odd_delta_angles_using_rd_cost() consults neighboring angle RD results. Pass 37 leaves the angle sequence untouched.

Before full transform/RD evaluation, the function applies configuration and speed-feature eligibility checks, an optional MB mode cache, HOG directional masks, the mode mask, and odd-angle RD pruning. It then calculates intra_model_rd() and calls prune_intra_y_mode(). That helper updates the rolling top-model-RD array, then rejects candidates against the selected top estimate and the best model RD so far. A candidate that passes reaches av1_pick_uniform_tx_size_type_yrd(), which builds the actual prediction and performs the expensive transform/RD search. The top-model list and best-model bound therefore make the number of full evaluations order-dependent: weak modes can be fully searched before stronger estimates tighten those bounds.

The search also retains the best actual RD result and copies its mode, block-skip map, and transform-type map. Later candidate evaluations mutate mode and transform scratch state. Pass 37 changes visit order only; it does not reuse candidate state, skip a candidate directly, or change transforms, syntax, reconstruction, or decoder behavior. CTRL output parity and EXP decoding are required to detect unintended effects.

Feasibility counters reused the Pass 35 aggregate diagnostic build (diagnostic timings are not used). On Googlewiki q5/s6, 21,967 RD intra blocks entered and 64,649 transform/RD calls were attempted. On Cat q35/s6, 59,064 blocks and 152,001 calls were observed. The counter evaluated is incremented immediately before av1_pick_uniform_tx_size_type_yrd(); pruned is a union of early rejection reasons and is not a count of only modeled-RD pruning. These counters establish active work, while source inspection establishes ordering sensitivity.

The cheap signal selected is the existing per-mode syntax cost bmode_costs[mode], already available in the block's above/left entropy context. Sorting 13 base modes requires no prediction pass or new pixel analysis. This is a plausible seed for the actual-RD and model-RD pruning state, but mode signaling cost may correlate weakly with prediction quality. The experiment is designed to reject the rule if search counts or whole-encode results do not improve.
