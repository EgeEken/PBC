# Pass 47 frozen experiment design

## Objective and control

Determine whether the canonical PBC 3.1 encoder's existing learned spatial proposal and generated-grid action space contains materially better decisions than its ordinary selected patch. No codec or stream change is planned by this experiment. Primary control source is the published release Python under `research_pass45/outputs/published_release_python/`; model SHA-256 is `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`; presets are the canonical `compression()` and `balanced()` factories with no overrides except the explicit model path if required to load that exact artifact.

The source files are the three declared Pass 47 development images:

| Input | SHA-256 |
|---|---|
| `hpt_data/coast.jpg` | `7cb3bd541613e2f048a49add80d6e83109686b6744f490d894938ac887765ea2` |
| `hpt_data/googlewiki.png` | `cb2cc48d6a03b603d217b59fc4a3d3e53c6bebdff09e0a8172640bf85ccde94e` |
| `hpt_data/mario64.png` | `19e195de8f46fc4c4ef19925461c21da5769d682b152c967a783d5d3868c00d9` |

Before encoding, each input is EXIF-transposed and converted to RGB exactly as the frozen Pass 46 canonical benchmark does; source-file hashes are checked before normalization.

The current environment has Python 3.13.5, NumPy 2.2.6, Numba 0.62.1 and Pillow 11.3.0. Pass 45/46 archived measurements used Python 3.11.16, NumPy 2.4.6 and Numba 0.68.0; the compatible Python 3.11 runtime is not installed here. Every Pass 47 control, instrumented control, alternative and counterfactual will therefore be run under the same available environment. Archived byte/pixel hashes are a cross-runtime diagnostic, not the direct comparison baseline. If instrumentation changes a same-runtime control stream or decoded RGB pixels, stop before oracle evaluation.

## Canonical selection path

The published model is the legacy 30-action model (no `head_sizes` in the NPZ), not the newer factored-head policy. For each patch step the canonical search draws up to 200 random boxes, prescores and retains up to 50, sorts those by prescore, and passes only the first retained box to the learned filler (`learned_filler_candidates=1`). The filler extracts one feature vector, predicts the 30 action logits and a value, selects the logit argmax, builds one native patch, and compares that patch's exact working-canvas SSE reduction minus `lambda_for_q(q, model.lmin, model.lmax) * patch_bits`. `learned_filler_top_k=1`; with one box this does not evaluate additional model-ranked boxes. The predicted value does not compare actions. The actual selected patch is then applied before the next step.

This experiment will separately record the random sample count, retained prescore pool size, boxes passed to the oracle, learned model evaluations, action predictions, exact native reconstructions scored, and ordinary selected patch. It must not call the 200 samples “200 full patch evaluations.”

## Fixed sampled states and bounded oracle

For each of three inputs and each of Compression/Balanced, inspect steps 1, 25 and 50, fixed before seeing results. There are at most 18 states. If the ordinary encoder stops before a scheduled step, record the state as unavailable; do not substitute a favorable step. At each present state capture the exact target/canvas, channel, step, config, model state inputs, RNG state immediately before proposal generation and immediately after the ordinary choice, proposal pool, and ordinary selected patch/action. RNG snapshots are for replay verification; instrumentation must not draw random numbers or reorder canonical calls.

At each state evaluate at most four boxes: retained proposal ranks 1–4 from the same canonical ordered pool. For each box evaluate at most two actions: the model's highest-logit action and its second-highest-logit action. The ordinary box-1/model-top-action patch is always included. No new proposal generator, additional cell-size/bit-depth sweep, or new representation is allowed. Maximum is eight exact patch constructions per state. A model forward pass may be computed for each of the four boxes solely to obtain its top two native actions; the canonical path computes one.

For every evaluated candidate, record its spatial rank, model value and action ranks, patch parameters, exact patch bit cost, native reconstructed delta, working-resolution SSE reduction, and construction/feature/model timing. The local RD score is `SSE_reduction - lambda * exact_patch_bits`, using the canonical legacy `lambda_for_q` and this preset's ordinary per-step q. Also report gain/bit and Pareto dominance against the exact ordinary selected patch. This is a local proxy, not a complete-file rate or trajectory result. Where available, record one-step original-resolution RGB effect by applying the candidate to a copy of the current canvas and using the canonical output conversion/resize path.

## Local gate and conditional continuation protocol

Gate B passes only if at least 3 of 18 scheduled states, spanning at least two inputs, have an evaluated alternative with at least 5% higher positive local RD score than the ordinary patch and no more than 5% greater exact raw patch bits. If ordinary score is nonpositive, the state is reported but does not qualify. Do not count multiple alternatives at one state as multiple wins.

If Gate B passes, select at most six counterfactual states before looking at any continuation result: rank qualifying state/candidate pairs by absolute local RD-score gain, keep at most two per image and require at least two images. Include one highest-ranked qualifying action alternative and one highest-ranked qualifying spatial-box alternative when both exist. If fewer than two images have qualifying pairs, no completed continuation is authorized by this design. Recreate the canonical state by deterministic full encode, verify its state/RNG/proposal and selected patch against the capture, then replace exactly one chosen patch. Continue all later steps with the unchanged canonical policy. Later proposal pools may differ naturally due the changed canvas; their RNG starts from the same post-choice generator state. Do not freeze later canonical patches.

Use at most six alternative full continuations total. Pair each with the saved same-runtime canonical file. Decode each file through a fresh process using the unchanged published source. Record completed bytes, bpp, original-resolution RGB MSE/PSNR, pixel/stream hashes, SSIMULACRA2 when available, complete encode and decode times. Compare actual file rates; ±3% is preferred, ±5% maximum. A screened improvement requires at least 5% lower MSE at comparable bytes (or equivalent smaller size at similar MSE) and perceptual evidence without a conspicuous visual regression. No interpolation or extrapolation is a completed-file win.

## Stop rules and budget

Gate A is same-runtime byte and pixel identity between the uninstrumented canonical encoder and instrumentation-only control on all six cases, plus deterministic repeat of both. Failure stops. Gate B uses only the fixed sample above. Gate C is the completed-file threshold above; if local opportunities fail Gate B, no full continuation runs. Gate D requires validated SSIMULACRA2 or documented source-aligned visual crops. Gate E (a cheap deployable rule) is not attempted unless Gate C passes on at least two images. Gate F (competitive AVIF comparison) is reported only for complete candidate files; AVIF q0 and pinned near-rate references remain distinct, and AVIF timing with CLI/I/O stays labeled not directly comparable.

Total development screen: 3 images × 2 presets × 3 fixed steps × at most 8 patch constructions. Conditional ceiling: 6 alternative continuations, no additional broad corpus, no model training, no production changes, and no Pass 48.
