# RNG-controlled speed attribution

The current-root NumPy-RNG variant matches the immutable release on all 100 image/preset cases and all three repeats: complete files, decoded RGB pixels, initialization/iterative patch payloads, counts and order. Every run also matches its frozen Pass 1 release file. This establishes output preservation for these default configurations and this local dependency/CPU environment; it does not extend to arbitrary warmup, overflow or custom settings.

The only experimental algorithm substitution is `pbc3_ops.PBC3Rng = numpy.random.default_rng` in a separate current-root worker. Production RNG/defaults are unchanged. Controls are release `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51` and root `3e9e8c94b7497ecefc5e482a66fde697dfad6e19`. Model SHA-256 is `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`.

Across all 25 images/four presets, mean per-case median encode changes -10.29% and decode -19.31% with identical output. These are combined engineering benefits. They do not individually attribute canvas storage, delta reuse, bit I/O, model caching or tracing overhead, and cannot be subtracted directly from Pass 1's differently timed aggregate to assign an exact RNG share.

| Set | Preset | Cases | Release encode s | Root NumPy encode s | Encode % | Release decode s | Root NumPy decode s | Decode % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | compression | 20 | 0.4242 | 0.3850 | -9.2395 | 0.0660 | 0.0586 | -11.1457 |
| hpt_data | balanced | 20 | 0.5476 | 0.4908 | -10.3619 | 0.0876 | 0.0741 | -15.3753 |
| hpt_data | quality | 20 | 0.5799 | 0.5140 | -11.3740 | 0.1099 | 0.0880 | -19.9831 |
| hpt_data | high_quality | 20 | 0.6293 | 0.5594 | -11.1050 | 0.1779 | 0.1347 | -24.2760 |
| hpt_data | all | 80 | 0.5452 | 0.4873 | -10.6271 | 0.1104 | 0.0889 | -19.4788 |
| hpt_data_val | compression | 5 | 0.3542 | 0.3334 | -5.8823 | 0.0574 | 0.0533 | -7.2601 |
| hpt_data_val | balanced | 5 | 0.3526 | 0.3220 | -8.6843 | 0.0736 | 0.0619 | -15.9675 |
| hpt_data_val | quality | 5 | 0.4851 | 0.4313 | -11.0975 | 0.0912 | 0.0742 | -18.6815 |
| hpt_data_val | high_quality | 5 | 0.5240 | 0.4826 | -7.9005 | 0.1509 | 0.1147 | -23.9958 |
| hpt_data_val | all | 20 | 0.4290 | 0.3923 | -8.5487 | 0.0933 | 0.0760 | -18.5377 |

## Timing and component evidence

Three interleaved warmed repeats, persistent source-isolated workers, CPU affinity 0 and numerical thread settings 1. Input loading and independent RGB scoring/parsing are outside timers; complete compress/decompress calls are inside. Warmup rows retain first invocation costs, including any disk-cache loading/compilation. This is not a pristine cold-start test. Per-case mean/median, extrema and CV are retained by the analysis script; source/model/config/dataset hashes and actual module paths are in rng_control_metadata.json and rng_control_provenance.json.

Separate encoder cProfile runs on flower, Monet and googlewiki, all presets, two repeats, compare individual function self/cumulative times and call counts in rng_profile.json. Profiled full timings include profiler overhead and are not primary latency estimates. Nested cumulative times must not be summed. Independent candidate ablations are in PBC_PERFORMANCE_RESULTS.md.

| Source | Function | Self s/case | Cumulative s/case | Calls/case |
| --- | --- | --- | --- | --- |
| numpy_root | sample_box | 0.0578 | 0.0696 | 4800.0000 |
| numpy_root | final_mse | 0.0029 | 0.0416 | 1.0000 |
| numpy_root | write_grid | 0.0758 | 0.0905 | 26.3333 |
| numpy_root | channel_sum_error | 0.0136 | 0.0230 | 26.3333 |
| release | sample_box | 0.0600 | 0.0717 | 4800.0000 |
| release | final_mse | 0.0028 | 0.0445 | 1.0000 |
| release | write_grid | 0.0703 | 0.1944 | 26.3333 |
| release | channel_sum_error | 0.0113 | 0.0242 | 26.3333 |

## Bounded HF discrepancy investigation

Retained original-RGBA Quality files were inspected locally. First quantized payload divergence: patch 14, zero-based, after four initialization patches. Earlier payloads reconstruct identically and the earlier prefix reserializes identically. Channel errors immediately before divergence: [933617.0, 952920.0, 841126.0, 0.0]. The local maximum anchor block score is shared by 1 blocks. Full candidate geometry, RNG state and ranked proposals are retained in hf_discrepancy.json.

| Local variant | Remote bytes equal | Saved local bytes equal | First divergence from remote |
| --- | --- | --- | --- |
| default_warm_cache | False | True | 14 |
| numpy_rng | False | False | 4 |
| float64_mlp | False | True | 14 |

CPU-dispatch control: `{"disabled_numpy_features": "AVX2,FMA3", "first_difference_from_remote": 9, "local_equal": false, "remote_equal": false, "stream_sha256": "24d0669fe77a4730207400331a90a187c07127c160b34075c6b51751e155990e"}`.

Retained-prefix replay: `{"cutoff_anchor_score_multiplicity": 2, "first_second_proposal_score_gap": 0.001904842528470077, "remote_box_in_local_200_pool": false, "remote_box_rank_in_local_top50": null, "remote_width_height_in_local_pool": true, "source": "Replay only retained common prefix/local recorded RNG state; no encode and no remote request", "top20_anchor_tied_scores": [12.53125, 13.0625], "top_proposal_score_tie_count": 1}`. The remote rectangle is absent from the locally sampled 200-box pool, although its width/height occur. Local top-20 anchor scores contain ties, including a two-way cutoff tie; the best local proposal score is not tied. With one learned candidate, geometry is chosen before model action inference. Anchor tie ordering/RNG/platform differences are therefore a plausible proposal-stage source. Local CPU dispatch demonstrably changes output, but did not reproduce the remote file; a live causal attribution is not established.

These checks use no uploads, remote compression calls, dependency installation or running-process introspection. Local float64 MLP inference and the release RNG are bounded diagnostic substitutions, not codec candidates. Equal earlier quantized patches constrain the divergence to later selection; they do not prove that every transient proposal/RNG state matched the remote process. The remote process's installed NumPy/BLAS versions, enabled CPU dispatch and loaded module/model state are not observable through the retained artifacts.

The next diagnostic would need the running process's exact dependency/CPU-dispatch and loaded input/model hashes, plus channel, RNG state, top-20 anchors and sampled proposals at iterative step 11 for this saved input. That requires explicitly authorized server introspection or an opt-in diagnostic run; it is not performed here. The subtask stops at this evidence boundary.

Reproduce: `python research_pass2/runner.py rng`; `python research_pass2/analyze_rng.py equivalence`; `python research_pass2/analyze_rng.py profile`; `python research_pass2/analyze_rng.py discrepancy`. For CPU dispatch, start a fresh process with `NPY_DISABLE_CPU_FEATURES=AVX2,FMA3` and run `python research_pass2/analyze_rng.py cpu_variant`. Raw evidence: rng_control.jsonl, rng_equivalence.json, rng_paired.json, rng_profile.json, hf_discrepancy.json and hf_cpu_variant.json.
