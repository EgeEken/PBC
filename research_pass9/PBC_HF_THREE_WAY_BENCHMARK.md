# HF comparison deferred

No Pass 9 HF comparison was run. The deployment idle-state guard found the existing ten-trial public run `20261026_141310`. The user authorized stopping it and asked to log the findings for now. It stopped without error at 996/9,800 calls; completed rows were backed up and checked. No HF push, restart or default change occurred.

HF repository and running revision remain `27f6c65225a5496fa065deb586c7e405ecdffc84`, cpu-basic. The prepared child `4237d954f4e1a04951c822c738113a53bbd58e9b` exists locally only. Its six paths add per-encode opt-in selection and bounded authenticated comparison support. It retains the current public RNG, presets, internal MSE, decoder, model, UI and database handling. `hf_deployment.json` identifies the exact proposed tree; it is not deployment evidence.

The controlled timing table and speed-reduction percentages cannot be supplied without measurements. Historical Pass 8 latencies and the existing public-run chart are not substituted. The chart and its complete first-trial means are logged separately in PBC_EXISTING_SPEED_RUN_LOG.md; its PNG screenshot bar contains only five images.

The ready script fixes cat, Eiffel, flower, googlewiki, Mario64, Monet, ratedistortion and seagull; all four presets; release, dev_release_rng and dev_current_rng; three warmed rotating repeats (288 calls). Before timing, run the three-image correctness smoke and extend equivalence to all 20 main/five holdout images. Separate persistent source processes share CPU affinity and numerical-thread constraints, original inputs and immutable model. Complete compression retains internal MSE; complete decoding retains final resize. File loading, external scoring/hashing/parsing, IPC and HTTP remain outside timers.

On resumption, reconfirm the current HF revision and that all jobs are idle, export the latest database before deployment, review the prepared child diff, then follow README.md. Do not silently deploy or promote a default from this deferred result. The optional decoder profile was not run.
