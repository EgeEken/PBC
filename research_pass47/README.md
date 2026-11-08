# PBC V3.1 native candidate oracle

Bounded research-only test of whether the canonical learned policy misses materially better patches within its own retained spatial proposals and native action set. It does not change codec syntax, production source, model, decoder, or presets.

Read `EXPERIMENT_DESIGN.md` before running. `scripts/prepare_codec_copy.py` copies the published implementation; `scripts/run_oracle.py` runs the control and bounded local screen; `scripts/measure_control_decode.py` adds control decode timing. Generated states, streams, reconstructions and raw rows belong in ignored `outputs/`. The canonical source and model provenance are linked to the Pass 45 manifest. No continuation or selection-rule phase ran because the local gate failed.
