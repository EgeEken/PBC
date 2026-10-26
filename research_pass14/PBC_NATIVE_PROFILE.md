# Pass 14 native optimization profile

## Scope and method

Profiled the frozen PBC 3.1 development source at `8006f30debfc4a57289df06df21636aeb52baffd`, baseline `pbc31_release_rng_baseline`. The six-image set was Cat, Googlewiki, Monet, Mario64, Eiffel, and Seagull, each with Quality and High Quality. The run used Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, one thread, CPU 0 affinity, one warmup, and three uninstrumented full calls per case. `pbc3_trace.py` supplied phase spans after the warmup. The raw events and per-case timings are in `profile_results.json`.

The figures below are the mean of the 12 per-case medians for full calls, and the mean of each phase's exclusive traced time across cases. Exclusive times avoid double-counting nested spans. Trace timings explain cost distribution; only the uninstrumented measurements are used for full-call timing.

## Decoder

Mean full decode median: **44.0 ms**. The largest phase is Pillow's bicubic final resize from the working image to source resolution: **24.2 ms/case** (about 55% of full decode). It already runs in Pillow's compiled code. Next are patch resampling/application (**9.4 ms**; Pillow resampling plus array update), patch read and grid unpack (**6.0 ms**), and LZMA unpack (**5.9 ms**, Python `lzma` backed by liblzma). Canvas-to-image conversion averages 1.6 ms.

These results do not support a native rewrite of the final resize or entropy unpack. A replacement would need to match Pillow's bicubic behavior or liblzma while preserving output pixels; both targets are already implemented in compiled libraries.

## Encoder

Mean full encode median: **475.9 ms**. The largest exclusive spans were LZMA packing (**86.0 ms**, liblzma), integral-error construction (**83.4 ms**, NumPy cumulative sums/padding), sampled-box generation (**51.9 ms**, Python loop and RNG calls), learned feature extraction (**47.5 ms**, NumPy/Python), final MSE (**33.3 ms**, array/compiled operations), and final resize (**24.0 ms**, Pillow).

The **single highest-value native candidate** is the repeated `sample_box` loop inside proposal generation. It is the largest clearly Python-heavy span and accounts for roughly 11% of measured full encode time. That is only an upper bound: a native implementation must preserve NumPy generator state and the conditional order of `uniform`/`integers` calls, and a Python/native boundary could consume part of the gain. The larger adjacent integral-image span is already in NumPy's compiled implementation. No measured result shows that translating box sampling would meet the 10% whole-call gate.

## Decision

Do not add a native dependency on this evidence. The Rust decoder audit below failed exact reconstruction and was much slower due in part to its LZMA subprocess fallback. A faithful native proposal sampler could be a later bounded experiment, but it needs an RNG-compatible extension design and a direct full-encode comparison before integration.
