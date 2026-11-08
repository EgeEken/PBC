# Profiling method

## Sampling and symbolization

The primary sampler was Windows Performance Recorder 10.0.26100 using its built-in `CPU` profile, which selected `CPU.Verbose.Memory`. The trace contains 1 ms sampled-profile events with kernel stack walks. The trace was converted and aggregated with the already installed `Microsoft.Diagnostics.Tracing.TraceEvent.dll` 2.0.74.0 from HP Overlay; no profiler was installed. GNU `addr2line` from Strawberry GCC 13.2.0 symbolized MinGW DWARF addresses.

The ordinary reference `avifenc.exe` was profiled for 25 repeated Googlewiki `qcolor=5`, speed-6 encodes. The trace has 7,011 encoder-process samples over 7.049 CPU seconds. Encoder-module frames were present in 6,894 samples (98.3%); 6,480 samples (92.4% of all samples) resolved to functions. The ETL sample interval reported by TraceEvent was 1 ms. The capture used one encoder thread. CPU affinity was not fixed for this trace.

`scripts/analyze_wpr.ps1` reads WPR ETL, gathers per-process sampled stacks, and uses DWARF in the symbol build to resolve functions. It counts each function once per sampled call stack for inclusive estimates and counts the sampled instruction's top frame for exclusive estimates. Inclusive percentages overlap and must not be summed.

## Symbol build and correctness

The symbol archive was built from Pass 35's pinned libaom 3.12.1 checkout with GCC 13.2.0, `-O3 -DNDEBUG -g`, Release configuration, NASM enabled, static libaom, AV1 encoder/decoder on, and tools/tests off. `PBC_PROFILE` and Pass 35 timing/counter options were not enabled. The existing pinned libavif 1.3.0 app objects and static dependencies were reused; `avifenc-symbol.exe` was manually linked against `research_pass35/install/symbol/lib/libaom.a`.

The profile executable's `.text` section is 9,372,472 bytes and byte-identical to the ordinary reference executable's `.text` section (SHA-256 `eefd91d43a9f6cc2aaea272201cd882af3098be5ade2626736fbb91ee4ac962c`). This keeps Release machine code unchanged while adding DWARF symbols. Static linkage is established by the explicit archive path in the link command and the local link output; `avifenc --version` reports libavif 1.3.0 and libaom 3.12.1.

Before using the symbols, three settings were encoded with both binaries and decoded in fresh `avifdec` processes. Complete AVIF hashes and decoded RGB pixels matched for Googlewiki q5 speed 6, Cat q35 speed 6, and Googlewiki q5 speed 8. Details and hashes are in `outputs/symbol_parity/result.json`.

## Overhead and limits

The verbose WPR trace's median `TraceProcess.CPUMSec` for the 25 Googlewiki speed-6 runs was 284 ms. A matching unprofiled run used 25 fresh processes after three warmups; median `Process.TotalProcessorTime` was 250 ms. The rough difference is +13.6%. This comparison is sensitive to CPU frequency and process-time quantization, but it is high enough that this WPR configuration does not qualify as low-overhead timing. No timing from the profiled run is used as an encode-speed result.

WPR's less verbose `CPU.light` profile was available, but the later elevated helper did not launch: it created no output directory, and `wpr -status` reported no active recording. No WPA, PerfView, or VTune analyzer was installed. The available ETL is sufficient for one function-level profile, but not the five-configuration, two-capture coverage required by Pass 36. The full-matrix helper is `scripts/record_wpr_profiles.ps1`; it has not completed.

Raw ETL and ETLX files remain in the local temporary profile directory. The trace covers only Googlewiki q5 speed 6, so it cannot support speed-8 function percentages or generalization claims.
