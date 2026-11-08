# AV1 Search Profile

## Instrumentation and coverage

The profile build enables libaom `CONFIG_COLLECT_PARTITION_STATS=2` and `CONFIG_COLLECT_COMPONENT_TIMING=1`, plus `PBC_PROFILE`-guarded counters in `partition_search.c`, `nonrd_pickmode.c`, and `intra_mode_search.c`. The patch does not change decisions or bitstream code. The instrumentation-off research control rebuilt with the counters compiled out retained exact reference bytes. Raw logs, CSVs and AVIF outputs are under `outputs/profile/`.

At speed 8 the active path applies variance-based precomputed partitions through `av1_nonrd_use_partition`; the ML/non-RD partition picker `av1_nonrd_pick_partition` is inactive. Across five cases the applied pattern had 101,535 NONE, 33,864 SPLIT, 101 VERT, and 29 HORZ nodes. Hybrid intra routed 60,439 blocks to RD intra and 41,264 to non-RD intra. RD luma loops considered 3,686,779 candidates, fully evaluated 207,655, pruned 3,479,124 before full transform/RD, and selected 60,439. Non-RD luma loops considered/evaluated 165,056 candidates, with 41,264 winners and no candidate pruning in this configuration.

Speed 6 takes the RD partition path. Its per-block-size CSVs record attempts and selected partitions; totals are:

| Speed 6 image | Partition attempts N/H/V/S | Selected N/H/V/S |
|---|---|---|
| cat.jpg | 39,564 / 7,500 / 2,343 / 11,002 | 34,660 / 3,047 / 203 / 5,639 |
| googlewiki.png | 8,959 / 3,657 / 3,516 / 3,001 | 8,224 / 985 / 270 / 2,032 |

Across these two cases RD luma loops considered 4,942,891 candidates, fully evaluated 216,650, pruned 4,726,241, and selected 64,123.

## Time distribution

At speed 8, `encode_sb_row_time` was 85–90% of libaom frame time. `nonrd_use_partition_time` was 79–88%; `hybrid_intra_mode_search_time` 36–63%; variance partition setup 0.8–1.8%; bitstream finalization 5–9%. These timers nest and must not be summed. At speed 6, RD partition search was 97–99% of `encode_sb_row_time`; RD intra mode evaluation was 10–16% of internal frame time.

One warmup and three interleaved timing repetitions measured median complete-call instrumentation overhead of 1.3% at speed 8 and 500.9% at speed 6. The speed-6 instrumentation is highly intrusive (individual medians rose about 411% and 591%); those internal times show path ordering only, not production runtime. All outputs remained byte-identical.

## Interpretation

1. RD intra search already rejects most candidate iterations before full transform evaluation; a feature-based change must improve on existing HOG, masks, angle-RD and model-RD pruning.
2. Speed-6 RD partition search dominates internal time; any speed change must be measured with profiling off.
3. Speed-8 variance partition setup is a small part of total encode time and is a weak speed target here.

Speed-8 partition-stat CSVs are zero because that configuration bypasses RD partition candidate search. They are nonzero at speed 6; this confirms speed-specific path coverage.
