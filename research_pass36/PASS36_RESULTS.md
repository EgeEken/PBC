# Pass 36 results

## Objective and provenance

Profile the pinned libaom speed-6 AVIF path with optimized native symbols, inspect repeated computations, and make at most one exact-output reuse change if evidence supports it.

- Branch: `research/experiments`
- Starting HEAD: `3723dc14de88c6c3dcbd98dd80a097e019a92a9f`
- libaom: 3.12.1, `10aece4157eb79315da205f39e19bf6ab3ee30d0`
- libavif: 1.3.0, `1aadfad932c98c069a1204261b1856f81f3bc199`
- Toolchain: Windows 11 x64, Strawberry MinGW-w64 GCC/G++ 13.2.0, CMake 3.29.2, Ninja 1.12.0; 20 logical CPUs.

Pass 35's untracked directory, sources, binaries, and outputs were preserved. PBC production source, model weights, GUI, releases, and HF files were not changed.

## Build and correctness

An isolated libaom archive was built with `-O3 -DNDEBUG -g`, static linkage, and no Pass 35 runtime counters or timers. The existing libavif 1.3.0 objects and dependencies were linked to that archive. The symbol build reports the pinned libavif/libaom versions. The ordinary and symbol-enabled executables have identical `.text` bytes (9,372,472 bytes; SHA-256 `eefd91d43a9f6cc2aaea272201cd882af3098be5ade2626736fbb91ee4ac962c`).

Fresh reference `avifdec` processes decoded all parity files. Complete AVIF hashes and RGB pixel bytes matched on all three checked operating points:

| Image | qcolor | Speed | AVIF bytes | AVIF SHA-256 | AVIF bytes equal | Decoded RGB equal |
|---|---:|---:|---:|---|---|---|
| googlewiki.png | 5 | 6 | 20,613 | `26c7e64a185987671beb47cc6a7f31ad018207bc4d4fac3198be90d6a00d31a1` | yes | yes |
| cat.jpg | 35 | 6 | 262,385 | `2747700288e5f84b2a44f6916292e87ab020320ea2bf7bf8e51e70e22d678881` | yes | yes |
| googlewiki.png | 5 | 8 | 15,985 | `8580f1bc71f624ab7370eaa2191536996204b1d2157ce4a501f2e8b708f3be2e` | yes | yes |

The equality results validate that `-g` did not alter encoder machine code or decisions for these settings. Dimensions were also verified after decode.

## Profile findings and reuse decision

WPR sampled one configuration only: 25 consecutive Googlewiki q5 speed-6 encodes. The observed path is dominated inclusively by recursive RD partition evaluation and its intra/transform descendants. Leaf samples spread across coefficient cost updates, intra prediction, allocation, and transform search. Function names, sources, sample shares, existing pruning, and limits are in [RD_SEARCH_PROFILE.md](RD_SEARCH_PROFILE.md).

The clearest source-level repeat is the local `log_sub_block_var` call for an all-intra 64×64 block, repeated before and after partition pre-pruning. It is deterministic over the same source block, but sampled below 1% inclusive in this case. It was not implemented. Other transform/RD candidates depend on reconstruction and coding state, and libaom already has conditional result reuse. The candidate review is in [COMPUTATION_REUSE_CANDIDATES.md](COMPUTATION_REUSE_CANDIDATES.md).

No encoder source modification was attempted. Therefore no modified-versus-reference complete-encode benchmark, memory comparison, held-out test, or speed-8 modification control was run. There is no measured encode-time improvement or regression to report.

## Profiling completion and decision gates

| Gate | Result | Reason |
|---|---|---|
| A — Reliable profiling | **Not passed** | One of five configurations and one independent trace were completed. The default WPR verbose profile had an estimated 13.6% CPU overhead. A later CPU.light helper did not launch, leaving no full matrix or sampled speed-8 control. |
| B — Safe computation reuse | **Not attempted** | The repeated variance helper is exact but under 1% inclusive in the single case; no candidate had a supported complete-encode benefit. |
| C — Complete-encode benefit | **Not evaluated** | No encoder modification was made. |

Held-out images were not encoded.

## Recommendation

Do not promote a reuse patch from this evidence. First complete the five-case, two-capture WPR CPU.light profile and measure that configuration's overhead. Based on the available samples, the next decision-changing AV1 hypothesis should target candidate ordering within `av1_rd_pick_intra_sby_mode` after existing HOG/mode/model-RD pruning. Keep existing AV1 RD as the final choice; test whether evaluating strong estimated candidates earlier improves the speed/RD frontier without hiding useful modes. This remains a proposal for a later pass, not a Pass 36 implementation.

## Worktree

At completion, the only Git status entries are the preserved untracked `research_pass35/` and new untracked `research_pass36/` directories. No files were staged, committed, or pushed. The worktree remains at the starting HEAD.
