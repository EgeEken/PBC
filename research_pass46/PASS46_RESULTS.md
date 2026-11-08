# PBC V4 Pass 46 results

## 1. Objective and scope

This pass tested a narrowly defined question: can a compact source-resolution procedural stroke recover perceptually important detail at canonical PBC 3.1 Compression rates? It also rechecked canonical PBC output, compared two distinct AVIF quality-0 configurations, and added pinned AVIF points around PBC's measured rates. No codec architecture or stream was changed.

## 2. Canonical reference

The reference was the published-ZIP embedded Python/model/preset runtime pinned in Pass 45, not a mutable checkout implementation. Pass 46 reproduced Coast Compression's completed size (6,461 bytes), stream SHA-256 (`792e9602ba3c4a407c143c1e1e2fdb6e15d35ae1ca8ad2446303770a0833a029`), decoded RGB SHA-256 (`0c0527f2fb9f95694509acd5fe91b1440126c80b068bd1476a77970e7150f655`), and MSE (73.1336). All three source file and RGB hashes are in the raw baseline JSON. PBC rows use three complete-call encode and decode samples; AVIF CLI timing includes process launch and I/O.

## 3. Perceptual and rate baseline

PBC Compression/ Balanced files were 6,461/11,729 bytes for Coast, 28,974/53,890 for Googlewiki, and 14,099/38,900 for Mario64. Pinned AVIF qcolor=0 is a quality parameter, not a fixed-rate setting: its corresponding files were 6,747, 5,218, and 4,139 bytes. Pillow quality=0 is reported separately.

At five near-rate pinned AVIF points (within 4.0% of PBC), AVIF had lower RGB MSE and higher SSIMULACRA2: Coast Balanced vs qcolor 7; Googlewiki Compression vs qcolor 12 and Balanced vs qcolor 28; Mario64 Compression vs qcolor 20 and Balanced vs qcolor 50. This three-image result does not establish broad AVIF superiority, but it gives no support for a source-resolution stroke mode as a competitive path against these controls. Quality-0 comparisons are image dependent and must not be described as rate matched.

## 4. Structural-detail experiment

The tested representation was a straight residual stroke with compact endpoint, width, and signed RGB amplitude parameters. It used deterministic coverage and the existing shared XZ/LZMA wrapper in isolated packets. The best single stroke removed 0.221% of Coast base SSE, 0.106% of Googlewiki, and 0.943% of Mario64. Groups of eight removed 0.889%, 0.327%, and 1.645%, but the selected group compressed to 164 bytes, and these packet sizes are not complete-file measurements.

The frozen gate required individual useful corrections, visible structural restoration, at least three wins across two content classes, and efficiency against a native PBC result with comparable gain. It failed. Only one individual correction passed the numeric 0.5% threshold; none of the 24 candidates had a native output within the gate's 80%-to-125% gain bracket. Reported gain/byte ratios against much higher-gain native outputs are not matched-gain evidence. Visual changes were subtle. Therefore no stroke mode, full codec, new decoder, or stream change was implemented.

The frozen design prose and runner also differed on shortlist ranking: the design says top eight by gain, while the runner ranked by gain per compressed packet byte. This deviation is disclosed in `FEASIBILITY_RESULTS.md` and limits the result. It does not reverse the failure evidence. A follow-up would need a newly frozen protocol if pursued.

## 5. Perceptual checks

SSIMULACRA2 used one pinned binary for all RGB PNG inputs. Group scores improved only 0.628, 0.768, and 0.968 points respectively over the canonical base. Source-aligned crop inspection found the fishing rod remained softened, the Googlewiki text/logo remained blurred, and Mario64 fence edges changed only subtly. No clear halo or tear was observed. Local stroke reconstructions are diagnostics, not complete V4 results.

## 6. Release consistency checks

The Pass 45 canonical Coast stream and pixel hashes match. The public HF Space was reachable but sleeping; no restart or upload was performed, so live behavior is not verified. The official release ZIP and executable were present, but desktop launch approval timed out; the executable was not run. No release/deployment files were changed.

## 7. Gates

| Gate | Result | Evidence |
|---|---|---|
| Canonical measurement integrity | Pass | Coast stream and pixel hash reproduced; source hashes checked |
| Structural local feasibility | Fail | Required independent useful wins, matched-gain native evidence, and persuasive visual restoration absent |
| Complete-file RD | Not attempted | Local gate failed; no complete codec implemented |
| Decode compatibility/performance | Not applicable | No decoder syntax or integration added |
| HF live consistency | Not verified | Space sleeping; no restart/upload |
| Published executable consistency | Not verified | Launch approval timed out; executable not run |
| Broad/holdout validation | Not attempted | Development feasibility gate failed |
| AVIF niche superiority | Not established | AVIF near-rate points were stronger on the three sampled images; no broad or timing claim |

## 8. Recommendation

Do not integrate straight strokes. The immediate evidence favors pausing this primitive and returning to a measurable source-resolution residual representation that offers much larger deterministic gain per byte, while retaining a frozen full-file budget and visual gate. Before spending effort on a new primitive, inspect whether the native cell-grid control can be better allocated locally without adding a mode, or define a different structural proposal family in a fresh experiment. Do not claim PBC superiority over AVIF from q0 labels or these three images. Any future AVIF niche claim needs completed files at measured near-rates, source-resolution quality and controlled encode/decode boundaries on a declared image class.
