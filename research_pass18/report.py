import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_pass18.diagnose import HERE, lf, provenance, sha, write_json


def main():
    diagnostics = json.loads((HERE / 'diagnostics.json').read_text())
    frozen = json.loads((HERE / 'frozen_results.json').read_text())
    representation = json.loads((HERE / 'representation.json').read_text())
    ledger = list(csv.DictReader((HERE / 'decisions.csv').open()))
    accepted = [r for r in ledger if r['accepted'] == 'True']
    iterative = [r for r in accepted if r['kind'] == 'iterative']
    attempts = [r for r in ledger if r['kind'] == 'iterative']
    distributions = {}
    for kind, records in [('iterative', iterative), ('initialization', [r for r in accepted if r['kind']=='initialization']), ('all', accepted)]:
        total = sum(int(r['grid_bits']) for r in records)
        distributions[kind] = [dict(bits=b, patches=sum(int(r['effective_bits'])==b for r in records), grid_bits=sum(int(r['grid_bits']) for r in records if int(r['effective_bits'])==b), grid_share_percent=100*sum(int(r['grid_bits']) for r in records if int(r['effective_bits'])==b)/total) for b in range(1,7)]
    same = [x for r in frozen['rows'] for x in r['same_region']]
    spatial = [x for r in frozen['rows'] for x in r['spatial']]
    frontiers = []
    for state in frozen['rows']:
        pool = [('ordinary', state['selected'])] + [(f"content_{i}_{x['kind']}", x) for i,x in enumerate(state['spatial'])]
        frontier = []
        for name, candidate in pool:
            if not any(other['gain']>=candidate['gain'] and other['raw_bits']<=candidate['raw_bits'] and (other['gain']>candidate['gain'] or other['raw_bits']<candidate['raw_bits']) for _,other in pool):
                frontier.append(name)
        q = .8 if state['preset']=='quality' else .95
        lam = lf.lambda_for_q(q)
        utilities = {name: x['gain']-lam*x['raw_bits'] for name,x in pool}
        frontiers.append(dict(id=state['id'], nondominated_evaluated=frontier, exact_gain_per_raw_bit_max=max(x['gain_per_bit'] for _,x in pool), gain_minus_lambda_bits=utilities, best_utility=max(utilities,key=utilities.get)))
    summary = dict(start_commit='2e7d03e47f9db4d501b1f4d2e2bf1b46a1b17735', baseline_id='pbc31_feature_histogram', completed_baseline_files=12, states=len(frozen['rows']), requested_action_attempts={str(b):sum(int(r['requested_bits'])==b for r in attempts) for b in range(1,6)}, requested_accepted_actions={str(b):sum(int(r['requested_bits'])==b for r in iterative) for b in range(1,6)}, distributions=distributions, same_region=dict(tested=len(same), different_1bit=sum(x['requested_bits']==1 and not x['exactly_same_patch'] for x in same), different_2bit=sum(x['requested_bits']==2 and not x['exactly_same_patch'] for x in same), strict_dominance=sum(x['strict_dominance'] for x in same), near_gain_cheaper=sum(x['near_gain_cheaper'] for x in same), useful_tradeoff=sum(x['useful_tradeoff'] for x in same)), spatial=dict(rectangles=sum(r['content_proposals'] for r in frozen['rows']), states_with_proposals=sum(r['content_proposals']>0 for r in frozen['rows']), exact_evaluations=len(spatial), binary_positive_gain=sum(x['kind']=='binary_cell2' and x['gain']>0 for x in spatial), strict_dominance=sum(x['strict_dominance'] for x in spatial), near_gain_cheaper=sum(x['near_gain_cheaper'] for x in spatial), useful_tradeoff=sum(x['useful_tradeoff'] for x in spatial), original_proposal_seconds_median=statistics.median(r['stochastic_seconds'] for r in frozen['rows']), content_proposal_seconds_median=statistics.median(r['content_seconds'] for r in frozen['rows']), frontiers=frontiers), representation=dict(native_gain_fraction_median=statistics.median(r['native_1bit']['gain']/r['selected_gain'] for r in representation['rows']), free_level_gain_fraction_median=statistics.median(r['free_level_gain_fraction'] for r in representation['rows']), zero_positive_palettes=sum(r['native_1bit']['palette'][0]==0 and r['native_1bit']['palette'][1]>0 for r in representation['rows']), residual_negative_sse_fraction_median=statistics.median(r['negative_residual_sse_fraction'] for r in representation['rows'])), decision='Stop after frozen feasibility. No integrated candidate, complete-file candidate screen, timing repeats, holdout, new format, production change or staging rebuild.')
    write_json('summary.json', summary)
    per_case = []
    for row in diagnostics['rows']:
        records = [r for r in iterative if r['image']==row['image'] and r['preset']==row['preset']]
        total = sum(int(r['grid_bits']) for r in records)
        counts = [sum(int(r['effective_bits'])==b for r in records) for b in (1,2)]
        share = 100*sum(int(r['grid_bits']) for r in records if int(r['effective_bits'])<=2)/total
        per_case.append(f"| {row['image']} | {row['preset']} | {len(records)} | {counts[0]} | {counts[1]} | {share:.4f}% | {row['bytes']:,} | {row['rgb_mse']:.3f} |")
    distribution_table = '\n'.join(f"| {r['bits']} | {r['patches']} | {100*r['patches']/len(iterative):.3f}% | {r['grid_bits']:,} | {r['grid_share_percent']:.6f}% |" for r in distributions['iterative'] if r['bits']<=5)
    (HERE/'PBC_LOW_BIT_DEPTH_ANALYSIS.md').write_text(f'''# Pass 18 low-bit-depth analysis

Stop at the frozen feasibility gate. The model rarely requests 1 bit, but these experiments do not show a broadly useful missed 1-bit opportunity. Two-bit patches are already common. No integrated replacement selector or new stream was implemented.

## Baseline and scope

Research start: `2e7d03e47f9db4d501b1f4d2e2bf1b46a1b17735`. Control: unchanged `pbc31_feature_histogram`, runtime/model byte-checked against Pass 16 commit `1c9c4016fdb213d7d16663542d268c7c3df4ed9b`. The actual model is legacy LearnedFiller, cheap=False, candidates=top_k=1, 30 actions. Its value head orders one proposal; termination is nonpositive exact gain, not a STOP head.

Six specified images, Quality/HQ, default seed 2003, 250k cap, YCbCr, initialization and internal RGB MSE enabled, normal LZMA. All expanded configurations, input/source hashes and environment are in diagnostics.json. Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0; i9-12900H logical P-core 12, numerical threads one. No actual-old-versus-new release claim is made: this research compares decisions within the current practical codec.

Twelve instrumented baseline encodes exactly reproduce Pass 17 full-file and independently decoded pixel hashes. The probe returns ordinary selections unchanged. It records every requested/effective precision, cell size, channel/rectangle, grid cells/bits, complete raw patch bits, local clipped channel SSE reduction, current whole-canvas clipped SSE, logits, proposal rank and acceptance. Raw logits/ledger remain local under ignored decisions_raw.json; frozen-state logits remain in diagnostics.json. decisions.csv contains every decision including four rejected stopping attempts. Local patch metrics are working-space SSE, not original RGB SSE.

## Actual decisions

There are 327 iterative attempts, 323 accepted iterative patches and 36 initialization patches. Requested maximum depths across attempts: 1/88/47/102/89 for depths 1/2/3/4/5. Accepted requested depths: 0/88/44/102/89. The sole requested 1-bit action, Monet Quality step 35, has zero gain and stops. This refines the initial observation about accepted actions: it is not zero requests across all attempts.

Actual accepted iterative payload:

| Effective bits | Patches | Patch share | Raw grid bits | Grid-bit share |
| ---: | ---: | ---: | ---: | ---: |
{distribution_table}

One-bit patches are 2.167% of accepted iterative patches but only 284 grid bits, **0.001546%** of iterative grid payload. Requested 2-bit actions are 27.245% of accepted actions; effective 2-bit patches are 28.793%, contributing **2.080634%** of iterative grid bits. The seven effective 1-bit patches came from larger requested maxima, not accepted 1-bit actions. They contain only 284 total cells.

Initialization is separate: 2 patches at 4 bits, 10 at 5, 24 at 6; 1,016,874 grid bits. Across all 359 serialized patches: 19,380,987 grid bits; 1-bit share 0.001465%, 2-bit share 1.971468%, and 6-bit initialization share 3.413510%. Initialization's 6-bit maximum is outside the legacy iterative action space.

| Image | Preset | Iterative patches | Effective 1-bit | Effective 2-bit | 1Ã¢â‚¬â€œ2-bit grid share | Final bytes | RGB MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(per_case)}

Four zero-gain stops occur at Cat Q step 19, Flower Q step 36, Monet Q step 35 and Eiffel Q step 17. HQ reaches its 20-patch budget for every image here. These current Python 3.11 trajectories differ from historical Pass 7's Python 3.13 sample; those historical counts were not substituted.

The whole-stream entropy wrapper is retained. Raw patch/grid bits do not assign compressed bytes to individual patches: LZMA savings depend on context. Final file bytes include initialization, metadata, padding and entropy framing. The validator parses every ordinary patch and checks exact bit positions against the ledger.

## Same-region counterfactuals

Captured steps 3/8/13/18 when reached: 47 positive accepted states. Target, canvas, selected rectangle, all 50 retained proposals, expanded config, logits and RNG before/after selection are preserved. Captured arrays stay ignored. Replaying stochastic proposals reproduces ordered boxes and RNG state.

Per state: ordinary action; requested 1 bit at the same cell; requested 1 bit at half cell size where distinct; requested 2 bits at the same cell. Only 120 alternative builds after deduplicating cell/bit pairs, never an exhaustive 30-action sweep. Seven builds exactly reproduce the ordinary requested representation; 73 are different requested-1-bit builds and 40 different requested-2-bit builds. Every candidate uses production construction, palette, quantization, bicubic reconstruction, clipping/SSE scoring and exact full raw patch bits.

Diagnostic categories, fixed before observing results: local dominance means at least as much gain at no greater raw cost, with one strict inequality; near-gain cheaper means >=98% gain at <=75% bits; useful tradeoff means >=90% gain at <=60% bits. These categories do not establish final-file RD.

- Same-cell requested 1-bit retained median **36.07%** of ordinary gain. No same-cell 1-bit alternative locally dominated or met either retention/cost category.
- One finer-grid 1-bit alternative dominates: Cat HQ step 8, cell 8 -> 4. Gain 11,728 -> 36,277, raw bits 4,725 -> 4,653. This is one geometry/precision opportunity, not a systematic same-grid precision error. It reduces an extra 24,549 working SSE against current total SSE 4,492,407; final RGB/entropy/trajectory consequences are unmeasured.
- One 2-bit near-gain opportunity: Googlewiki HQ step 8, cell 8, gain retention 99.622%, raw bits retention 69.097%.
- Two further 2-bit tradeoffs: Flower Q step 18 (90.861% gain/59.420% bits), Flower HQ step 13 (91.227%/56.250%). Neither is local dominance.

The strict 1-bit win is one of 47 states (one of 73 different 1-bit builds). Same-grid 2-bit savings occur only in three states, mostly trading away gain. This does not justify a general fallback, another weight sweep or an image-specific trigger.

## Why 1 bit fails here

In 46/47 same-cell tests, the actual generated 1-bit palette is zero plus one positive level. With mask[0] reserving zero and only one remaining palette entry, palette_generator assigns that entry to the first active nonzero range; negative ranges often receive none. The median negative residual share is 49.82% of squared residual energy. A single zero/positive residual correction cannot directly repair negative residuals. This is a concrete property of the present generated palette, not a claim that a 1-bit source image needs only positive corrections.

An explanatory, **unencoded** fit allowed two free signed levels on the same projected grid, initialized at extrema and updated four deterministic Lloyd iterations. Exact production interpolation/clipped SSE scoring retained median **52.83%** of ordinary gain, versus 36.07% for native 1 bit. Only three states retained >=90%; one is an already degenerate native representation. This is a small heuristic representation probe, not a globally optimal bound, a valid v0 palette, a serialized candidate or an RD result. Better two-level quantization alone does not recover ordinary gain across these states.

The working residual after initialization is often signed and multilevel even for graphic inputs. Fine-grid native 1-bit failures show that bicubic blur alone cannot explain the losses; palette capacity/placement and residual distribution also matter. No numerical causal share is assigned to quantization, interpolation, downsampling or later trajectory changes.

## Disposition

The measured model underselection of requested 1 bit is real, but useful missed alternatives are rare. Two-bit coding is already common in Quality Googlewiki (40/50 iterative patches). No completed-file candidate screen was warranted; no completed-file RD, runtime, generalization or holdout gain is claimed. Spatial and resolution diagnostics are in PBC_SPATIAL_PROPOSAL_RESULTS.md.
''',encoding='utf-8',newline='\n')
    cmed=summary['spatial']['content_proposal_seconds_median']*1000
    smed=summary['spatial']['original_proposal_seconds_median']*1000
    (HERE/'PBC_SPATIAL_PROPOSAL_RESULTS.md').write_text(f'''# Pass 18 deterministic spatial proposals

The tested contrast/binary-fit proposal generator did not identify a useful replacement for the ordinary chosen patch. It is rejected at frozen feasibility; this is a negative result for this one method, not proof that the stochastic search is optimal.

## Frozen method

Same 47 states and current channel as the low-bit analysis. The ordinary comparison is the actual ranked stochastic pool: search_depth 200, proposal_depth 50, followed by one model action/exact evaluation on the first retained box. RNG replay exactly matches the original ordered pool. Only retained 50-box equivalence is assessed; there is no claim about all unretained rectangles or all possible actions.

The deterministic method tiles the working signed residual into complete 32x32 tiles. For each tile, split at its mean and calculate group means and the fraction of centered variance explained by that split. Require explained fraction >=0.8, group-mean contrast >=24 and each group >=5% of pixels. Thresholds and geometry were fixed for the first run; no threshold/cell-count rescue sweep followed. The analyzed target enters through target-minus-current-canvas, rather than classifying source RGB colors as residual codes.

Four-connected eligible tiles produce component bounding boxes, capped at the existing 400-pixel maximum. Rank by explained uncentered residual SSE divided by estimated cell-2 grid cells plus metadata; retain at most three boxes. No neural training, exhaustive rectangles, source-name switch, second encode or new production field. Existing SearchState residuals differ from this signed residual map, so this prototype charges its own map/statistics; it does not claim reuse of a binary map already computed by production. Partial edge tiles under 32 pixels are excluded, an explicit limitation.

Each returned box receives exactly two production evaluations: its ordinary learned action and a fixed cell-2 1-bit action. The chosen baseline gets its actual single ordinary evaluation; the original 50 retained boxes are not all exact-scored or relabeled an exhaustive oracle. This compares the evaluated shortlists, not their unknown best possible 30-action patch.

## Results

Only three rectangles, across two of 47 states: two Cat HQ step-3 tiles and one Mario64 HQ step-3 tile. All other states, including Googlewiki, return none. Six exact content evaluations total. Three binary patches have positive gain, but **zero** locally dominate, retain >=98% gain at <=75% bits, or retain >=90% gain at <=60% bits relative to the ordinary chosen patch.

| State / candidate | Gain / ordinary gain | Raw cost / ordinary cost | Gain per raw bit |
| --- | ---: | ---: | ---: |
| Cat HQ step 3 / tile (224,96), model | 10.060% | 4.467% | 281.369 |
| Cat HQ step 3 / tile (224,96), binary | 2.696% | 1.193% | 282.445 |
| Cat HQ step 3 / tile (320,128), model | 8.004% | 4.467% | 223.881 |
| Cat HQ step 3 / tile (320,128), binary | 3.475% | 1.193% | 363.997 |
| Mario64 HQ step 3 / tile (352,192), model | 1.341% | 1.241% | 126.898 |
| Mario64 HQ step 3 / tile (352,192), binary | 0.277% | 0.331% | 98.257 |

All three rectangles are 32x32. Their maximum IoU against the 50 retained stochastic proposals is 0.0878/0.0972/0.0357. Thus the exact tiles are different regions, but their absence does not imply a useful omission: they make very small total corrections per patch. Some lie on a local gain/cost frontier and have higher gain per bit; none wins the actual gain-minus-lambda*bits utility. The remaining iteration budget makes replacing one large useful patch with one tiny efficient patch a nontrivial allocation tradeoff. All evaluated local frontiers/utilities are in summary.json; no single gain-per-bit ratio is called completed-codec improvement.

Warmed median proposal time: ordinary **{smed:.3f} ms**, deterministic **{cmed:.3f} ms**. Ordinary retains 50 candidates and exact-evaluates one; deterministic returns at most three and exact-evaluates two per returned box. Deterministic time includes residual construction/statistics/components/ranking and occurs even on the 45 states returning none. These one-pass component timings are not full-codec speedups or an integrated overhead measurement. A warmup explicitly replays the first ordinary/content proposal paths outside these measurements. Initial unwarmed component evidence remains ignored and excluded. No three-repeat full-call timing was run because no practical candidate passed.

## Googlewiki and working resolution

Googlewiki source luminance has 225 eligible binary tiles of 625 complete 32x32 tiles. The 250k working luminance has 32/225; 500k has 79/484. These are different physical tile extents, not paired source-region statistics or a causal attribution of lost error. The default full preparation roundtrip has substantial loss (Pass 17 source-roundtrip RGB MSE 684.311), so an existing-resolution region selector cannot restore all source edges.

One permitted targeted diagnostic encoded **Googlewiki HQ at 500k**, with all other defaults unchanged: 214,902 final bytes, independent original RGB MSE 439.912, versus ordinary 250k HQ 176,905 / 714.432. This is about +21.48% bytes and -38.43% MSE, an operating-point tradeoff, not a matched-rate win. It is the only added resolution point, not a sweep.

At higher-resolution steps 3/8/13/18, binary residual proposals still number zero. Maximum mean-split explained variance among contrast-eligible residual tiles is approximately 0.654/0.667/0.694/0.694, below the frozen 0.8 threshold. Source/target bimodality does not imply bimodal residuals after initialization. These observations support both downsampling loss and a residual-representation mismatch; they do not establish that loosening the threshold or raising resolution would be efficient.

## Decision and next hypothesis

Do not integrate this spatial rule or force 1-bit actions. The strongest next hypothesis is a narrowly gated signed hard-edge correction that distinguishes source structure from the initialized residual, with two explicitly fitted signed levels and geometry chosen jointly with cell scale. Present evidence argues against a bare free-two-level palette replacement; any architecture needs to explain how it handles multilevel initialized residuals and stays cheap. The non-interpolated mask itself was **not** implemented or encoded in this pass. Another full-resolution dense layer would repeat Pass 17's rate/runtime risks.

A better geometry/precision policy remains possible: the isolated Cat finer-grid dominance is an example. It does not justify a global fallback, broader benchmark or production promotion. No complete-file candidate, new serializer/decoder, repeated timing, holdout, release rebuild or deployment ran. Production v0 decoding is unchanged.
''',encoding='utf-8',newline='\n')
    (HERE/'PBC_PASS18_SUMMARY.md').write_text('''# Pass 18 summary

Stopped after baseline, same-state alternatives, deterministic spatial proposals and one targeted higher-resolution diagnostic. No practical candidate passed the feasibility gate. Production remains pbc31_feature_histogram; staged ZIP/defaults/model/decoder are unchanged.

1. **Policy frequency:** 327 iterative attempts; one requested 1 bit and was rejected at zero gain. None of 323 accepted iterative patches requested 1 bit. Requested 2 bits: 88 accepted patches (27.24%). Actual effective depths: seven 1-bit (2.17%) and 93 2-bit (28.79%) patches.
2. **Serialized grid share:** iterative 1-bit 284/18,364,113 bits = 0.001546%; 2-bit 382,090/18,364,113 = 2.080634%. Including initialization: 0.001465% and 1.971467%. Raw bits only; no per-patch LZMA-byte attribution.
3. **Missed same-region actions:** one of 47 states has a locally dominating finer-grid 1-bit alternative, Cat HQ step 8. No same-cell 1-bit win. One near-gain 2-bit saving on Googlewiki HQ, plus two weaker Flower tradeoffs. No systematic fallback justified.
4. **Spatial omissions:** three deterministic tiles differ from the retained stochastic pool, but none offers comparable total gain cheaply. This does not prove existing search optimal or identify a useful missing binary region.
5. **Content proposals:** one fixed mean-split residual binary-fit/component method, at most three rectangles. Returns three rectangles in two states, six exact tests, no local dominance or high-gain-retention win.
6. **Completed-file RD:** no integrated candidate, hence no completed-file improvement established. Twelve baseline files exactly match Pass 17. The one 500k Googlewiki point is a larger-file/better-quality tradeoff.
7. **Generalization:** no new mechanism demonstrated gains across text/graphics/photos. No holdout or broad validation warranted.
8. **Photographs:** native 1-bit alternatives generally lose substantial gain on Cat/Flower. Cat has one finer-grid local win, Flower two 2-bit tradeoffs. Production paths remain unchanged; no candidate photographic regression claim is possible without candidate files.
9. **Runtime:** median warmed proposal components about 6.36 ms ordinary versus 2.78 ms deterministic; not whole-codec results. No full encode/decode overhead was measured for an integrated mechanism. Instrumented baseline walls are not performance evidence. No promising candidate existed for three-repeat timing.
10. **Format:** every tested legal patch uses unchanged v0 construction/decoding. The free-signed-level explanatory fit is unencoded. No new mask format or decoder was built.
11. **Broader validation:** not recommended for this method. Stop rather than sweep thresholds, weights or candidate counts.
12. **Next hypothesis:** jointly choose signed hard-edge representation and geometry only where initialized residuals support it; explicitly fitted levels alone are insufficient. The ordinary 1-bit palette was zero/positive in 46/47 states despite median ~50% negative residual SSE. Even a four-update free-signed-level probe retained median only 52.83% of ordinary gain. Working-resolution loss is additional, not the sole cause.
13. **Research implementation:** diagnose.py captures unchanged decisions/states; frozen.py tests 120 focused alternatives and six content actions; proposals.py implements the fixed deterministic method; representation.py diagnoses signed levels and the one 500k encode; validate.py parses all baseline bits and round-trips 173 candidate/selected patch payloads; report.py regenerates reports/summary. Compact CSV/JSON are committed, generated arrays/streams/full logits ledger/caches ignored. Two fresh subprocesses decode saved files with policy loading forbidden.
14. **Git handoff:** focused research commit and verified GitHub HEAD are supplied in the chat handoff. Only research_pass18 files are staged; no HF/protected branch/ZIP publication.

Reproduce from research checkout using Python 3.11 and research_pass18/requirements.txt (the existing Pass 17 environment works; SciPy connected components is the only SciPy algorithm used here):

```
research_pass16/.venv/Scripts/python.exe research_pass18/diagnose.py
research_pass16/.venv/Scripts/python.exe research_pass18/frozen.py
research_pass16/.venv/Scripts/python.exe research_pass18/representation.py
research_pass16/.venv/Scripts/python.exe research_pass18/report.py
research_pass16/.venv/Scripts/python.exe research_pass18/validate.py
```

Restored hpt_data assets and unchanged model are required. Runtime/source/config/input identities accompany diagnostics; raw output digests and checks accompany validation. No network jobs, training, double-encode routing, full 100-case screen, DCT retuning or release packaging is invoked.
''',encoding='utf-8',newline='\n')
    meta = provenance()
    meta['timing'] = 'Frozen component timings only; no completed-codec candidate or runtime claim.'
    meta['research_start_commit'] = summary['start_commit']
    meta['staged_zip_sha256_unchanged'] = '08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591'
    meta['measured_files'] = {p.name:sha(p.read_bytes()) for p in sorted(HERE.glob('*.json')) if p.name not in ('provenance.json','validation.json') and not p.name.endswith('_raw.json')}
    meta['raw_evidence_sha256'] = {p.name:sha(p.read_bytes()) for p in sorted(HERE.glob('*_raw.json'))}
    write_json('provenance.json', meta)
    print('reports generated; no integrated candidate')


if __name__ == '__main__':
    main()
