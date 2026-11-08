# PBC research handoff — resume after Pass 47

## Status

Research is paused. The usable released codec is PBC V3.1. No validated V4 prototype improves canonical V3.1 generally. The `research/experiments` branch is an archive, not a release branch. Start with [the archive](PBC_RESEARCH_ARCHIVE.md), [Pass 45 baseline](PBC31_CANONICAL_BASELINE.md), [machine index](RESEARCH_INDEX.json), and [closure note](PBC_RESEARCH_CLOSURE.md).

## Codec principles

PBC is a lossy procedural residual codec. The encoder normalizes an image, optionally plans a working resolution, builds an initialization, then selects spatial residual patches and algorithmic actions. Those action parameters deterministically generate correction samples. The learned policy is encoder-side only; the PBC3 v0 decoder parses and reconstructs without model weights. LZMA2 is part of the two-stage rate, so only final complete-file bytes measure compression.

V2.3 introduced downsample initialization, giving smooth/blurred backgrounds a low-cost base. V3 combined it with coarse procedural residual patches. Both contribute to V3.1's low-rate behavior; Pass 43 measured that V4-A's larger palette initialization was already much more expensive before native refinements. Do not attribute all low-rate strength to downsampling alone.

## Canonical identities

Use the published PBC 3.1 release ZIP as the compatibility reference, as Pass 45 did; the tag tree is not a substitute for the shipped artifact.

- ZIP SHA-256: `3ee8f68b45adb3a89128251b8905d93cb99a96d911a055f91e124e724abbe069`
- `patch_policy.npz` SHA-256: `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`
- Normalized preset JSON SHA-256: `5a2eaab04bbfe373ec0d8e4e1f059b2003b5d90f31d12178382550682780338d`
- The manifest remains **provisional**: Python source probes passed, but the packaged executable and exact HF Linux native parity were not verified.
- Encoder details: seed 2003; fresh NumPy generator per encode; YCbCr; Pillow bicubic with configured reducing gap; PBC3 v0; STORE/raw LZMA2 choice. Four exact dictionaries are in `research_pass10/baseline_manifest.json` and the canonical JSON.
- Source/data provenance: `research_pass45/INPUT_PROVENANCE.md`, `research_pass45/PASS45_SUMMARY.json`, `research/PBC31_CANONICAL_BASELINE.json`, and each pass manifest.

## Fair reproduction procedure

1. Verify source/model/input hashes and canonical config identity. Normalize with EXIF transpose then RGB (or preserve RGBA when required by the tested path).
2. Compare actual output files. Report exact bytes and bpp over original pixel count. Preset names are not matched rates; use close measured pairs or brackets and state the rate gap.
3. Decode the completed stream independently where possible. Compute source-resolution RGB MSE with promoted arithmetic. Record PSNR and validated perceptual metrics (e.g. SSIMULACRA2) when available; inspect source-aligned crops for text, thin edges, gradients and artifacts. MSE alone is not perceptual ground truth.
4. Time complete encode/decode boundaries. Record CPU, thread limits, warmup, repetitions, process start and I/O inclusion. Do not directly rank AVIF CLI times against in-process PBC calls with different boundaries.
5. Keep photos, synthetic graphics, text and resolution classes distinct. Tune only on development images; preserve holdouts for confirmation. Save raw rows, hashes, commands, configs and compact results.
6. Treat quality, bitrate, encode time and decode time as four practical axes. An MSE reduction at higher rate is a tradeoff. A speedup over a very slow unsuccessful prototype is not competitive evidence.

Identify AVIF encoder/decoder implementation and settings. HF-compatible AVIF and pinned research libavif/libaom produce different quality-0 points; quality 0 is not a fixed bitrate.

## Prior work and what not to repeat blindly

The [archive](PBC_RESEARCH_ARCHIVE.md) gives hypotheses, methods, outcomes, limitations and links through Pass 47. Tested directions include PBC search/allocation/palette changes; exact V3.1 runtime work; dense/sparse DCT, fixed-grid prediction, multiscale/hybrid bases; masks/strokes; decoder-aware target change; patch omission; AV1 mode/AQ/chroma changes; V4-A/B; and a native-action oracle.

Do not repeat arbitrary proposal ranking, another small patch-pruning rule, low-bit palette sweeps, generic DCT add-ons, fixed-base sidecars or minor parameter tuning without a new information-theoretic hypothesis. Pass 47 failed its cross-image local gate but did not prove global policy optimality.

A credible future direction must explain which information current syntax cannot encode efficiently, why a decoder-simple representation can lower complete-file cost, and how the encoder can select it without excessive search. Begin with a bounded representation analysis. No V4 architecture has been selected.

## Known gaps

The historical HF screenshot discrepancy remains unresolved. Some raw runs, generated files, build trees, binaries and environments are excluded from Git; see the archive artifact manifest and each pass reproduction guide. A number of commands require external benchmark data or pinned upstream source. Pass 47 produced no alternative complete-file continuation. Do not overstate reproducibility.
