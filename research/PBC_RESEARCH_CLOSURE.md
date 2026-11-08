# PBC research closure

Research is paused after Pass 47. The current usable release is PBC V3.1. No tested V4 architecture produced a meaningful general completed-file rate-distortion improvement over canonical V3.1, and no V4 release is claimed.

PBC's main accomplishments are an algorithmic decoder that needs no learned weights; low-cost downsample initialization that works well on smooth photographic backgrounds; V3's efficient coarse procedural corrections; V3.1 semantics-preserving performance optimization; and a controlled record of codec, runtime, packaging and AVIF investigations.

Established weaknesses include V3.1's quality ceiling and weaker ordinary/high-quality RD against modern AVIF; image-dependent patch efficiency; encoder search cost; and benchmark sensitivity to provenance, preprocessing, actual completed rate and timer boundaries.

Passes 17–47 tested DCT/hybrid, sparse-detail, multiscale, predictive, mask/stroke, allocation, AV1 and selection-oracle ideas. Some narrow local or completed-file signals are preserved, but the larger approaches failed rate, generality, runtime or perceptual gates. Pass 47 did not establish that the existing policy is globally optimal; it did not find enough cross-image opportunity to justify full counterfactual encodes.

Before resuming, read the [research archive](PBC_RESEARCH_ARCHIVE.md), [index](README.md), [future handoff](PBC_FUTURE_HANDOFF.md), [canonical baseline](PBC31_CANONICAL_BASELINE.md), and [public audit](PBC_PUBLIC_MAINTENANCE_AUDIT.md). Fair comparisons need source/model/config/input hashes, completed-file bytes, source-resolution reconstruction, close measured rates, MSE plus perceptual/visual checks, and controlled encode/decode timings. Report quality, bitrate, encode and decode together.

If development resumes, begin with a distinct representation hypothesis about information current PBC syntax encodes inefficiently. Do not launch another incremental patch heuristic by default. Keep work on the research branch; changes to main, releases, Hugging Face or portfolio need separate authorization.
