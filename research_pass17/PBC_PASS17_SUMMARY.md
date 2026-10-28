# PBC Pass 17 summary

The full-image signed DCT residual layer demonstrates useful completed-file RD gains on four of six images. It uses a reduced Quality p10 base, a fixed color transform, 8x8 DCT, three predeclared quantizers and a separately decodable P17D container. Production remains pbc31_feature_histogram; this is an R&D prototype, not a release candidate.

Against measured ordinary HQ, selected hybrid points reduce both bytes and RGB MSE: Monet (q64), Googlewiki (q128), Mario64 (q128), Eiffel (q32). Three actual controls within 5% rate separately show large MSE gains, with signed rate differences retained. Googlewiki has strong measured dominance but no 5%-near ordinary point. Sparse conservative bracket estimates are reported separately; there is no extrapolation or universal frontier claim. Measured hybrid quality brackets support matched-quality interpolation only for Monet and Eiffel; those estimates remain separate from actual files.

Cat and Flower remain too expensive at full-image layer rates. Independent tile coding at fixed total bytes worsened 10/12 points; its HQ Googlewiki result is dominated by the stronger full-image point. No alternative architecture or rescue sweep was needed because the primary architecture already warrants broader research.

All 30 completed experimental files pass actual decoding, SHA/size/MSE checks, signed coefficient readback, boundary/determinism checks and fresh-process file-only decoding with policy loading forbidden. The ordinary v0 decoder is unchanged and rejects P17D. Full mode adds a 30-byte header plus its actual entropy-coded coefficients; tile mode also adds 8 bytes per selected tile, with XZ framing counted inside payload bytes.

The RD gain costs compute: repeated Googlewiki hybrid/ordinary HQ encode is about 4.0x, decode 1.56x; Eiffel q64/ordinary Quality encode is 1.76x, decode 1.53x. Complete calls include all base/detail work and internal base MSE. Runtime is Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0 and SciPy 1.17.1, one i9-12900H P-core, one numerical thread.

The frozen research prototype is pbc17_dct_detail_v1. Recommend a user-approved 8 to 10-image independent follow-up with actual close-rate ordinary controls, including a few larger-working-resolution controls. Do not automatically run the 100-case matrix. Joint base/detail allocation and multiscale detail coding for large photographs are the strongest next hypothesis, based on their rate failure and distributed low/mid-frequency residuals.

No production source/model/default, HF deployment, protected branch, release tag or staging artifact changed. Staged ZIP SHA-256 remains 08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591. Reports and compact per-file rows are in this directory; generated files are ignored. The final research commit is supplied in the chat handoff.
