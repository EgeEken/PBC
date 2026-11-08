# Input provenance

Research Coast input: `hpt_data/coast.jpg`; SHA-256 `7cb3bd541613e2f048a49add80d6e83109686b6744f490d894938ac887765ea2`; JPEG, 4240×2832, EXIF orientation 1, no ICC profile recorded, converted with Pillow `ImageOps.exif_transpose(...).convert("RGB")`. RGB pixel SHA-256: `29af11f09492532ea4d62bde434cacf99f7fe79d5952f8c5bbefd4a8a53d6887`.

This is the exact Pass 44 benchmark input. The uploaded file from the user’s HF session is unavailable, so its byte hash and decoded-pixel hash cannot be compared. The user-observed AVIF size/MSE pair is reproduced by the local Pillow AVIF path on this research asset, which is strong corroborating evidence but is not a pixel-hash proof of uploaded-input identity. No source image was uploaded to an external service.

Other available fixed-corpus assets and hashes are recorded in `research_pass10/baseline_manifest.json`; this pass did not encode them across implementations.
