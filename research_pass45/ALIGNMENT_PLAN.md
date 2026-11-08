# Alignment plan

1. Preserve the public ZIP and HF deployment unchanged. Use the ZIP’s embedded Python as the provisional compatibility oracle.
2. Run Coast with all four presets from the release Python, frozen research, and current HF snapshot in a pinned Python 3.11 environment; require expected stream and decoded-pixel hashes. Test the bundled executable independently.
3. Obtain the exact HF upload file or its decoded RGB hash plus response `.pbc` bytes and request parameters. Without these, the triggering PBC discrepancy cannot be assigned to codec behavior.
4. Compare release and HF source history to identify the ZIP’s source revision; record linked native DLL versions and AVIF runtime build.
5. Add the golden preset/hash check to routine benchmark setup. Any implementation change must update the canonical manifest intentionally and review PBC3 v0 compatibility.
6. Only after that evidence, prepare product alignment patches. No patches to HF, the desktop app, or production sources were justified or staged in Pass 45.
