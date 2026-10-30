# Publication procedure — requires separate user approval

Nothing in Pass 28 creates a release tag, changes main, uploads release assets or
deploys HF. Final 3.2.0 files are local. RC1 is preserved under artifacts/dist,
artifacts/release-source and artifacts/accepted-rc1.

1. Review PBC32_RELEASE_ACCEPTANCE.md, PBC32_RELEASE_READINESS.md and the exact
   paths/hashes in PBC32_DISTRIBUTION_MANIFEST.json. Run the supplied consumer
   smoke bundle on a fresh Windows installation if that coverage is required.
2. Use PBC-3.2.0-release-source.zip as the complete release source. Review its
   file-level diff against main in an isolated release checkout. Do not merge
   research/experiments or tag its tip as the release source. The source tree
   contains the runtime, unchanged policy model, notices, package metadata and
   self-contained Windows build tools; it excludes research, models in training,
   caches, DLLs, executables and datasets.
3. After explicit approval for the release-source commit/target branch, construct
   that commit from exactly this snapshot. Its root tree must equal the
   git_tree_sha1 recorded in SOURCE_TREE.json. Its parent/history may differ;
   the complete source tree must match. Review any removals from main before
   applying them. Do not silently mix old root packaging into the snapshot.
4. Rebuild with BUILD.md, verify IMPLEMENTATION.json, and run the acceptance and
   installed-source checks against that release checkout. If source or dependency
   inputs change, regenerate manifests and rerun affected validation. Executable
   hashes may differ across paths/build timestamps; source/runtime payload and
   output checks remain mandatory. Do not rename an RC binary to make it stable.
5. With explicit tag/release approval, tag the exact verified source commit
   v3.2.0. Confirm its root tree matches SOURCE_TREE.json, then publish the
   reviewed artifacts and SHA256SUMS.txt with RELEASE_NOTES.md. GitHub's automatic
   source archive must expose the same source tree as the attached source ZIP.
   Preserve published 3.1 assets/tags. No HF deployment is part of this procedure.
6. Recommend combined and decoder directory ZIPs; optional one-file-with-notices
   ZIP; Python wheel/sdist; exact release-source ZIP; consumer smoke bundle and
   checksum/source manifests. A raw one-file EXE is also staged locally, but its
   companion notices should remain readily available. Add no graphical installer
   or updater without a demonstrated need. Signing remains a separate process;
   do not describe these unsigned files as signed or SmartScreen-approved.
