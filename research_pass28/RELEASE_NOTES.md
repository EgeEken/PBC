# PBC 3.2.0

Standalone Windows x64 applications are available as a combined encoder/decoder
and a smaller decoder-only download. Neither requires a separate Python or
compiler installation. The decoder does not include Numba, LLVM or the policy
model. Keep the executable and _internal directory together after extraction.

Encoding retains PBC 3.1's presets, reconstruction and PBC3 v0 streams. The
Windows application selects an optional native C final-MSE backend per encode;
reference and native backends produce identical streams. The portable Python
wheel uses the reference backend when no DLL is installed.

The Numba cache now lives outside the application directory and is reused across
working directories. First-ever encoding still compiles kernels. Python 3.11
wheel/source installation includes the model in the correct package location.

Acceptance: Windows Server 2022 and 2025 hosted runners, local non-admin and
read-only application directories, historical 3.0/3.1 streams and malformed-input
checks. Hosted runners contain development tools; a clean consumer Windows
installation has not been tested. A Python-free consumer smoke-test bundle is
provided. Executables are unsigned. No installer or automatic updater is needed.

In the six-image, four-preset acceptance set, native encoding reduced mean case
median time by 6.96% against the candidate reference backend (5.97% against
published 3.1). Individual cases vary; no general decoder speedup is claimed.
Cat/Flower native peak working sets were about 147 MiB lower in repeated isolated
measurements. These are local single-core results, not HF or universal timings.

Decode limits: 32 million pixels, 64 MiB compressed input, 128 MiB expanded body,
10,000 patches. Valid files exceeding those limits are rejected. Dependencies
retain their licenses; keep THIRD_PARTY_NOTICES.md and licenses with downloads.
