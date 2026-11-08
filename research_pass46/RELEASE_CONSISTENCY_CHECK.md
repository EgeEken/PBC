# Pass 45 release checks continued in Pass 46

## Canonical V3.1 reference — verified

The canonical published-ZIP Python source, model and effective presets are pinned by `research/PBC31_CANONICAL_BASELINE.json`. Pass 45 reproduced Coast Compression exactly against frozen research (6,461 bytes, stream SHA-256 `792e9602ba3c4a407c143c1e1e2fdb6e15d35ae1ca8ad2446303770a0833a029`, decoded RGB SHA-256 `0c0527f2fb9f95694509acd5fe91b1440126c80b068bd1476a77970e7150f655`, MSE 73.1335828). It also matched all four presets for Coast/Cat/Googlewiki and eight archived Pass 26 files. Pass 46 uses that canonical source-level Python reference.

## HF direct check — not verified

The public Space page was reachable, but its status was `Sleeping`; no live encode response could be obtained. It was not restarted. No benchmark image was uploaded.

## Windows executable — not verified

The published 76.9 MB ZIP and its 8,362,749-byte `PBC3_1_Release/PBC.exe` were available, and the archive path manifest was checked before extracting the package to a unique temporary directory. Desktop-app launch approval timed out. The executable was not run; no output-equivalence claim is made from its embedded Python source.

No deployment or release files were changed.
