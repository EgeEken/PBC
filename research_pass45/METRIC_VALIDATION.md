# Metric validation

Canonical RGB MSE used here is `mean((int32(source_rgb) - int32(decoded_rgb))**2)` over the full original-resolution H×W×3 array. The research input is orientation-transposed and converted to RGB before encoding. Decodes are converted to RGB. No alpha was present in Coast.

For Coast Compression, the independent MSE from the stored Pass 42 decoded PNG is **73.13358281255552**, matching the Pass 44 benchmark’s PBC internal MSE exactly. The decoded RGB hash is `0c0527f2fb9f95694509acd5fe91b1440126c80b068bd1476a77970e7150f655`. The same complete stream hash is `792e9602ba3c4a407c143c1e1e2fdb6e15d35ae1ca8ad2446303770a0833a029`.

The demo’s reported values are rounded to two decimal places. This pass did not obtain the PBC stream returned by the user’s demo session, so its displayed MSE could not be independently checked. Pass 9 public API checks established preview/decode and reported-MSE agreement for the baseline corpus, not this specific incident.

For RGBA, benchmark RGB MSE and alpha error separately; do not include alpha in RGB MSE. Alpha is preserved by the public API, but no RGBA incident fixture was run in Pass 45.
