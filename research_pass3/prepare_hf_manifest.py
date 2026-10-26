from support import ROOT, OUT, RELEASE, images, save, sha
from hf_cpu_benchmark import module_hashes

save('hf_manifest', dict(current_modules_lf=module_hashes(ROOT), reference_modules_lf=module_hashes(RELEASE),
    candidate_lf=sha((OUT / 'integration_candidate' / 'pbc3_stream.py').read_text().encode()),
    model_sha256=sha((ROOT / 'patch_policy.npz').read_bytes()),
    inputs={p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in images()}))
