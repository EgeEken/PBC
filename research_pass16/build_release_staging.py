import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREVIOUS_SHA='395e76290965c0e44fe1844b2738c3a0a193d0ce3c3090c6e274af8c10d30312'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--staging',type=Path,required=True)
    args=parser.parse_args()
    staging=args.staging.resolve()
    parent=staging.parent
    assert staging.name=='PBC3_1_Release_Staging'
    candidate=parent/'PBC3_1_Release_Staging_Pass16_Candidate'
    retained=parent/'PBC3_1_Release_Staging_Pass15_Integral_Fused'
    assert all(p.resolve().parent==parent for p in (staging,candidate,retained))
    assert not candidate.exists() and not retained.exists()
    assert sha(staging/'PBC3_1_Release.zip')==PREVIOUS_SHA
    screen=json.loads((HERE/'screen.json').read_text())
    full=json.loads((HERE/'full.json').read_text())
    edges=json.loads((HERE/'integrated_edges.json').read_text())
    assert screen['all_streams_identical'] and screen['summary']['overall']['reduction_percent']>=3
    assert all(r['reduction_percent']>0 for r in screen['summary'].values())
    assert full['all_streams_identical'] and len(full['cases'])==len(full['diagnostics'])==100
    assert edges['all_edge_streams_pixels_metrics_identical']
    assert (ROOT/'pbc3_features.py').read_text().find('def _integer_order_stats(')>=0
    shutil.copytree(staging,candidate)
    source=candidate/'GitHub_Source'
    shutil.copy2(ROOT/'pbc3_features.py',source/'pbc3_features.py')
    notes='''# PBC 3.1 performance development candidate

Current candidate: `pbc31_feature_histogram`. It adds exact histogram-based residual percentiles and quantized unique counting to the previously validated `pbc31_integral_fused` implementation. Model weights, presets, release-compatible NumPy RNG, internal MSE and the PBC3 v0 stream remain unchanged.

Against `pbc31_integral_fused`, the warmed local four-image/four-preset/three-repeat screen measured 3.88% lower full encoding time (Compression 3.57%, Balanced 3.10%, Quality 3.96%, High Quality 4.62%). All 16 case medians improved; all 48 paired measured streams were byte-identical. All 100 main/holdout image-preset pairs also matched complete streams, ordered payloads, decoded RGB, metrics, feature vectors, policy outputs and proposal/RNG diagnostics. RGB/RGBA and concurrent encoding checks passed. These are same-runtime, warmed measurements on one i9-12900H P-core, not cold-start or HF measurements.

Historical results for the preceding integral-fused candidate remain separate: its repeated local comparison against immutable PBC 3.0 measured 34.00% lower encoding time and 64.28% lower decoding time. Its incremental encoding improvement against frozen unfused PBC 3.1 was 12.00%. The earlier pre-fused HF comparison measured 28.83%/64.38% lower encode/decode time than PBC 3.0. These percentages are not combined, and the new feature candidate has not been measured on HF or in a repeated direct PBC 3.0 comparison.

The previous Pass 15 ZIP and source are retained in `PBC3_1_Release_Staging_Pass15_Integral_Fused`. This artifact is local staging; it is not a published release. No rate-distortion change is claimed. Python 3.11, NumPy 2.4.6, Numba 0.68.0 and Pillow 11.3.0 are the validated runtime. The first use may compile a Numba kernel. Cross-platform encoder bitstream identity remains unverified. See README.md and INSTALL.md for installation and API usage.
'''
    for path in (source/'RELEASE_NOTES.md',candidate/'RELEASE_NOTES.md'):
        path.write_text(notes,encoding='utf-8',newline='\n')
    readme=source/'README.md'
    text=readme.read_text()
    text=text.replace('A separate local eight-image test','A separate local eight-image test of the preceding integral-fused candidate')
    text=text.replace('The fused integral-image candidate measured','The preceding fused integral-image candidate measured')
    text='Current staged development candidate: `pbc31_feature_histogram`. Its incremental warmed local full-encode improvement over `pbc31_integral_fused` is 3.88%, with exact matched streams. See RELEASE_NOTES.md for separate historical measurements.\n\n'+text
    readme.write_text(text,encoding='utf-8',newline='\n')
    manifest=json.loads((candidate/'RELEASE_MANIFEST.json').read_text())
    manifest['development_candidate_id']='pbc31_feature_histogram'
    manifest['pass16_control_revision']=screen['baseline_commit']
    manifest['pass16_control_candidate']='pbc31_integral_fused'
    manifest['previous_zip_sha256']=PREVIOUS_SHA
    manifest['previous_staging_directory']=retained.name
    manifest['staged_source_sha256']['pbc3_features.py']=sha(ROOT/'pbc3_features.py')
    manifest['validated_feature_candidate']={'screen_summary':screen['summary'],'screen_paired_streams':48,'full_image_preset_pairs':100,'proposal_rng_feature_policy_equivalence':True,'concurrent_rgb_rgba_equivalence':True,'source_parent_commit':screen['baseline_commit'],'source_feature_sha256':sha(ROOT/'pbc3_features.py'),'isolated_release_validation_report':'research_pass16/staging_validation.json'}
    manifest['validation_of_preceding_pass15_artifact']=manifest.pop('validation')
    manifest['source_files_sha256_excluding_manifest']={str(p.relative_to(source)).replace('\\','/'):sha(p) for p in sorted(source.rglob('*')) if p.is_file() and p.name!='RELEASE_MANIFEST.json'}
    manifest_text=json.dumps(manifest,indent=2)+'\n'
    for path in (candidate/'RELEASE_MANIFEST.json',source/'RELEASE_MANIFEST.json'):
        path.write_text(manifest_text,encoding='utf-8',newline='\n')
    archive=candidate/'PBC3_1_Release.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(source.rglob('*')):
            if path.is_file(): z.write(path,path.relative_to(source).as_posix())
    lines=[sha(archive)+'  PBC3_1_Release.zip']+[sha(p)+'  GitHub_Source/'+p.relative_to(source).as_posix() for p in sorted(source.rglob('*')) if p.is_file()]
    (candidate/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='ascii',newline='\n')
    subprocess.run([sys.executable,str(ROOT/'research_pass15/validate_local_release_staging.py'),'--staging',str(candidate),'--output-json',str(HERE/'staging_validation.json'),'--reference-image',str(ROOT/'hpt_data/cat.jpg')],cwd=ROOT,check=True)
    assert staging.resolve().parent==candidate.resolve().parent==retained.resolve().parent==parent
    staging.rename(retained)
    candidate.rename(staging)
    assert sha(retained/'PBC3_1_Release.zip')==PREVIOUS_SHA
    result={'staging':str(staging),'retained':str(retained),'previous_zip_sha256':PREVIOUS_SHA,'zip_sha256':sha(staging/'PBC3_1_Release.zip'),'source_feature_sha256':sha(ROOT/'pbc3_features.py'),'validated':True}
    (HERE/'staging.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
