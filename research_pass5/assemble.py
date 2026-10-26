import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main():
    prior = ROOT / 'research_pass4'
    assert subprocess.check_output(['git', 'rev-parse', 'origin/main'], text=True).strip() == '7faa6365186d201f44b842f100645b60273ffd4c'
    for name in ('hf_candidate_grid.py', 'hf_candidate_search.py'):
        shutil.copyfile(prior / name, HERE / name)
    job = (prior / 'hf_runtime_job.py').read_text(encoding='utf-8')
    job = job.replace("MODES = ('baseline','grid_skip','search','all')", "MODES = ('baseline','all_mse','all')")
    job = job.replace("INPUTS = ('googlewiki.png','flower.jpg')", "INPUTS = ('cat.jpg','monet.jpg','flower.jpg','cathedral.png','googlewiki.png','stockpoint.jpg','mario64.png','eiffel.png')")
    job = job.replace("('images',INPUTS,2),('presets',PRESETS,2),('variants',MODES,4)", "('images',INPUTS,8),('presets',PRESETS,4),('variants',MODES,3)")
    job = job.replace("len(set(value))!=len(value) or any(v not in allowed for v in value)", "any(not isinstance(v,str) for v in value) or len(set(value))!=len(value) or any(v not in allowed for v in value)")
    job = job.replace("mode in ('grid_skip','all')", "mode in ('all_mse','all')")
    job = job.replace("mode in ('search','all')", "mode in ('all_mse','all')")
    job = job.replace("mode not in ('grid_skip','all')", "mode != 'all'")
    job = job.replace("metadata=dict(label=", "metadata=dict(worker_pid=os.getpid(),label=")
    job = job.replace("        pixels=np.asarray(decoded.image.convert('RGB'))", "        import resource\n        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n        pixels=np.asarray(decoded.image.convert('RGB'))" if __import__('os').name != 'nt' else "        peak_rss_kib=None\n        if sys.platform == 'linux':\n            import resource\n            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n        pixels=np.asarray(decoded.image.convert('RGB'))")
    job = job.replace("encode_seconds=encode,decode_seconds=decode,", "encode_seconds=encode,decode_seconds=decode,peak_process_rss_kib=peak_rss_kib,")
    job = job.replace("    return dict(metadata=metadata,rows=rows,exact=True", "    assert all(r['internal_mse'] is None if r['variant']=='all' else r['internal_mse'] is not None for r in rows)\n    return dict(metadata=metadata,rows=rows,exact=True")
    (HERE / 'hf_runtime_job.py').write_text(job, encoding='utf-8', newline='\n')
    for name in ('server.py', 'pbc3_benchmark.py'):
        source = (prior / 'integration_candidate' / name).read_text(encoding='utf-8')
        source = source.replace('research_pass4', 'research_pass5')
        if name == 'pbc3_benchmark.py':
            source = source.replace('    _STOP.clear()\n    with _LOCK:\n        if _STATE["running"]:', '    with _LOCK:\n        if _STATE["running"]:')
            source = source.replace('        _STATE.update(running=True, done=0, total=None, current={"research": True},', '        _STOP.clear()\n        _STATE.update(running=True, done=0, total=None, current={"research": True},')
            source = source.replace('raise RuntimeError("Research worker failed; see isolated stderr.log")', 'raise RuntimeError("Research worker failed: " + (output / "stderr.log").read_text()[-4000:])')
        (ROOT / name).write_text(source, encoding='utf-8', newline='\n')
    manifest = json.loads((prior / 'hf_runtime_manifest.json').read_text(encoding='utf-8'))
    manifest['source_lf'] = {name.replace('research_pass4/', 'research_pass5/'): value for name,value in manifest['source_lf'].items()}
    for name in manifest['source_lf']:
        manifest['source_lf'][name] = hashlib.sha256((ROOT / name).read_text(encoding='utf-8').encode()).hexdigest()
    tree = ast.parse(job)
    inputs = ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id == 'INPUTS'))
    manifest['inputs'] = {name: hashlib.sha256((ROOT / 'hpt_data' / name).read_bytes()).hexdigest() for name in inputs}
    (HERE / 'hf_runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files = ['server.py','pbc3_benchmark.py'] + ['research_pass5/'+name for name in ('hf_candidate_grid.py','hf_candidate_search.py','hf_runtime_job.py','hf_runtime_manifest.json')]
    (HERE / 'deployment_files.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(dict(files=files,images=inputs,variants=['baseline','all_mse','all'])))


if __name__ == '__main__':
    main()
