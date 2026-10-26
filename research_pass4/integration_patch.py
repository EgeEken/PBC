import difflib
import json

from support import ROOT, OUT, PASS3, save, sha, snapshot


def main():
    before=snapshot()
    for source,target in ((PASS3/'integration_candidate'/'pbc3_stream.py',OUT/'hf_candidate_grid.py'),
                          (PASS3/'search_reuse.py',OUT/'hf_candidate_search.py')):
        target.write_bytes(source.read_bytes())
    server=(ROOT/'server.py').read_text(encoding='utf-8')
    needle='    body = await request.json()\n    return _guarded_benchmark_start(\n        "benchmark",'
    replacement='''    body = await request.json()
    if "research" in body:
        import hmac
        expected = os.environ.get("PBC_RESEARCH_BENCHMARK_TOKEN", "")
        if not expected:
            raise HTTPException(status_code=404, detail="Research benchmark disabled")
        if not hmac.compare_digest(request.headers.get("x-pbc-benchmark-token", ""), expected):
            raise HTTPException(status_code=403, detail="Research benchmark authorization required")
        try:
            from research_pass4.hf_runtime_job import validate
            spec = validate(body["research"])
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=400, detail=str(error))
        return _guarded_benchmark_start("benchmark", pbc3_benchmark.start_research,
            pbc3_benchmark.status, spec)
    return _guarded_benchmark_start(
        "benchmark",'''
    assert server.count(needle)==1
    proposed_server=server.replace(needle,replacement)
    backend=(ROOT/'pbc3_benchmark.py').read_text(encoding='utf-8')
    addition='''

def start_research(spec):
    global _THREAD
    from research_pass4.hf_runtime_job import validate
    spec = validate(spec)
    _STOP.clear()
    with _LOCK:
        if _STATE["running"]:
            return {"error": "A benchmark is already running."}
        _STATE.update(running=True, done=0, total=None, current={"research": True},
            started=time.time(), error=None, log=[], research_result=None)

    def run():
        import json
        import subprocess
        import sys
        import uuid
        from pathlib import Path
        output = Path(PROJECT_DIR) / "research_pass4" / "hf_runs" / uuid.uuid4().hex
        try:
            output.mkdir(parents=True, exist_ok=False)
            with (output / "stderr.log").open("w") as log:
                process = subprocess.Popen([sys.executable, str(Path(PROJECT_DIR) / "research_pass4" / "hf_runtime_job.py"),
                    "--spec-json", json.dumps(spec)], cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                    stderr=log, text=True)
                deadline = time.monotonic() + 1200
                while True:
                    try:
                        stdout, _ = process.communicate(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if _STOP.is_set() or time.monotonic() > deadline:
                            process.terminate()
                            try:
                                process.communicate(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.communicate()
                            raise RuntimeError("Research benchmark stopped or timed out")
                if process.returncode:
                    raise RuntimeError("Research worker failed; see isolated stderr.log")
            result = json.loads(stdout)
            (output / "results.json").write_text(json.dumps(result, indent=2))
            with _LOCK:
                _STATE.update(research_result=result, done=len(result["rows"]), total=len(result["rows"]))
        except Exception as error:
            with _LOCK:
                _STATE["error"] = str(error)
        finally:
            with _LOCK:
                _STATE.update(running=False, current=None)
    _THREAD = threading.Thread(target=run, daemon=True)
    _THREAD.start()
    return {"started": True, "research": True, "spec": spec}
'''
    proposed_backend=backend+addition
    patch=''.join(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        for name,a,b in (('server.py',server,proposed_server),('pbc3_benchmark.py',backend,proposed_backend)))
    (OUT/'proposed_hf_benchmark.patch').write_text(patch,encoding='utf-8')
    candidate=OUT/'integration_candidate'
    candidate.mkdir(exist_ok=True)
    (candidate/'server.py').write_text(proposed_server,encoding='utf-8')
    (candidate/'pbc3_benchmark.py').write_text(proposed_backend,encoding='utf-8')
    names=('PBC3.py','pbc3_types.py','pbc3_ops.py','pbc3_heads.py','pbc3_features.py','learned_filler.py','pbc3_stream.py','pbc3_trace.py')
    extra=('research_pass4/hf_candidate_grid.py','research_pass4/hf_candidate_search.py','research_pass4/hf_runtime_job.py')
    manifest=dict(source_lf={n:sha((ROOT/n).read_text(encoding='utf-8').encode()) for n in names+extra},
        model_sha256=sha((ROOT/'patch_policy.npz').read_bytes()),
        inputs={n:sha((ROOT/'hpt_data'/n).read_bytes()) for n in ('googlewiki.png','flower.jpg')})
    save('hf_runtime_manifest',manifest)
    save('integration_patch_provenance',dict(before=before,after=snapshot(),applied=False,
        production_paths=['server.py','pbc3_benchmark.py'],stage_files=list(extra)+['research_pass4/hf_runtime_manifest.json']))


if __name__=='__main__':
    main()
