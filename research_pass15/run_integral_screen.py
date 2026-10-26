import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKER = HERE / "benchmark_worker.py"
IMAGES = ("cat.jpg", "eiffel.png", "monet.jpg", "seagull.jpg")
PRESETS = ("compression", "balanced", "quality", "high_quality")
THREADS = {key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)}
BASELINE_COMMIT = "8006f30debfc4a57289df06df21636aeb52baffd"


def send(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def start(name, source, python, images, cpu, candidate):
    env = os.environ.copy()
    env.update(THREADS)
    env.update({
        "PBC_CPU": str(cpu), "PBC_IMAGES": json.dumps([str(path) for path in images]),
        "PBC_SOURCE": str(source), "PYTHONPATH": str(source), "PYTHONUNBUFFERED": "1",
        "PBC_INTEGRAL_CANDIDATE": "1" if candidate else "0",
    })
    process = subprocess.Popen(
        [str(python), str(WORKER)], cwd=source, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    process._name = name
    return process


def stop(process):
    if process.poll() is None:
        process.stdin.write('{"op":"stop"}\n')
        process.stdin.flush()
        process.stdin.close()
        process.wait(timeout=30)
    if process.returncode:
        raise RuntimeError(f"{process._name} exited {process.returncode}: {process.stderr.read()[-3000:]}")


def check_integral(python):
    script = r'''import json, statistics, sys, time, numpy as np
sys.path.insert(0, __HERE__)
from integral_candidate import integral as candidate, _integral_kernel
def baseline(a):
    return np.pad(a.astype(np.int64).cumsum(0).cumsum(1), ((1,0),(1,0)))
rng=np.random.default_rng(8128)
arrays=[]
for dtype in (np.uint8,np.int16,np.int64,np.float32,np.float64):
    for h,w in ((0,0),(0,7),(9,0),(1,1),(17,23),(64,71)):
        a=rng.integers(-10000,10001,size=(h,w),dtype=np.int64).astype(dtype)
        arrays.extend((a,np.asfortranarray(a),a[:,::2]))
arrays += [np.array([[np.iinfo(np.int64).max,1],[-1,np.iinfo(np.int64).min]],dtype=np.int64),
           np.array([[2**63,2**64-1]],dtype=np.uint64),
           np.array([[True,False]],dtype=bool),
           np.array([[1.9,-1.9],[0.5,-0.5]],dtype=np.float64),
           np.array([[np.nan,np.inf,-np.inf]],dtype=np.float64),
           np.empty((0,0),dtype=np.float64)]
for i,a in enumerate(arrays):
    x,y=baseline(a),candidate(a)
    assert x.shape==y.shape and x.dtype==y.dtype==np.int64 and np.array_equal(x,y), (i,a.dtype,a.shape)
for bad in (np.ones(3),np.ones((2,2,2))):
    try: candidate(bad)
    except ValueError: pass
    else: raise AssertionError('candidate accepted non-2D input')
sizes=((128,128),(512,512),(1024,1024),(2048,2048))
bench={}
for h,w in sizes:
    a=rng.integers(0,256,size=(h,w),dtype=np.uint8)
    baseline(a); candidate(a)
    result={}
    for name,fn in [('numpy',baseline),('numba_fused',candidate)]:
        samples=[]
        for _ in range(7):
            t=time.perf_counter(); fn(a); samples.append(time.perf_counter()-t)
        result[name]=statistics.median(samples)
    bench[f'{h}x{w}']=result
print(json.dumps({'arrays_checked':len(arrays),'benchmarks':bench}))
'''.replace("__HERE__", repr(str(HERE)))
    return json.loads(subprocess.check_output([str(python), "-c", script], cwd=ROOT, env={**os.environ, **THREADS}, text=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--output", type=Path, default=HERE / "integral_screen.json")
    parser.add_argument("--full-correctness", action="store_true")
    args = parser.parse_args()
    if args.full_correctness:
        image_names = sorted(path.name for dataset in (args.data_dir, args.data_dir.parent / "hpt_data_val") for path in dataset.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        repetitions = 1
    else:
        image_names = list(IMAGES)
        repetitions = 3
    images = [args.data_dir / name if (args.data_dir / name).is_file() else args.data_dir.parent / "hpt_data_val" / name for name in image_names]
    assert all(path.is_file() for path in images)
    check = check_integral(args.python)
    baseline_temp = tempfile.TemporaryDirectory(prefix="pbc31-integral-baseline-")
    baseline_source = Path(baseline_temp.name) / "source"
    baseline_source.mkdir()
    archive = subprocess.check_output(["git", "archive", "--format=tar", BASELINE_COMMIT], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(baseline_source)
    workers = {
        "baseline": start("baseline", baseline_source, args.python, images, args.cpu, False),
        "candidate": start("candidate", ROOT, args.python, images, args.cpu, True),
    }
    rows = []
    try:
        for name, process in workers.items():
            assert send(process, {"op": "warm"}).get("ready"), name
        for rep in range(repetitions):
            for image_index, image in enumerate(image_names):
                for preset in PRESETS:
                    order = ("baseline", "candidate") if rep % 2 == 0 else ("candidate", "baseline")
                    for variant in order:
                        row = send(workers[variant], {"op": "measure", "image_index": image_index, "preset": preset})
                        row.update({"variant": variant, "rep": rep})
                        rows.append(row)
                print(f"completed repetition {rep + 1}/{repetitions} for {image} ({len(rows)}/{len(image_names) * len(PRESETS) * repetitions * 2} variant observations)", flush=True)
    finally:
        for process in workers.values():
            stop(process)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["image"], row["preset"], row["rep"]), {})[row["variant"]] = row
    keys = sorted(grouped)
    comparisons = []
    for key in keys:
        a, b = grouped[key]["baseline"], grouped[key]["candidate"]
        comparisons.append({"image": key[0], "preset": key[1], "rep": key[2],
            "identical_stream": a["stream_sha256"] == b["stream_sha256"],
            "identical_pixels": a["decoded_rgb_sha256"] == b["decoded_rgb_sha256"],
            "identical_mse": a["rgb_mse"] == b["rgb_mse"],
            "baseline_encode_seconds": a["encode_seconds"], "candidate_encode_seconds": b["encode_seconds"],
            "baseline_decode_seconds": a["decode_seconds"], "candidate_decode_seconds": b["decode_seconds"],
        })
    assert len(rows) == len(image_names) * len(PRESETS) * repetitions * 2
    assert len(keys) == len(image_names) * len(PRESETS) * repetitions
    assert all(row["identical_stream"] and row["identical_pixels"] and row["identical_mse"] for row in comparisons), "candidate changed codec output"
    summary = {}
    for preset in PRESETS:
        entries = [r for r in comparisons if r["preset"] == preset]
        old = statistics.mean([statistics.median([r["baseline_encode_seconds"] for r in entries if r["image"] == image]) for image in image_names])
        new = statistics.mean([statistics.median([r["candidate_encode_seconds"] for r in entries if r["image"] == image]) for image in image_names])
        summary[preset] = {"baseline_encode_seconds": old, "candidate_encode_seconds": new, "encode_reduction_percent": 100 * (1 - new / old)}
    old = statistics.mean([r["baseline_encode_seconds"] for r in comparisons])
    new = statistics.mean([r["candidate_encode_seconds"] for r in comparisons])
    summary["overall"] = {"baseline_encode_seconds": old, "candidate_encode_seconds": new, "encode_reduction_percent": 100 * (1 - new / old)}
    result = {"status": "passed" if summary["overall"]["encode_reduction_percent"] > 0 else "rejected_no_encode_gain",
        "baseline_commit": BASELINE_COMMIT, "images": image_names, "presets": list(PRESETS),
        "repetitions": repetitions, "paired_case_repetitions": len(keys), "variant_observations": len(rows),
        "all_outputs_byte_identical": True, "correctness": check, "summary": summary,
        "pair_observations": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "correctness", "summary")}, indent=2))


if __name__ == "__main__":
    main()
