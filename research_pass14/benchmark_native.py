import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

THREADS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
os.environ.update(THREADS)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numba
import psutil
from PIL import Image

numba.set_num_threads(1)
cpu = min(psutil.Process().cpu_affinity())
psutil.Process().cpu_affinity([cpu])
from PBC3 import PBC3

SAMPLES = 5


def timing(values):
    return dict(n=len(values), mean_seconds=statistics.mean(values), median_seconds=statistics.median(values), cv_percent=100 * statistics.pstdev(values) / statistics.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-bench", required=True)
    args = parser.parse_args()
    fixtures = json.loads((HERE / "native_cases" / "fixtures.json").read_text(encoding="utf-8"))
    cases = [r for r in fixtures if r["name"].rsplit("_", 1)[-1] in ("quality", "high_quality") and r["name"].split("_", 1)[0] in ("cat", "googlewiki", "monet", "mario64", "eiffel", "seagull")]
    cases.sort(key=lambda r: (r["name"].split("_", 1)[0], r["name"].rsplit("_", 1)[-1]))
    paths = [str(HERE / "native_cases" / r["file"]) for r in cases]
    proc = subprocess.Popen([args.rust_bench, *paths], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        psutil.Process(proc.pid).cpu_affinity([cpu])
        ready = proc.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"Rust warmup failed: {ready} {proc.stderr.read()[-1000:]}")
        data = [Path(path).read_bytes() for path in paths]
        for stream in data:
            PBC3.decompress(stream)
        python_times = {i: [] for i in range(len(cases))}
        rust_times = {i: [] for i in range(len(cases))}
        rust_internal = {i: [] for i in range(len(cases))}
        for repeat in range(SAMPLES):
            for i, stream in enumerate(data):
                order = ("python", "rust") if (repeat + i) % 2 == 0 else ("rust", "python")
                for variant in order:
                    if variant == "python":
                        start = time.perf_counter()
                        PBC3.decompress(stream)
                        python_times[i].append(time.perf_counter() - start)
                    else:
                        proc.stdin.write(f"{i}\n")
                        proc.stdin.flush()
                        fields = proc.stdout.readline().strip().split("\t")
                        if len(fields) != 4:
                            raise RuntimeError(f"unexpected Rust timing: {fields!r}")
                        rust_internal[i].append(float(fields[0]))
                        rust_times[i].append(float(fields[1]))
        rows = []
        for i, case in enumerate(cases):
            py = timing(python_times[i])
            rust = timing(rust_times[i])
            rows.append(dict(name=case["name"], file_bytes=case["file_bytes"], patch_count=case["patch_count"], python=py, rust_function=rust, rust_internal=timing(rust_internal[i]), python_over_rust_median_ratio=py["median_seconds"] / rust["median_seconds"], rust_faster_percent=100 * (1 - rust["median_seconds"] / py["median_seconds"])))
        result = dict(schema="pbc-pass14-rust-decoder-bench-v1", python=sys.version, platform=sys.platform, numpy=__import__("numpy").__version__, numba=numba.__version__, pillow=Image.__version__, rust="1.99.1 x86_64-pc-windows-gnu", cpu_affinity=cpu, thread_variables=THREADS, warmups_per_case=1, paired_repeats=SAMPLES, pair_order="alternated by case and repeat", timer="same-process direct PBC3::decompress and PBC3.decompress; streams loaded before timing; Rust runner startup and input reads excluded; Rust timing includes any LZMA fallback", rows=rows, aggregate=dict(python_mean_case_median_seconds=statistics.mean(r["python"]["median_seconds"] for r in rows), rust_mean_case_median_seconds=statistics.mean(r["rust_function"]["median_seconds"] for r in rows), median_case_speed_change_percent=statistics.median(r["rust_faster_percent"] for r in rows), cases_rust_faster=sum(r["rust_faster_percent"] > 0 for r in rows), cases=len(rows)))
        (HERE / "rust_timing_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["aggregate"], indent=2))
    finally:
        if proc.stdin:
            proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
