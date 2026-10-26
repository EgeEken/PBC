import argparse
import inspect
import sys

from support import OUT, PASS2, load, initial, run_plan, read


def worker(source):
    variant = load('pass4_variants', OUT / 'variants.py')
    sys.modules['optimizations'] = variant
    original = load('pass2_runner', PASS2 / 'runner.py')
    code = inspect.getsource(original.worker)
    needle = "'source': source, 'optimization': optimization,"
    assert code.count(needle) == 1
    code = code.replace(needle, needle + " 'experimental_stats': variant_stats(),")
    code = code.replace("if job.get('save'):", "if job.get('save') and job.get('retain_stream', True):")
    code = code.replace('result = PBC3.compress(image, config=cfg)', "result = PBC3.compress(image, config=cfg, trace=True) if job.get('trace') else PBC3.compress(image, config=cfg)")
    code = code.replace("'internal_mse': result.mse}", "'internal_mse': result.mse, 'codec_trace': result.timings if job.get('trace') else None}")
    namespace = dict(vars(original), variant_stats=variant.stats)
    exec(compile(code, str(OUT / 'worker_extension.py'), 'exec'), namespace)
    namespace['worker'](source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?')
    parser.add_argument('--worker', choices=('current', 'release', 'numpy_root'))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.mode == 'initial':
        initial()
    else:
        plan = read(args.mode + '_plan')
        run_plan(args.mode, plan['cases'], plan.get('repeats', 3))
