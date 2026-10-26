import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('control','diagnose','S1','S2'), required=True)
    args=parser.parse_args()
    from research_pass6 import worker
    original = worker.main
    if args.mode != 'control':
        import stop_policy
        sys.modules['refinement'] = stop_policy
        # Reuse the full-call timer, independent decoder and stream/trajectory reader.
        worker_source = 'init'
        context = stop_policy.enabled
        stop_policy.enabled = lambda ignored, stats: context(args.mode, stats)
    else:
        worker_source = 'integrated'
    sys.argv=[sys.argv[0],'--source',worker_source]
    original()


if __name__=='__main__':
    main()
