import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('control','robust'),required=True)
    mode=parser.parse_args().mode
    from research_pass6 import worker
    if mode=='robust':
        import palette_scope
        sys.modules['refinement']=palette_scope
    sys.argv=[sys.argv[0],'--source','integrated' if mode=='control' else 'init']
    worker.main()


if __name__=='__main__':
    main()
