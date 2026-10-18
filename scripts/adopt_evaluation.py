"""Adopt existing evaluation data in place. Does not open either database."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pygmalion_hoard.hoard_link.atomic import write_text_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--evaluation-data',type=Path,required=True)
    args=parser.parse_args()
    evaluation=args.evaluation_data.resolve()
    if not evaluation.is_dir():parser.error('The existing evaluation data directory does not exist.')
    data=args.data_dir.resolve();data.mkdir(parents=True,exist_ok=True)
    target=data/'modules.json'
    saved=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
    paths=saved.setdefault('data_paths',{})
    paths.setdefault('training',str(data));paths['evaluation']=str(evaluation)
    saved['enabled']=[]
    write_text_atomic(target,json.dumps(saved,indent=2,ensure_ascii=False)+'\n')
    print('Existing data mapped in place; all modules OFF. No database was opened.')
    return 0

if __name__=='__main__':raise SystemExit(main())
