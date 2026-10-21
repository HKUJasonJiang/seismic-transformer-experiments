"""Archive diagnostic artifacts to the configured private dataset repository."""
import sys
sys.dont_write_bytecode=True
import argparse
import json
import os
from pathlib import Path
from huggingface_hub import HfApi


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='*',default=[])
    p.add_argument('--reports',nargs='+',default=['diagnosis-basis-check','diagnosis-nonlinear-summary']);args=p.parse_args()
    stage=Path(__file__).resolve().parents[1];repo=os.environ['HF_REPO_ID'];api=HfApi()
    if not api.repo_info(repo,repo_type='dataset').private:raise RuntimeError('Diagnostics must remain private')
    receipts=[]
    for name in args.runs:
        if not name.startswith('diagnosis-') or '/' in name or '\\' in name:raise ValueError('Invalid run name')
        for kind,root in (('artifacts','runs'),('reports','reports')):
            result=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=stage/root/name,path_in_repo=f'{kind}/{name}',
                                     allow_patterns=['*.pt','*.json','*.npz'] if kind=='artifacts' else None,
                                     commit_message='Archive nonlinear training diagnostic '+name)
            receipt={'kind':kind,'run':name,'commit':result.oid};receipts.append(receipt);print(json.dumps(receipt),flush=True)
    for name in args.reports:
        if not name.startswith('diagnosis-') or '/' in name or '\\' in name:raise ValueError('Invalid report name')
        result=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=stage/'reports'/name,path_in_repo=f'reports/{name}',commit_message='Archive nonlinear diagnostic conclusion')
        receipt={'kind':'reports','run':name,'commit':result.oid};receipts.append(receipt);print(json.dumps(receipt),flush=True)
    receipt_path=stage/'reports'/'diagnosis-nonlinear-summary'/'archive_receipt.json'
    previous=json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    receipt_path.write_text(json.dumps(previous+receipts,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
