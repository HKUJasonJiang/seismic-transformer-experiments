"""HF authentication through stdin and dataset/artifact synchronization on 5090."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True

from huggingface_hub import HfApi,login

STAGE=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--login-stdin",action="store_true")
    p.add_argument("--data",action="store_true")
    p.add_argument("--data-subdir",type=str)
    p.add_argument("--run",type=str)
    p.add_argument("--summary",action="store_true")
    args=p.parse_args()
    repo=os.environ["HF_REPO_ID"]
    if args.login_stdin:
        secret=sys.stdin.readline().strip()
        if not secret:raise RuntimeError("No credential received on stdin")
        login(token=secret,add_to_git_credential=False)
        del secret
    api=HfApi()
    who=api.whoami()
    info=api.repo_info(repo,repo_type="dataset")
    # User authorized public GitHub code, while simulation data stays private.
    if not info.private:api.update_repo_settings(repo,repo_type="dataset",private=True)
    print(json.dumps({"account":who["name"],"private":True}),flush=True)
    if args.data or args.data_subdir:
        name=args.data_subdir or "poc_fit24"
        folder=(STAGE/"data"/name).resolve()
        if not folder.is_relative_to((STAGE/"data").resolve()):raise ValueError("Data directory escapes stage1/data")
        result=api.upload_folder(repo_id=repo,repo_type="dataset",folder_path=folder,path_in_repo="stage1/"+name,
                                 commit_message="Archive C++ response dataset and record splits")
        print(json.dumps({"dataset_commit":result.oid}),flush=True)
    if args.run:
        result=api.upload_folder(repo_id=repo,repo_type="dataset",folder_path=STAGE/"runs"/args.run,path_in_repo="artifacts/"+args.run,
                                 allow_patterns=["*.pt","*.json","*.npz"],commit_message="Archive small-sample fit checkpoint and provenance")
        print(json.dumps({"artifact_commit":result.oid}),flush=True)
        result=api.upload_folder(repo_id=repo,repo_type="dataset",folder_path=STAGE/"reports"/args.run,path_in_repo="reports/"+args.run,
                                 commit_message="Archive fit metrics and response comparison figures")
        print(json.dumps({"report_commit":result.oid}),flush=True)
    if args.summary:
        result=api.upload_folder(repo_id=repo,repo_type="dataset",folder_path=STAGE/"reports",path_in_repo="reports/stage1",
                                 allow_patterns=["poc_result.md","poc_fit24_quality.json","poc_fit24_derived.json","model_check_patch60.json"],
                                 commit_message="Archive POC summary and simulation quality checks")
        print(json.dumps({"summary_commit":result.oid}),flush=True)


if __name__=="__main__":main()
