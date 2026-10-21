"""Download private runtime assets on 5090 and verify archive/path integrity."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import zipfile

from huggingface_hub import hf_hub_download,HfApi


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sha256",required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    source=hf_hub_download("JasonXF/seismic-transformer-stage1","runtime/windows-mdof-runtime.zip.b64.txt",repo_type="dataset")
    archive=base64.b64decode(Path(source).read_text(encoding="ascii"),validate=True)
    actual=hashlib.sha256(archive).hexdigest()
    if actual!=args.sha256.lower():raise RuntimeError("Runtime archive hash mismatch")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for member in z.infolist():
            target=(root/member.filename).resolve()
            if not target.is_relative_to(root):raise RuntimeError("Archive path escapes project root")
        z.extractall(root)
    info={"runtime_archive_sha256":actual,"destination":str(root),
          "dataset_revision":HfApi().repo_info("JasonXF/seismic-transformer-stage1",repo_type="dataset").sha}
    output=root/"experiment/stage1/reports/runtime_deployment.json"
    output.write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(info),flush=True)


if __name__=="__main__":main()
