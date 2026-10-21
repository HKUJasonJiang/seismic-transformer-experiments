"""Archive this version to the user's existing PRIVATE dataset repository."""
import os
from huggingface_hub import HfApi
from common import EXP

if __name__ == '__main__':
    repo = os.environ['HF_REPO_ID']
    api = HfApi()
    info = api.repo_info(repo_id=repo, repo_type='dataset')
    if not info.private: raise RuntimeError('Full experiment artifacts require a private repository')
    result = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(EXP),
                               path_in_repo='experiments/' + EXP.name,
                               ignore_patterns=['**/__pycache__/**', '**/*.zip', '**/*.log', '**/*.tmp'],
                               commit_message='Archive exp3 response teacher: code, configuration, checkpoints, MSE and figures')
    print('Private archive commit: ' + result.oid)
