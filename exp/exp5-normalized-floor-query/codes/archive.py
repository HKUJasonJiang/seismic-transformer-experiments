"""Upload code, configuration, predictions, frozen weights and figures privately."""
import os
from pathlib import Path
from huggingface_hub import HfApi

if __name__ == '__main__':
    exp = Path(__file__).resolve().parents[1]
    repo = os.environ['HF_REPO_ID']
    api = HfApi()
    if not api.repo_info(repo_id=repo, repo_type='dataset').private:
        raise RuntimeError('Full experiment artifacts require a private repository')
    result = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(exp),
                              path_in_repo='experiments/' + exp.name,
                              ignore_patterns=['**/__pycache__/**', '**/*.zip', '**/*.log', '**/*.tmp'],
                              commit_message='Archive exp5 normalized floor query training and separate generalization tests')
    files = api.list_repo_files(repo_id=repo, repo_type='dataset', revision=result.oid)
    prefix = 'experiments/' + exp.name + '/'
    archived = [f for f in files if f.startswith(prefix)]
    expected = [p.relative_to(exp).as_posix() for p in exp.rglob('*') if p.is_file()
                and '__pycache__' not in p.parts and p.suffix not in ('.zip', '.log', '.tmp')]
    assert set(prefix+p for p in expected).issubset(archived)
    print('Private archive commit: ' + result.oid)
    print('Verified file count: ' + str(len(archived)))
