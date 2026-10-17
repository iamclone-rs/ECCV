"""Online Kaggle cell: package ECCV source, Python wheels and CLIP weights.

Paste this file into an Internet-enabled Kaggle notebook and Run All.
ECCV_SOURCE can point at an attached copy of THIS edited project (recommended).
Otherwise ECCV_REPO/ECCV_REF select the Git revision to clone.
"""

from datetime import datetime, timezone
from importlib.metadata import distribution, version
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request


REPO = os.environ.get('ECCV_REPO', 'https://github.com/iamclone-rs/ECCV.git')
REF = os.environ.get('ECCV_REF', 'main')
WORKING = Path('/kaggle/working')
BUNDLE = WORKING / ('seco_quickdraw_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')) / 'offline_bundle'
SOURCE = BUNDLE / 'source' / 'ECCV'
WHEELS = BUNDLE / 'wheels'
ROOTS = ['pytorch-lightning>=2.2,<3', 'torchmetrics>=1.3,<2', 'ftfy', 'regex',
         'tensorboard', 'numpy', 'pillow', 'matplotlib', 'scikit-learn', 'tqdm', 'packaging', 'bresenham']
CLIP_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
CLIP_URL = f'https://openaipublic.azureedge.net/clip/models/{CLIP_SHA256}/ViT-B-32.pt'


def run(command, cwd=WORKING):
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    WORKING.mkdir(parents=True, exist_ok=True)
    WHEELS.mkdir(parents=True)
    SOURCE.parent.mkdir(parents=True)
    attached = os.environ.get('ECCV_SOURCE', '')
    if not attached and '__file__' in globals():
        candidate = Path(__file__).resolve().parent.parent
        if (candidate / 'src' / 'dataset_protocol.py').is_file():
            attached = str(candidate)
    if attached:
        local = Path(attached).resolve()
        if local == SOURCE.resolve() or local in SOURCE.resolve().parents:
            raise ValueError('ECCV_SOURCE must be outside the output bundle directory.')
        shutil.copytree(local, SOURCE, ignore=shutil.ignore_patterns(
            '.git', '__pycache__', '.pytest_cache', '.venv', 'saved_models',
            'tb_logs', 'csv_logs', 'offline_bundle', '*.ckpt', '*.zip'))
        commit = 'local-source'
        if (local / '.git').exists():
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=local, text=True).strip()
    else:
        run(['git', 'clone', '--branch', REF, '--single-branch', REPO, str(SOURCE)])
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    for filename in ('src/dataset_protocol.py', 'src/retrieval.py', 'test/kaggle_train.py'):
        if not (SOURCE / filename).is_file():
            raise FileNotFoundError(f'{filename} is missing. Attach the edited ECCV source and set ECCV_SOURCE, '
                                    'or push these edits to ECCV_REPO/ECCV_REF before building.')
    print('[1/4] Source:', SOURCE, 'base commit:', commit, flush=True)

    # Keep the CUDA/PyTorch installation provided by the Kaggle image.
    constraints = BUNDLE / 'torch_constraints.txt'
    stack = {}
    for name in ('torch', 'torchvision', 'torchaudio'):
        try:
            stack[name] = version(name)
        except Exception:
            pass
    constraints.write_text(''.join(f'{name}=={value}\n' for name, value in stack.items()), encoding='utf-8')
    run([sys.executable, '-m', 'pip', 'install', '-q', '-c', str(constraints), *ROOTS])
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pending = [Requirement(spec).name for spec in ROOTS]
    resolved = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in resolved or name in {'torch', 'torchvision', 'torchaudio', 'triton'} or name.startswith(('nvidia-', 'cuda-')):
            continue
        installed = distribution(name)
        resolved[name] = installed.version
        for spec in installed.requires or []:
            requirement = Requirement(spec)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    requirements = BUNDLE / 'requirements.txt'
    requirements.write_text(''.join(f'{name}=={value}\n' for name, value in sorted(resolved.items())), encoding='utf-8')
    run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
         '--dest', str(WHEELS), '-r', str(requirements)])
    print('[2/4] Wheels:', len(list(WHEELS.glob('*.whl'))), flush=True)

    weights = BUNDLE / 'clip_cache' / 'ViT-B-32.pt'
    weights.parent.mkdir()
    cache = Path.home() / '.cache' / 'clip' / weights.name
    if cache.is_file() and sha256(cache) == CLIP_SHA256:
        shutil.copy2(cache, weights)
    else:
        urllib.request.urlretrieve(CLIP_URL, weights)
    if sha256(weights) != CLIP_SHA256:
        raise RuntimeError('CLIP checkpoint checksum mismatch.')
    print('[3/4] CLIP ViT-B/32 weights verified.', flush=True)
    run([sys.executable, '-m', 'src.main_train', '--help'], cwd=SOURCE)

    # Inventory the actual source, including edits that have not been committed.
    files = []
    for path in sorted(BUNDLE.rglob('*')):
        if path.is_file() and '.git' not in path.relative_to(BUNDLE).parts and '__pycache__' not in path.parts:
            files.append({'path': path.relative_to(BUNDLE).as_posix(),
                          'size': path.stat().st_size, 'sha256': sha256(path)})
    manifest = {'task': 'seco_sbir_quickdraw', 'schema_version': 1, 'repository': REPO,
                'ref': REF, 'base_commit': commit, 'entrypoint': 'src.main_train',
                'python_major_minor': list(sys.version_info[:2]), 'torch_stack': stack,
                'runtime_requirements': ROOTS, 'files': files}
    (BUNDLE / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('[4/4] Complete:', BUNDLE)
    print('Save Version -> Save & Run All, preserving outputs. Attach these outputs to the offline notebook.')


if __name__ == '__main__':
    main()
