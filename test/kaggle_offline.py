"""Offline Kaggle cell 1: restore a bundle built by kaggle_online.py."""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys


WORKING = Path('/kaggle/working')
PROJECT = WORKING / 'ECCV'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    selected = os.environ.get('ECCV_BUNDLE', '')
    if selected:
        candidates = [Path(selected) / 'bundle_manifest.json']
    else:
        candidates = []
        # Bound the search depth; never walk the individual QuickDraw images.
        for depth in range(1, 7):
            pattern = '/kaggle/input/' + '*/' * depth + 'offline_bundle/bundle_manifest.json'
            candidates.extend(Path(p) for p in glob.glob(pattern))
    matches = []
    for path in sorted(set(candidates)):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('task') == 'seco_sbir_quickdraw' and manifest.get('schema_version') == 1:
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise RuntimeError(f'Found {len(matches)} ECCV bundles. Attach one bundle, or set ECCV_BUNDLE to its offline_bundle directory.')
    bundle, manifest = matches[0]
    if manifest['python_major_minor'] != list(sys.version_info[:2]):
        raise RuntimeError('Python version differs from the online builder. Rebuild using the same Kaggle image.')
    for record in manifest['files']:
        path = (bundle / record['path']).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError('Invalid bundle file path.')
        if not path.is_file() or path.stat().st_size != record['size'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'Bundle integrity check failed: {path}')
    from importlib.metadata import version
    for name, expected in manifest['torch_stack'].items():
        if version(name) != expected:
            raise RuntimeError(f'{name} version differs: expected {expected}, got {version(name)}. Rebuild on this Kaggle image.')
    WORKING.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
                    str(bundle / 'wheels'), '--only-binary=:all:',
                    '-c', str(bundle / 'requirements.txt'), '-c', str(bundle / 'torch_constraints.txt'),
                    *manifest['runtime_requirements']], cwd=WORKING, check=True)
    if PROJECT.exists():
        count = 1
        backup = WORKING / f'ECCV_previous_{count}'
        while backup.exists():
            count += 1
            backup = WORKING / f'ECCV_previous_{count}'
        PROJECT.rename(backup)
        print('Previous project preserved:', backup)
    shutil.copytree(bundle / 'source' / 'ECCV', PROJECT, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    weights = WORKING / 'clip_cache' / 'ViT-B-32.pt'
    weights.parent.mkdir(exist_ok=True)
    shutil.copy2(bundle / 'clip_cache' / weights.name, weights)
    os.environ['ECCV_CLIP_WEIGHTS'] = str(weights)
    os.environ['HF_HUB_OFFLINE'] = '1'
    subprocess.run([sys.executable, '-m', 'src.main_train', '--help'], cwd=PROJECT,
                   check=True, stdout=subprocess.DEVNULL)
    print('Offline setup complete:', PROJECT)
    print('Next: run test/kaggle_train.ipy. Set QUICKDRAW_ROOT if automatic dataset detection is ambiguous.')


if __name__ == '__main__':
    main()
