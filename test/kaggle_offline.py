"""Offline Kaggle cell 1: restore a bundle built by kaggle_online.py."""

from pathlib import Path
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


WORKING = Path('/kaggle/working')
PROJECT = WORKING / 'ECCV'
STARTED = time.perf_counter()


def log(message):
    print(f'[+{time.perf_counter() - STARTED:.1f}s] {message}', flush=True)


def sha256(path):
    digest = hashlib.sha256()
    total = path.stat().st_size
    checked = 0
    last_report = time.perf_counter()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
            checked += len(block)
            if time.perf_counter() - last_report >= 5:
                log(f'  SHA256 {path.name}: {checked / 1024**2:.1f}/{total / 1024**2:.1f} MiB')
                last_report = time.perf_counter()
    return digest.hexdigest()


def main():
    log('[1/7] Looking for the offline bundle (CPU/disk; training starts in the next cell).')
    selected = os.environ.get('ECCV_BUNDLE', '')
    if selected:
        log(f'Using ECCV_BUNDLE: {selected}')
        candidates = [Path(selected) / 'bundle_manifest.json']
    else:
        candidates = []
        # Bound the search depth; never walk the individual QuickDraw images.
        for depth in range(1, 7):
            log(f'Searching Kaggle inputs at depth {depth}/6...')
            pattern = '/kaggle/input/' + '*/' * depth + 'offline_bundle/bundle_manifest.json'
            candidates.extend(Path(p) for p in glob.glob(pattern))
    log(f'Manifest files found: {len(set(candidates))}')
    matches = []
    for path in sorted(set(candidates)):
        log(f'Reading manifest: {path}')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        log(f"  Task: {manifest.get('task')}; base commit: {manifest.get('base_commit', 'unknown')}")
        if manifest.get('task') == 'seco_sbir_quickdraw' and manifest.get('schema_version') == 1:
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise RuntimeError(f'Found {len(matches)} ECCV bundles. Attach one bundle, or set ECCV_BUNDLE to its offline_bundle directory.')
    bundle, manifest = matches[0]
    log(f'Selected bundle: {bundle}')
    log('[2/7] Checking Python and file checksums...')
    if manifest['python_major_minor'] != list(sys.version_info[:2]):
        raise RuntimeError('Python version differs from the online builder. Rebuild using the same Kaggle image.')
    log(f'Python {sys.version.split()[0]}: OK')
    records = manifest['files']
    total_bytes = sum(record['size'] for record in records)
    log(f'Files to verify: {len(records)} ({total_bytes / 1024**2:.1f} MiB)')
    verified_bytes = 0
    for index, record in enumerate(records, start=1):
        log(f"  [{index}/{len(records)}] Checking {record['path']} ({record['size'] / 1024**2:.1f} MiB)...")
        path = (bundle / record['path']).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError('Invalid bundle file path.')
        if not path.is_file() or path.stat().st_size != record['size'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'Bundle integrity check failed: {path}')
        verified_bytes += record['size']
        log(f'  OK ({verified_bytes / 1024**2:.1f}/{total_bytes / 1024**2:.1f} MiB verified)')
    log('[3/7] Checking installed PyTorch versions...')
    from importlib.metadata import version
    for name, expected in manifest['torch_stack'].items():
        actual = version(name)
        log(f'  {name}: installed={actual}, expected={expected}')
        if actual != expected:
            raise RuntimeError(f'{name} version differs: expected {expected}, got {actual}. Rebuild on this Kaggle image.')
    log('[4/7] Installing dependencies from offline wheels; pip output follows...')
    log(f"Wheel directory: {bundle / 'wheels'}")
    WORKING.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
                    str(bundle / 'wheels'), '--only-binary=:all:',
                    '-c', str(bundle / 'requirements.txt'), '-c', str(bundle / 'torch_constraints.txt'),
                    *manifest['runtime_requirements']], cwd=WORKING, check=True)
    log('Offline dependencies installed.')
    log(f'[5/7] Copying source to {PROJECT}...')
    if PROJECT.exists():
        count = 1
        backup = WORKING / f'ECCV_previous_{count}'
        while backup.exists():
            count += 1
            backup = WORKING / f'ECCV_previous_{count}'
        PROJECT.rename(backup)
        log(f'Previous project preserved: {backup}')
    shutil.copytree(bundle / 'source' / 'ECCV', PROJECT, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    log('Source copied.')
    weights = WORKING / 'clip_cache' / 'ViT-B-32.pt'
    weights.parent.mkdir(exist_ok=True)
    log(f'[6/7] Copying CLIP weights to {weights}...')
    shutil.copy2(bundle / 'clip_cache' / weights.name, weights)
    log(f'CLIP weights copied: {weights.stat().st_size / 1024**2:.1f} MiB')
    os.environ['ECCV_CLIP_WEIGHTS'] = str(weights)
    os.environ['HF_HUB_OFFLINE'] = '1'
    log('[7/7] Checking training entrypoint imports with --help (no training yet)...')
    subprocess.run([sys.executable, '-m', 'src.main_train', '--help'], cwd=PROJECT,
                   check=True, stdout=subprocess.DEVNULL)
    log(f'Offline setup complete: {PROJECT}')
    log('Next: run test/kaggle_train.ipy to start GPU training. Set QUICKDRAW_ROOT if automatic dataset detection is ambiguous.')


if __name__ == '__main__':
    main()
