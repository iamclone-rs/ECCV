"""Kaggle training launcher. Imports the exact split from src/data_config.py."""

import argparse
import glob
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.dataset_protocol import category_split, image_paths


def find_dataset(explicit):
    if explicit:
        return Path(explicit)
    roots = set()
    for depth in range(1, 6):
        for sketch in glob.glob('/kaggle/input/' + '*/' * depth + 'sketch'):
            root = Path(sketch).parent
            if (root / 'photo').is_dir():
                roots.add(root)
    # Reject attached Sketchy/TU-Berlin roots instead of picking the first dataset.
    valid = []
    for root in sorted(roots):
        try:
            train, unseen = category_split(argparse.Namespace(root=str(root), dataset='quickdraw', use_classes=80, seed=42))
            if len(train) == 80 and len(unseen) == 30 and len(list(p for p in (root / 'sketch').iterdir() if p.is_dir() and not p.name.startswith('.'))) == 110:
                valid.append(root)
        except (ValueError, OSError):
            continue
    if len(valid) != 1:
        raise RuntimeError(f'Found {len(valid)} compatible QuickDraw roots. Set QUICKDRAW_ROOT to the directory containing sketch/ and photo/. Candidates: {valid}')
    return valid[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=os.environ.get('QUICKDRAW_ROOT', ''))
    parser.add_argument('--clip_weights', default=os.environ.get('ECCV_CLIP_WEIGHTS', '/kaggle/working/clip_cache/ViT-B-32.pt'))
    parser.add_argument('--output_dir', default='/kaggle/working')
    parser.add_argument('--exp_name', default='seco_quickdraw')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--check_only', action='store_true')
    args = parser.parse_args()
    root = find_dataset(args.root)
    split_args = argparse.Namespace(root=str(root), dataset='quickdraw', use_classes=None, seed=42)
    train, unseen = category_split(split_args)
    if len(train) != 80 or len(unseen) != 30:
        raise ValueError(f'QuickDraw requires 80 seen/30 unseen classes; got {len(train)}/{len(unseen)}')
    counts = {}
    for modality in ('sketch', 'photo'):
        for category in train + unseen:
            paths = image_paths(root / modality / category)
            if not paths:
                raise ValueError(f'Empty class: {root / modality / category}')
            counts[f'{modality}/{category}'] = len(paths)
    report = {'root': str(root), 'train_classes': train, 'test_classes': unseen, 'counts': counts}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'quickdraw_preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'QuickDraw ready: {len(train)} train classes / {len(unseen)} test classes.', flush=True)
    print('Test classes from src/data_config.py:', unseen, flush=True)
    if args.check_only:
        return
    if not Path(args.clip_weights).is_file():
        raise FileNotFoundError(f'CLIP checkpoint missing: {args.clip_weights}. Run kaggle_offline.py first.')
    command = [sys.executable, '-u', '-m', 'src.main_train',
        '--root', str(root), '--dataset', 'quickdraw', '--use_classes', '80',
        '--backbone', 'ViT-B/32', '--max_size', '224',
        '--n_ctx', '2', '--img_ctx', '2', '--prompt_depth', '12',
        '--lr', '2e-5', '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
        '--test_batch_size', str(args.test_batch_size), '--workers', str(args.workers),
        '--momentum', '0.9', '--weight_decay', '0.001', '--lr_step_size', '5', '--lr_gamma', '0.1',
        '--triplet_margin', '0.3', '--temperature', '0.07', '--loss_protocol', 'paper',
        '--lambda_class', '1', '--lambda_consistency', '1', '--seed', '42',
        '--eval_query_batch_size', '32', '--clip_weights', args.clip_weights,
        '--output_dir', args.output_dir, '--exp_name', args.exp_name, '--progress']
    if args.ckpt_path:
        command += ['--ckpt_path', args.ckpt_path, '--resume']
    if args.smoke_test:
        command += ['--smoke_test']
    print('Training command:', ' '.join(command), flush=True)
    # Stream output to the cell and preserve it even if training is interrupted.
    log = output / f'{args.exp_name}.log'
    with log.open('a', encoding='utf-8', buffering=1) as stream:
        with subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


if __name__ == '__main__':
    main()
