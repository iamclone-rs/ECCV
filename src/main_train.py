import os
import torch
import numpy as np
import random
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint 

from src.sketchy_dataset import TrainDataset, ValidDataset
from src.model import ZS_SBIR
from src.utils import get_all_categories

def get_datasets(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_dataset = TrainDataset(args, args.proportion)
    val_sketch = ValidDataset(args, mode='sketch')
    val_photo = ValidDataset(args)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True)
    val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=args.test_batch_size, num_workers=args.workers, shuffle=False)
    val_photo_loader = DataLoader(dataset=val_photo, batch_size=args.test_batch_size, num_workers=args.workers, shuffle=False)

    return train_loader, val_sketch_loader, val_photo_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../datasets/tuberlin", help="path to dataset")
    parser.add_argument("--ckpt_path", type=str, default="", help="checkpoint weights, or full state with --resume")
    parser.add_argument("--dataset", type=str, default="tuberlin", help="type of dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--img_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--use_classes", type=int, default=None, help="seen class count; default: all seen classes")
    parser.add_argument("--test_classes_file", default="", help="optional text file, one unseen class per line")
    parser.add_argument("--clip_weights", default="", help="local CLIP .pt file, bypasses downloads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="restore optimizer, scheduler and epoch from ckpt_path")
    parser.add_argument("--triplet_margin", type=float, default=0.3)
    parser.add_argument("--lambda_class", type=float, default=1.0)
    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    parser.add_argument("--loss_protocol", choices=["repo", "paper"], default="repo")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--lr_step_size", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--eval_query_batch_size", type=int, default=32)
    parser.add_argument("--smoke_test", action="store_true", help="two train/val batches; not a benchmark run")
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--distill", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    def boolean(value):
        if str(value).lower() in ('true', '1', 'yes'):
            return True
        if str(value).lower() in ('false', '0', 'no'):
            return False
        raise argparse.ArgumentTypeError('Expected true or false')
    for name in ('use_adapt_sk', 'use_adapt_ph', 'use_adapt_txt', 'use_co_sk', 'use_co_ph'):
        parser.add_argument('--' + name, type=boolean, nargs='?', const=True, default=True)
    for name in ('progress', 'visualize', 'gzs'):
        parser.add_argument('--' + name, type=boolean, nargs='?', const=True, default=False)
    
    parser.add_argument('--exp_name', type=str, default='Co_prompt')
    
    args = parser.parse_args()
    if args.temperature <= 0 or args.triplet_margin <= 0:
        parser.error('temperature and triplet_margin must be positive')
    if args.batch_size < 2 or args.eval_query_batch_size < 1:
        parser.error('batch_size must be >= 2 and eval_query_batch_size must be >= 1')
    if args.ckpt_path and not Path(args.ckpt_path).is_file():
        parser.error(f'Checkpoint does not exist: {args.ckpt_path}')
    if args.resume and not args.ckpt_path:
        parser.error('--resume requires --ckpt_path')
    pl_root = Path(args.output_dir or '.')
    logger = TensorBoardLogger(str(pl_root / 'tb_logs'), name=args.exp_name)
    csv_logger = CSVLogger(str(pl_root / 'csv_logs'), name=args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        dirpath=str(pl_root / 'saved_models' / args.exp_name),
        filename="{epoch:02d}-{mAP:.4f}",
        save_top_k=1,
        mode='max',
        save_last=True)
    
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    train_loader, val_sketch_loader, val_photo_loader = get_datasets(args)
    if len(train_loader) == 0:
        raise ValueError('No complete training batch; reduce batch_size or increase proportion.')
    config_path = pl_root / 'saved_models' / args.exp_name / 'run_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({'args': vars(args),
        'train_classes': train_loader.dataset.all_categories,
        'test_classes': val_sketch_loader.dataset.unseen_classes,
        'train_sketches': len(train_loader.dataset),
        'test_sketches': len(val_sketch_loader.dataset),
        'gallery_photos': len(val_photo_loader.dataset)}, indent=2), encoding='utf-8')
    print(f'Seen/unseen classes: {len(train_loader.dataset.all_categories)}/{len(val_sketch_loader.dataset.unseen_classes)}')
    trainer = Trainer(accelerator='gpu', devices=1, 
        min_epochs=1, max_epochs=args.epochs,
        benchmark=True,
        logger=[logger, csv_logger],
        num_sanity_val_steps=0,
        fast_dev_run=2 if args.smoke_test else False,
        enable_checkpointing=not args.smoke_test,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        callbacks=[] if args.smoke_test else [checkpoint_callback]
    )

    classnames = get_all_categories(args)
 
    if ckpt_path is None or args.resume:
        model = ZS_SBIR(args=args, classname=classnames)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]

        skip = [
            "model.prompt_learner_photo.token_prefix",
            "model.prompt_learner_photo.token_suffix",
            "model.prompt_learner_sketch.token_prefix",
            "model.prompt_learner_sketch.token_suffix",
        ]
        for k in skip:
            sd.pop(k, None)

        model = ZS_SBIR(args=args, classname=classnames)  # classnames = 220
        missing, unexpected = model.load_state_dict(sd, strict=False)

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader],
                ckpt_path=ckpt_path if args.resume else None)
