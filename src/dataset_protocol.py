"""Resolve class splits once for both the datasets and text prototypes."""

import random
import re
from pathlib import Path

from src.data_config import GENERALIZED_CLASSES, UNSEEN_CLASSES


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def image_paths(directory):
    return sorted(str(p) for p in Path(directory).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def category_split(args):
    root = Path(args.root)
    categories = sorted(p.name for p in (root / 'sketch').iterdir()
                        if p.is_dir() and not p.name.startswith('.'))
    split_file = getattr(args, 'test_classes_file', '')
    if split_file:
        unseen = [line.strip() for line in Path(split_file).read_text(encoding='utf-8').splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
    else:
        unseen = list(UNSEEN_CLASSES[args.dataset])

    if args.dataset == 'quickdraw':
        # Dataset releases use both palm_tree and palm tree. Keep the real names.
        normalize = lambda name: re.sub(r'[ _-]+', ' ', name.lower()).strip()
        lookup = {}
        for category in categories:
            key = normalize(category)
            if key in lookup:
                raise ValueError(f'Ambiguous QuickDraw class names: {lookup[key]}, {category}')
            lookup[key] = category
        unseen = [lookup.get(normalize(name), name) for name in unseen]

    if not unseen or len(unseen) != len(set(unseen)):
        raise ValueError('Test classes must be nonempty and unique.')
    missing = sorted(set(unseen) - set(categories))
    if missing:
        raise ValueError(f'Test classes missing from {root / "sketch"}: {missing}')
    for category in categories:
        if not (root / 'photo' / category).is_dir():
            raise ValueError(f'Missing photo directory for class: {category}')

    train = sorted(set(categories) - set(unseen))
    count = getattr(args, 'use_classes', None)
    if count is not None and count != len(train):
        if not 2 <= count <= len(train):
            raise ValueError(f'use_classes must be between 2 and {len(train)}, got {count}')
        required = sorted(set(GENERALIZED_CLASSES.get(args.dataset, ())) & set(train))
        if count < len(required):
            raise ValueError('use_classes is smaller than the required generalized class set.')
        others = sorted(set(train) - set(required))
        rng = random.Random(getattr(args, 'seed', 42))
        train = sorted(required + rng.sample(others, count - len(required)))
    if len(train) < 2:
        raise ValueError('Triplet training requires at least two seen classes.')
    return train, unseen
