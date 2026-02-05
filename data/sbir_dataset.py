import os
import random
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp"}

def _list_images(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXT:
                out.append(os.path.join(dirpath, fn))
    return out

def _classname_from_path(path):
    return os.path.basename(os.path.dirname(path))

@dataclass
class Split:
    seen: list
    unseen: list

def build_seen_unseen(all_classes, unseen_classes):
    unseen = [c for c in all_classes if c in set(unseen_classes)]
    seen = [c for c in all_classes if c not in set(unseen_classes)]
    return Split(seen=seen, unseen=unseen)

class SBIRTripletTrain(Dataset):
    def __init__(self, root, seen_classes, tfm_sketch, tfm_photo):
        self.root = root
        self.seen_classes = sorted(seen_classes)
        self.tfm_sketch = tfm_sketch
        self.tfm_photo = tfm_photo

        self.sk_root = os.path.join(root, "sketch")
        self.ph_root = os.path.join(root, "photo")

        self.sk_by_class = {}
        self.ph_by_class = {}

        for c in self.seen_classes:
            sk_dir = os.path.join(self.sk_root, c)
            ph_dir = os.path.join(self.ph_root, c)
            sk_list = _list_images(sk_dir) if os.path.isdir(sk_dir) else []
            ph_list = _list_images(ph_dir) if os.path.isdir(ph_dir) else []
            if len(sk_list) > 0 and len(ph_list) > 0:
                self.sk_by_class[c] = sk_list
                self.ph_by_class[c] = ph_list

        self.valid_classes = sorted(list(self.sk_by_class.keys()))
        self.class_to_idx = {c:i for i,c in enumerate(self.valid_classes)}

        self.samples = []
        for c in self.valid_classes:
            for p in self.sk_by_class[c]:
                self.samples.append((p, c))

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        sk_path, c = self.samples[idx]

        pos_ph_path = random.choice(self.ph_by_class[c])

        neg_c = c
        while neg_c == c:
            neg_c = random.choice(self.valid_classes)
        neg_ph_path = random.choice(self.ph_by_class[neg_c])

        sk = self.tfm_sketch(self._load(sk_path))
        ph_pos = self.tfm_photo(self._load(pos_ph_path))
        ph_neg = self.tfm_photo(self._load(neg_ph_path))

        y_pos = self.class_to_idx[c]
        y_neg = self.class_to_idx[neg_c]

        return sk, ph_pos, ph_neg, y_pos, y_neg

class SBIREvalIndex(Dataset):
    def __init__(self, root, classes, modality, tfm):
        self.root = root
        self.classes = sorted(classes)
        self.modality = modality
        self.tfm = tfm

        self.base = os.path.join(root, modality)
        self.items = []
        for c in self.classes:
            d = os.path.join(self.base, c)
            if not os.path.isdir(d):
                continue
            for p in _list_images(d):
                self.items.append((p, c))

        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        p, c = self.items[idx]
        x = self.tfm(self._load(p))
        y = self.class_to_idx[c]
        return x, y
