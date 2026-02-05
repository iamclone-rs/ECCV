import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from src.data_config import UNSEEN_CLASSES

def aumented_transform():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, opts, return_orig=False):
        self.opts = opts
        self.transform = normal_transform()
        self.return_orig = return_orig
        self.aumentation = aumented_transform()
        unseen_classes = UNSEEN_CLASSES[self.opts.dataset]

        self.all_categories = os.listdir(os.path.join(self.opts.root, 'sketch'))
        self.all_categories = list(set(self.all_categories) - set(unseen_classes))

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.opts.root, 'sketch', category, '*')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.opts.root, 'photo', category, '*'))

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        sk_aug_tensor = self.aumentation(sk_data)
        img_aug_tensor = self.aumentation(img_data)
        
        return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, self.all_categories.index(category)

class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()
        unseen_classes = UNSEEN_CLASSES[self.args.dataset]
        self.all_categories = list(set(unseen_classes))

        self.paths = []
        for category in self.all_categories:
            if self.mode == "photo":
                self.paths.extend(glob.glob(os.path.join(self.args.root, 'photo', category, '*')))
            else:
                self.paths.extend(glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))

    def __getitem__(self, index):
        filepath = self.paths[index]                
        category = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        image_tensor = self.transform(image)
        
        return image_tensor, self.all_categories.index(category)
    
    def __len__(self):
        return len(self.paths)