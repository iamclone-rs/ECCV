import os
import numpy as np
import torch
import random
from torchvision import transforms
from PIL import Image, ImageOps
from src.data_config import GENERALIZED_CLASSES, VISUALIZE_CLASSES
from src.dataset_protocol import category_split, image_paths

def aumented_transform():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
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
    def __init__(self, args, proportion=1.0):
        self.args = args
        self.proportion = proportion
        self.transform1 = normal_transform()
        self.transform2 = aumented_transform()
        
        self.all_categories, _ = category_split(args)
        if not 0 < proportion <= 1:
            raise ValueError('proportion must be in (0, 1].')

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            sketch_paths = image_paths(os.path.join(self.args.root, 'sketch', category))
            photo_paths = image_paths(os.path.join(self.args.root, 'photo', category))
            if not sketch_paths or not photo_paths:
                raise ValueError(f'Empty sketch/photo class: {category}')
            
            if self.proportion != 1:
                num_sketch = max(1, int(len(sketch_paths) * self.proportion))
                num_photo  = max(1, int(len(photo_paths) * self.proportion))
                
                sketch_paths = random.sample(sketch_paths, num_sketch)
                photo_paths = random.sample(photo_paths, num_photo)
            
            self.all_sketches_path.extend(sketch_paths)
            self.all_photos_path[category] = photo_paths

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

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))

        sk_tensor  = self.transform1(sk_data)
        img_tensor = self.transform1(img_data)
        neg_tensor = self.transform1(neg_data)
        
        sk_aug_tensor = self.transform2(sk_data)
        img_aug_tensor = self.transform2(img_data)
        
        return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, self.all_categories.index(category)


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()
        self.gzs_perc = 0.2
        self.seed = 42
        
        self.global_categories, unseen = category_split(args)
        
        if self.args.visualize:
            self.unseen_classes = VISUALIZE_CLASSES[self.args.dataset]
            # self.unseen_classes = UNSEEN_CLASSES[self.args.dataset]
        else:
            self.unseen_classes = unseen

        self.label_classes = sorted(set(self.unseen_classes) | set(self.global_categories))
            
        unseen_paths = []
        for category in self.unseen_classes:
            if self.mode == 'photo':
                paths = image_paths(os.path.join(self.args.root, 'photo', category))
            else:
                paths = image_paths(os.path.join(self.args.root, 'sketch', category))
            if not paths:
                raise ValueError(f'Empty validation class ({mode}): {category}')
            unseen_paths.extend(paths)

        self.paths = list(unseen_paths)

        if self.mode == 'photo':
            if self.args.gzs:
                generalized_classes = GENERALIZED_CLASSES.get(self.args.dataset)
                if generalized_classes is None:
                    raise ValueError(f'No GZS protocol defined for {self.args.dataset}')
                for category in generalized_classes:
                    self.paths.extend(image_paths(os.path.join(self.args.root, 'photo', category)))
            
    def __getitem__(self, index):
        filepath = self.paths[index]                
        category = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        image_tensor = self.transform(image)
        
        labels = self.unseen_classes if self.args.visualize else self.label_classes
        return image_tensor, labels.index(category)
    
    def __len__(self):
        return len(self.paths)
