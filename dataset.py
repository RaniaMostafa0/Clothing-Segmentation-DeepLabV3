from datasets import load_dataset
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image

LABEL_MAP = {
    0: 0, 1: 1, 2: 0, 3: 2, 4: 3, 5: 4,
    6: 5, 7: 6, 8: 7, 9: 8, 10: 8,
    11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
    16: 9, 17: 10
}

NUM_CLASSES = 11

CLASS_NAMES = [
    'Background', 'Hat', 'Sunglasses', 'Upper-clothes',
    'Skirt', 'Pants', 'Dress', 'Belt', 'Shoes', 'Bag', 'Scarf'
]

def remap_mask(mask_array):
    new_mask = np.zeros_like(mask_array)
    for original, new in LABEL_MAP.items():
        new_mask[mask_array == original] = new
    return new_mask

class ClothingSegmentationDataset(Dataset):
    def __init__(self, data, image_size=(256, 256), augment=False):
        self.data = data
        self.image_size = image_size
        self.augment = augment
        
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image'].convert('RGB')
        mask = sample['mask'].convert('L')
        
        # resize
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        # augmentation — training only
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, 
                         brightness_factor=random.uniform(0.8, 1.2))
        
        # convert to tensors
        image = self.image_transform(image)
        mask_array = np.array(mask)
        mask_array = remap_mask(mask_array)
        mask = torch.tensor(mask_array, dtype=torch.long)
        
        return image, mask


def get_datasets(image_size=(256, 256), val_split=0.1, test_split=0.1, seed=42):
    full_data = load_dataset("mattmdjaga/human_parsing_dataset")['train']
    
    total = len(full_data)
    val_size = int(total * val_split)
    test_size = int(total * test_split)
    train_size = total - val_size - test_size
    
    indices = list(range(total))
    random.seed(seed)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = full_data.select(train_indices)
    val_data = full_data.select(val_indices)
    test_data = full_data.select(test_indices)
    
    train_dataset = ClothingSegmentationDataset(train_data, image_size=image_size, augment=True)
    val_dataset = ClothingSegmentationDataset(val_data, image_size=image_size, augment=False)
    test_dataset = ClothingSegmentationDataset(test_data, image_size=image_size, augment=False)
    
    return train_dataset, val_dataset, test_dataset