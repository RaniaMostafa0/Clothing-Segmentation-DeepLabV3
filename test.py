# test.py
from dataset import get_datasets
import torch
from model import get_model
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')

train_dataset, val_dataset, test_dataset = get_datasets()
print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

loader = DataLoader(train_dataset, batch_size=2)
images, masks = next(iter(loader))
print(f'Image shape: {images.shape}')
print(f'Mask shape: {masks.shape}')
print(f'Mask unique values: {masks.unique()}')

model = get_model(device=device)
with torch.no_grad():
    output = model(images.to(device))['out']
print(f'Output shape: {output.shape}')
print('All good!')