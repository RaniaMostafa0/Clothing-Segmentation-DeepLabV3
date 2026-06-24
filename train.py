from dataset import get_datasets, NUM_CLASSES
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from model import get_model
import os

# hyperparameters
epochs = 15
batch_size = 8
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data
train_dataset, val_dataset, test_dataset = get_datasets(image_size=(256, 256))
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                        shuffle=False, pin_memory=True)

# model
model = get_model(device=device)

# loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc='Training')
    
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# training loop
all_train_losses = []
all_val_losses = []
best_val_loss = float('inf') 

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}')
    
    # save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Model saved at epoch {epoch+1}')

torch.save(model.state_dict(), 'final_model.pth')
with open('losses.json', 'w') as f:
    json.dump({'train': all_train_losses, 'val': all_val_losses}, f)
print('Done.')