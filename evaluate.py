from dataset import get_datasets, NUM_CLASSES, CLASS_NAMES
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from model import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = get_model(device=device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# data
_, val_dataset, test_dataset = get_datasets(image_size=(256, 256))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# accumulators for per-class intersection and union
intersection = torch.zeros(NUM_CLASSES)
union = torch.zeros(NUM_CLASSES)
correct_pixels = 0
total_pixels = 0

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc='Evaluating on test set'):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)['out']
        preds = outputs.argmax(dim=1)
        
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()
        
        for c in range(NUM_CLASSES):
            pred_c = (preds == c)
            mask_c = (masks == c)
            intersection[c] += (pred_c & mask_c).sum().item()
            union[c] += (pred_c | mask_c).sum().item()

# per-class IoU
iou_per_class = intersection / (union + 1e-8)
mean_iou = iou_per_class.mean().item()
pixel_accuracy = correct_pixels / total_pixels

print(f'Mean IoU (test set): {mean_iou:.4f}')
print(f'Pixel Accuracy (test set): {pixel_accuracy:.4f}')
print('\nPer-class IoU:')
for i, name in enumerate(CLASS_NAMES):
    print(f'{name}: {iou_per_class[i]:.4f}')

# save metrics
results = {
    'mean_iou': mean_iou,
    'pixel_accuracy': pixel_accuracy,
    'per_class_iou': {CLASS_NAMES[i]: iou_per_class[i].item() for i in range(NUM_CLASSES)}
}
with open('results/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Metrics saved.')

# visualization — show a few test predictions
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)

images, masks = next(iter(test_loader))
images_gpu = images.to(device)
with torch.no_grad():
    outputs = model(images_gpu)['out']
    preds = outputs.argmax(dim=1).cpu()

n_show = 4
fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 12))
for i in range(n_show):
    img = denormalize(images[i]).permute(1, 2, 0).numpy()
    axes[0, i].imshow(img)
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(masks[i].numpy(), cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
    axes[1, i].set_title('Ground Truth')
    axes[1, i].axis('off')
    
    axes[2, i].imshow(preds[i].numpy(), cmap='tab20', vmin=0, vmax=NUM_CLASSES-1)
    axes[2, i].set_title('Prediction')
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('results/test_predictions.png')
plt.close()
print('Visualization saved.')