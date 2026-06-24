from datasets import load_dataset
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from model import get_model
from dataset import NUM_CLASSES, CLASS_NAMES
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = get_model(device=device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# preprocessing — same as training
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def segment_image(image_path, alpha=0.5, save_path=None):
    # load original image
    original = Image.open(image_path).convert('RGB')
    original_size = original.size  # (width, height)
    
    # preprocess for model
    input_tensor = preprocess(original).unsqueeze(0).to(device)
    
    # predict
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
    
    # resize mask back to original dimensions
    pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_resized = pred_mask_pil.resize(original_size, Image.NEAREST)
    pred_mask_resized = np.array(pred_mask_resized)
    
    # apply colormap
    cmap = plt.get_cmap('tab20')
    colored_mask = cmap(pred_mask_resized / (NUM_CLASSES - 1))[:, :, :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    colored_mask_pil = Image.fromarray(colored_mask)
    
    # overlay on original
    original_np = np.array(original)
    overlay = (original_np * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)
    
    # plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask_pil)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay_pil)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # legend
    cmap_tab20 = plt.get_cmap('tab20')
    patches = [mpatches.Patch(
        color=cmap_tab20(i / (NUM_CLASSES - 1)),
        label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved to {save_path}')
    
    plt.show()
    return pred_mask_resized

# run inference
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python inference.py <image_path>')
        print('Example: python inference.py photo.jpg')
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f'Image not found: {image_path}')
        sys.exit(1)
    
    os.makedirs('results', exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f'results/inference_{filename}.png'
    
    segment_image(image_path, alpha=0.5, save_path=save_path)