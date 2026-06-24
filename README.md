# Clothing-Segmentation-DeepLabV3plus

Semantic segmentation of clothing items from photographs of people, using DeepLabV3+ with a ResNet101 backbone, fine-tuned on a remapped subset of the ATR (Active Template Regression) dataset. Includes training, quantitative evaluation (IoU, pixel accuracy), a standalone inference pipeline, and an interactive Streamlit web app.

## Overview

Semantic segmentation (rather than instance segmentation) was chosen because the task requires per-pixel labeling of clothing on a single person, matching ATR's annotation format. Body-part labels (hair, face, arms, legs) were remapped to background, and left/right shoe were merged into a single `Shoes` class, reducing the label space to 11 task-relevant classes:

`Background, Hat, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt, Shoes, Bag, Scarf`

## Architecture

- **Backbone**: ResNet101 (ImageNet/COCO-pretrained), encoding 256×256 input down to a 16×16, 2048-channel feature map.
- **ASPP (Atrous Spatial Pyramid Pooling)**: Parallel atrous convolutions at dilation rates 6, 12, 18, capturing multi-scale context without further downsampling.
- **Decoder**: Upsamples ASPP output and fuses it with low-level ResNet features via a skip connection to recover spatial detail.
- **Output head**: Final classifier replaced to predict 11 classes instead of the original 21 COCO classes.
- **Loss**: CrossEntropyLoss (standard for multiclass, mutually-exclusive pixel classification).

## Dataset

- **Source**: ATR dataset, loaded via HuggingFace (`mattmdjaga/human_parsing_dataset`), 17,706 image-mask pairs.
- **Split**: 80% train (14,166) / 10% val (1,770) / 10% test (1,770), fixed seed for reproducibility.
- **Augmentation** (training only): random horizontal flip (50%), random brightness adjustment ±20% (50%).

## Results

| Metric | Value |
|---|---|
| Mean IoU (test set) | **0.601** |
| Pixel Accuracy (test set) | **0.961** |
| Training epochs | 15 |
| Best checkpoint | Epoch 8 (val loss 0.1185) |
| Hardware | NVIDIA RTX 2080 |
| Optimizer | Adam, lr 1e-4, batch size 8 |

The gap between pixel accuracy (96.1%) and mean IoU (60.1%) reflects class imbalance — Background alone covers most pixels (0.975 IoU), inflating pixel accuracy independently of smaller-class performance.

### Per-Class IoU (test set)

| Class | IoU |
|---|---|
| Background | 0.975 |
| Pants | 0.767 |
| Upper-clothes | 0.762 |
| Bag | 0.653 |
| Skirt | 0.634 |
| Shoes | 0.622 |
| Hat | 0.608 |
| Dress | 0.611 |
| Scarf | 0.372 |
| Sunglasses | 0.379 |
| Belt | 0.222 |

Large, frequent classes segment well; small/infrequent accessories (Belt, Sunglasses, Scarf) underperform due to limited pixel footprint and lower training representation.

### Training Curve
![Loss curves](results/loss_curves.png)

### Test Set Predictions
![Test predictions](results/test_predictions.png)

## Real-World Testing

The model was additionally tested on 45 personal photographs beyond the ATR test set to assess generalization:

- **Strengths**: Reliable segmentation of large clothing categories (Dress, Upper-clothes, Pants); correctly handles multi-person images when subjects are clearly separated.
- **Limitations**: 
  - Small accessory classes (Belt, Sunglasses, Scarf) frequently misclassified.
  - Clothing missed/misclassified when the subject occupies a small fraction of the frame.
  - Segmentation degrades with 4-6+ people in frame.
  - Occasional misclassification on unusual camera angles (e.g. overhead shots).

Sample inference outputs are in `results/inference_*.png`.

## Project Structure

- `model.py` — DeepLabV3+ model setup (ResNet101 backbone, modified classifier head)
- `dataset.py` — Dataset loading, label remapping, augmentation, train/val/test split
- `train.py` — Training/validation loop
- `evaluate.py` — Test-set IoU and pixel accuracy computation, sample prediction visualization
- `inference.py` — Standalone inference on arbitrary photographs (CLI tool)
- `plot_losses.py` — Loss curve visualization
- `app.py` — Streamlit web app for interactive segmentation
- `test.py` — Sanity-check script (dataset/model shape verification)
- `examples/` — Sample personal photographs used for qualitative testing
- `results/` — Evaluation metrics, plots, and inference outputs
- `losses.json` — Per-epoch train/val loss history

**Note**: Trained model weights (`best_model.pth`, `final_model.pth`, ~233MB each) are excluded from this repository due to GitHub's file size limits. Run `train.py` to reproduce them.

## Setup

```bash
git clone https://github.com/RaniaMostafa0/Clothing-Segmentation-DeepLabV3plus.git
cd Clothing-Segmentation-DeepLabV3plus

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install PyTorch with CUDA support (adjust cu121 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Reproducing Results

```bash
# 1. Train the model (downloads ATR dataset automatically via HuggingFace, ~15 epochs)
python train.py
# Saves: best_model.pth, final_model.pth, losses.json

# 2. Evaluate on the held-out test set
python evaluate.py
# Saves: results/metrics.json, results/test_predictions.png
# Expected: Mean IoU ≈ 0.601, Pixel Accuracy ≈ 0.961

# 3. Plot training/validation loss curves
python plot_losses.py
# Saves: results/loss_curves.png

# 4. Run inference on a custom photo
python inference.py path/to/photo.jpg
# Saves: results/inference_<filename>.png

# 5. (Optional) Launch the interactive web app
streamlit run app.py
```

## Future Work

- Combined CrossEntropy + Dice loss to improve performance on underrepresented classes.
- Training at higher resolution to recover finer boundary detail.
- Expanded training data for multi-person scenes.
