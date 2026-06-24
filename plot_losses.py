import json
import matplotlib.pyplot as plt

with open('losses.json', 'r') as f:
    losses = json.load(f)

train_losses = losses['train']
val_losses = losses['val']
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.axvline(x=8, color='red', linestyle='--', alpha=0.5, label='Best epoch (8)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('results/loss_curves.png')
plt.close()
print('Loss curves saved.')