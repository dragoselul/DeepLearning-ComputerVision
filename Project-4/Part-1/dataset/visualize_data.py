import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import os
from potholes_dataset import PotholeDataset


def visualize_samples(dataset, num_samples=4, output_file='visualization.png'):
    # Handle if dataset is smaller than requested samples
    num_samples = min(len(dataset), num_samples)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1: axes = [axes]  # Ensure axes is iterable

    # Pick random indices from the dataset
    indices = torch.randperm(len(dataset))[:num_samples]

    for i, idx in enumerate(indices):
        img, target = dataset[idx]

        # 1. Un-normalize logic (Critical step!)
        # Your loader uses ImageNet stats: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        # We must reverse: pixel = (input * std) + mean
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img_np = img.permute(1, 2, 0).numpy()  # Move C to last dim: (H, W, C)
        img_np = std * img_np + mean  # Reverse normalization
        img_np = np.clip(img_np, 0, 1)  # Clip to valid 0-1 range for imshow

        ax = axes[i]
        ax.imshow(img_np)
        ax.axis('off')

        # 2. Draw Boxes
        boxes = target['boxes']
        for box in boxes:
            xmin, ymin, xmax, ymax = box.numpy()
            width = xmax - xmin
            height = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

        ax.set_title(f"Image ID: {target['image_id'].item()}")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", "potholes")

    if os.path.exists(ROOT):
        # 1. Create the loaders
        train_loader, _, _ = PotholeDataset.create_dataloaders(ROOT)

        # 2. Pass the DATASET (not the loader) to the visualizer
        visualize_samples(train_loader.dataset)
    else:
        print(f"Error: Dataset not found at {ROOT}")
