"""
Training script for weakly supervised segmentation with point annotations
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.UNetModel import UNet
from models.EncDecModel import EncDec
from weak_supervision.weak_dataset import WeakPH2Dataset
from weak_supervision.point_losses import PointSupervisionLoss, PointSupervisionWithRegularizationLoss
from measure import evaluate_all_metrics, load_mask
import numpy as np


def train_weak_model(model, train_loader, val_loader, loss_fn, opt, device,
                     save_name, epochs=20, patience=5):
    """
    Train model with point supervision

    Args:
        model: Segmentation model
        train_loader: Weak training data (with points)
        val_loader: Validation data (with full masks for evaluation)
        loss_fn: Point supervision loss
        opt: Optimizer
        device: cuda/cpu
        save_name: Path to save model
        epochs: Max epochs
        patience: Early stopping patience
    """
    best_val_dice = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training with Point Supervision")
    print(f"{'='*60}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        print(f'\n* Epoch {epoch+1}/{epochs}')

        avg_loss = 0
        for X_batch, point_labels, point_coords, _ in train_loader:
            X_batch = X_batch.to(device)
            point_labels = point_labels.to(device)
            point_coords = point_coords.to(device)

            opt.zero_grad()

            # Forward pass
            y_pred = model(X_batch)

            # Point supervision loss
            loss = loss_fn(y_pred, point_labels, point_coords)

            loss.backward()
            opt.step()

            avg_loss += loss / len(train_loader)

        print(f' - train_loss: {avg_loss:.4f}')

        # Validation phase - evaluate on FULL masks
        if val_loader is not None:
            model.eval()
            all_dice = []
            all_iou = []

            with torch.no_grad():
                for X_val, _, _, Y_val in val_loader:
                    X_val = X_val.to(device)
                    Y_val = Y_val.cpu().numpy()

                    # Predict
                    val_pred = model(X_val)
                    val_pred_prob = torch.sigmoid(val_pred)
                    val_pred_binary = (val_pred_prob > 0.5).float().cpu().numpy()

                    # Compute metrics per image
                    for i in range(len(X_val)):
                        pred_mask = val_pred_binary[i, 0]
                        true_mask = Y_val[i, 0]

                        # Compute Dice
                        intersection = np.sum(pred_mask * true_mask)
                        dice = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(true_mask) + 1e-8)
                        all_dice.append(dice)

                        # Compute IoU
                        union = np.sum(pred_mask) + np.sum(true_mask) - intersection
                        iou = intersection / (union + 1e-8)
                        all_iou.append(iou)

            avg_dice = np.mean(all_dice)
            avg_iou = np.mean(all_iou)

            print(f' - val_dice: {avg_dice:.4f}, val_iou: {avg_iou:.4f}')

            # Early stopping based on Dice
            if avg_dice > best_val_dice:
                best_val_dice = avg_dice
                patience_counter = 0
                best_epoch = epoch + 1
                torch.save(model, save_name)
                print(f'   ✓ New best model saved (Dice: {avg_dice:.4f})')
            else:
                patience_counter += 1
                print(f'   No improvement ({patience_counter}/{patience})')

                if patience_counter >= patience:
                    print(f'\n⚠ Early stopping! Best epoch: {best_epoch}')
                    print(f'   Best val Dice: {best_val_dice:.4f}')
                    return save_name
        else:
            torch.save(model, save_name)

    print(f"\n✓ Training finished!")
    if val_loader is not None:
        print(f"  Best model from epoch {best_epoch} (Dice: {best_val_dice:.4f})")

    return save_name


def evaluate_weak_model(model_path, test_loader, device, output_dir=None):
    """
    Evaluate weakly trained model on test set with full masks

    Args:
        model_path: Path to trained model
        test_loader: Test dataloader (with full masks)
        device: cuda/cpu
        output_dir: Optional directory to save predictions

    Returns:
        metrics: Dictionary with average metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Model")
    print(f"{'='*60}")

    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()

    all_metrics = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get image paths from dataset for proper naming
    dataset = test_loader.dataset
    # Handle Subset wrapper
    if hasattr(dataset, 'dataset'):
        # It's a Subset, get the underlying dataset
        base_dataset = dataset.dataset
        image_paths = base_dataset.image_paths
    else:
        image_paths = dataset.image_paths

    image_idx = 0
    with torch.no_grad():
        for batch_idx, (X_test, _, _, Y_test) in enumerate(test_loader):
            X_test = X_test.to(device)
            Y_test = Y_test.cpu().numpy()

            # Predict
            y_pred = model(X_test)
            y_pred_prob = torch.sigmoid(y_pred)
            y_pred_binary = (y_pred_prob > 0.5).float().cpu().numpy()

            # Evaluate each image
            for i in range(len(X_test)):
                pred_mask = y_pred_binary[i, 0]
                true_mask = Y_test[i, 0]

                # Compute all metrics
                metrics = evaluate_all_metrics(pred_mask, true_mask)
                all_metrics.append(metrics)

                # Save prediction if requested
                if output_dir:
                    from PIL import Image
                    pred_img = (pred_mask * 255).astype(np.uint8)

                    # Get original image filename
                    original_filename = os.path.basename(image_paths[image_idx])
                    # Remove extension and add .png
                    filename_no_ext = os.path.splitext(original_filename)[0]
                    pred_filename = f'{filename_no_ext}.png'

                    Image.fromarray(pred_img).save(
                        os.path.join(output_dir, pred_filename)
                    )

                image_idx += 1

    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        std_metrics = np.std([m[key] for m in all_metrics])
        print(f"{key.capitalize():15s}: {avg_metrics[key]:.4f} ± {std_metrics:.4f}")

    return avg_metrics


if __name__ == '__main__':
    print("Testing weak supervision training...")

    # Configuration
    size = (256, 256)
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    label_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    # Create weak dataset
    print("\nLoading weak PH2 dataset...")
    train_dataset = WeakPH2Dataset(
        split='train',
        transform=transform,
        label_transform=label_transform,
        num_positive=5,
        num_negative=5,
        sampling_strategy='mixed',
        augment=True
    )

    val_dataset = WeakPH2Dataset(
        split='val',
        transform=transform,
        label_transform=label_transform,
        num_positive=5,
        num_negative=5,
        sampling_strategy='mixed',
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Setup model and training
    print("\nSetting up model...")
    model = UNet().to(device)
    loss_fn = PointSupervisionWithRegularizationLoss(
        point_weight=1.0,
        reg_weight=0.01,
        reg_type='tv'
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train for a few epochs as test
    print("\nStarting training test (3 epochs)...")
    trained_model = train_weak_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        opt=opt,
        device=device,
        save_name='test_weak_unet.pth',
        epochs=3,
        patience=10
    )

    print("\n✓ Weak supervision training test complete!")

