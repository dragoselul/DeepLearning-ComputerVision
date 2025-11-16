"""
Dataset for weakly supervised segmentation with point annotations
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.PH2Dataset import PH2Dataset as PH2DatasetBase
from weak_supervision.generate_clicks import sample_points_from_mask


class WeakPH2Dataset(Dataset):
    """
    PH2 Dataset with weak point annotations instead of full masks
    """
    def __init__(self, split, transform, label_transform=None, 
                 num_positive=5, num_negative=5, 
                 sampling_strategy='mixed', augment=False):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: Image transforms
            label_transform: Label transforms (for size matching)
            num_positive: Number of positive clicks per image
            num_negative: Number of negative clicks per image
            sampling_strategy: 'random', 'centroid', 'boundary', 'mixed'
            augment: Whether to apply augmentation
        """
        # Use base PH2 dataset to get image/label paths
        self.base_dataset = PH2DatasetBase(
            split=split, 
            transform=transform, 
            label_transform=label_transform,
            augment=augment
        )
        
        self.transform = transform
        self.label_transform = label_transform
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.sampling_strategy = sampling_strategy
        self.augment = augment
        
        # Access paths from base dataset
        self.image_paths = self.base_dataset.image_paths
        self.label_paths = self.base_dataset.label_paths
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor (C, H, W)
            point_labels: Tensor of point labels (N,) - 0 or 1
            point_coords: Tensor of point coordinates (N, 2) - (y, x) normalized [0, 1]
            full_mask: Full ground truth mask for evaluation only (C, H, W)
        """
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        
        # Apply synchronized augmentation if enabled
        if self.augment:
            import torchvision.transforms.functional as TF
            import random
            
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)
            
            # Color jitter (only for image)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        
        # Convert label to numpy for click sampling BEFORE transforms
        label_np = np.array(label)
        label_np = (label_np > 127).astype(np.uint8)
        
        # Sample point clicks from original mask
        positive_points, negative_points = sample_points_from_mask(
            label_np,
            num_positive=self.num_positive,
            num_negative=self.num_negative,
            strategy=self.sampling_strategy
        )
        
        # Get original size for normalization
        orig_h, orig_w = label_np.shape
        
        # Apply transforms to image and label
        X = self.transform(image)
        if self.label_transform is not None:
            Y = self.label_transform(label)
        else:
            import torchvision.transforms.functional as TF
            Y = TF.resize(label, X.shape[-2:])
            Y = TF.to_tensor(Y)
        
        # Binarize label
        Y = (Y > 0.5).float()
        
        # Get transformed size
        _, new_h, new_w = X.shape
        
        # Scale point coordinates to match transformed size
        # Points are in (y, x) format
        scaled_positive = [(y * new_h / orig_h, x * new_w / orig_w) 
                          for y, x in positive_points]
        scaled_negative = [(y * new_h / orig_h, x * new_w / orig_w) 
                          for y, x in negative_points]
        
        # Combine points and create labels
        all_points = scaled_positive + scaled_negative
        all_labels = [1] * len(scaled_positive) + [0] * len(scaled_negative)
        
        # Normalize coordinates to [0, 1]
        normalized_coords = [(y / new_h, x / new_w) for y, x in all_points]
        
        # Convert to tensors
        point_coords = torch.tensor(normalized_coords, dtype=torch.float32)  # (N, 2)
        point_labels = torch.tensor(all_labels, dtype=torch.float32)  # (N,)
        
        return X, point_labels, point_coords, Y


def test_weak_dataset():
    """Test the weak dataset"""
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    print("Testing WeakPH2Dataset...")
    
    size = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    label_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    
    # Test different click configurations
    configs = [
        (3, 3, 'random'),
        (5, 5, 'centroid'),
        (10, 10, 'mixed'),
    ]
    
    for num_pos, num_neg, strategy in configs:
        print(f"\n--- Config: {num_pos}+ / {num_neg}- clicks, {strategy} strategy ---")
        
        dataset = WeakPH2Dataset(
            split='train',
            transform=transform,
            label_transform=label_transform,
            num_positive=num_pos,
            num_negative=num_neg,
            sampling_strategy=strategy,
            augment=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading
        X, point_labels, point_coords, full_mask = dataset[0]
        
        print(f"Image shape: {X.shape}")
        print(f"Point labels shape: {point_labels.shape}")
        print(f"Point coords shape: {point_coords.shape}")
        print(f"Full mask shape: {full_mask.shape}")
        print(f"Point labels: {point_labels}")
        print(f"Coord range: [{point_coords.min():.3f}, {point_coords.max():.3f}]")
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        X_batch, labels_batch, coords_batch, masks_batch = next(iter(loader))
        
        print(f"\nBatch shapes:")
        print(f"  Images: {X_batch.shape}")
        print(f"  Labels: {labels_batch.shape}")
        print(f"  Coords: {coords_batch.shape}")
        print(f"  Masks: {masks_batch.shape}")
    
    print("\n✓ WeakPH2Dataset test complete!")


if __name__ == '__main__':
    test_weak_dataset()

