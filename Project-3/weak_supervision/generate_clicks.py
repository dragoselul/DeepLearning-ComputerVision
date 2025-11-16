"""
Generate weak annotations (point clicks) from full segmentation masks
Simulates user clicking on lesions (positive) and background (negative)
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import random


def sample_points_from_mask(mask, num_positive=5, num_negative=5, strategy='centroid'):
    """
    Sample point clicks from a binary mask
    
    Args:
        mask: Binary mask (H, W) with 0=background, 1=foreground
        num_positive: Number of positive clicks (on lesion)
        num_negative: Number of negative clicks (on background)
        strategy: Sampling strategy - 'random', 'centroid', 'boundary', 'mixed'
    
    Returns:
        positive_points: List of (y, x) coordinates for positive clicks
        negative_points: List of (y, x) coordinates for negative clicks
    """
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    
    positive_points = []
    negative_points = []
    
    # === POSITIVE CLICKS (on lesion) ===
    foreground_pixels = np.argwhere(mask > 0)
    
    if len(foreground_pixels) == 0:
        # No foreground, sample from center
        positive_points = [(h//2, w//2)] * num_positive
    else:
        if strategy == 'random':
            # Random sampling from foreground
            indices = np.random.choice(len(foreground_pixels), 
                                     min(num_positive, len(foreground_pixels)), 
                                     replace=False)
            positive_points = [tuple(foreground_pixels[i]) for i in indices]
            
        elif strategy == 'centroid':
            # Sample near centroid (simulates user clicking center)
            cy, cx = np.mean(foreground_pixels, axis=0).astype(int)
            positive_points.append((cy, cx))
            
            # Sample remaining points around centroid with some spread
            for _ in range(num_positive - 1):
                # Add gaussian noise around centroid
                offset_y = int(np.random.normal(0, h * 0.1))
                offset_x = int(np.random.normal(0, w * 0.1))
                y = np.clip(cy + offset_y, 0, h-1)
                x = np.clip(cx + offset_x, 0, w-1)
                
                # Make sure it's still on foreground
                if mask[y, x] > 0:
                    positive_points.append((y, x))
                else:
                    # Fallback to random foreground pixel
                    idx = np.random.choice(len(foreground_pixels))
                    positive_points.append(tuple(foreground_pixels[idx]))
                    
        elif strategy == 'boundary':
            # Sample near boundaries (simulates clicking edges)
            dist_transform = distance_transform_edt(mask)
            # Get points at different distances from boundary
            for i in range(num_positive):
                # Sample at different depth percentiles
                percentile = (i + 1) / (num_positive + 1) * 100
                threshold = np.percentile(dist_transform[mask > 0], percentile)
                candidates = np.argwhere((dist_transform >= threshold * 0.9) & 
                                       (dist_transform <= threshold * 1.1) & 
                                       (mask > 0))
                if len(candidates) > 0:
                    idx = np.random.choice(len(candidates))
                    positive_points.append(tuple(candidates[idx]))
                else:
                    idx = np.random.choice(len(foreground_pixels))
                    positive_points.append(tuple(foreground_pixels[idx]))
                    
        elif strategy == 'mixed':
            # Mix of strategies (most realistic)
            # 1 centroid click
            cy, cx = np.mean(foreground_pixels, axis=0).astype(int)
            positive_points.append((cy, cx))
            
            # Remaining clicks distributed
            remaining = num_positive - 1
            for i in range(remaining):
                if i % 2 == 0:
                    # Random click
                    idx = np.random.choice(len(foreground_pixels))
                    positive_points.append(tuple(foreground_pixels[idx]))
                else:
                    # Click near existing point (simulates correction)
                    base_point = positive_points[-1]
                    offset_y = int(np.random.normal(0, h * 0.05))
                    offset_x = int(np.random.normal(0, w * 0.05))
                    y = np.clip(base_point[0] + offset_y, 0, h-1)
                    x = np.clip(base_point[1] + offset_x, 0, w-1)
                    if mask[y, x] > 0:
                        positive_points.append((y, x))
                    else:
                        idx = np.random.choice(len(foreground_pixels))
                        positive_points.append(tuple(foreground_pixels[idx]))
    
    # === NEGATIVE CLICKS (on background) ===
    background_pixels = np.argwhere(mask == 0)
    
    if len(background_pixels) == 0:
        # No background (shouldn't happen), use corners
        negative_points = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)][:num_negative]
    else:
        if strategy in ['random', 'mixed']:
            # Random background sampling
            indices = np.random.choice(len(background_pixels), 
                                     min(num_negative, len(background_pixels)), 
                                     replace=False)
            negative_points = [tuple(background_pixels[i]) for i in indices]
            
        elif strategy in ['centroid', 'boundary']:
            # Sample background points near foreground boundary
            # (simulates user clicking around the lesion)
            if len(foreground_pixels) > 0:
                # Dilate mask to get near-boundary region
                kernel = np.ones((15, 15), np.uint8)
                dilated = cv2.dilate(mask, kernel, iterations=1)
                near_boundary = (dilated > 0) & (mask == 0)
                boundary_bg_pixels = np.argwhere(near_boundary)
                
                if len(boundary_bg_pixels) > 0:
                    # Sample from near boundary
                    indices = np.random.choice(len(boundary_bg_pixels), 
                                             min(num_negative, len(boundary_bg_pixels)), 
                                             replace=False)
                    negative_points = [tuple(boundary_bg_pixels[i]) for i in indices]
                else:
                    # Fallback to random
                    indices = np.random.choice(len(background_pixels), 
                                             min(num_negative, len(background_pixels)), 
                                             replace=False)
                    negative_points = [tuple(background_pixels[i]) for i in indices]
            else:
                indices = np.random.choice(len(background_pixels), 
                                         min(num_negative, len(background_pixels)), 
                                         replace=False)
                negative_points = [tuple(background_pixels[i]) for i in indices]
    
    # Pad lists if needed
    while len(positive_points) < num_positive:
        positive_points.append(positive_points[-1] if positive_points else (h//2, w//2))
    while len(negative_points) < num_negative:
        negative_points.append(negative_points[-1] if negative_points else (0, 0))
    
    return positive_points[:num_positive], negative_points[:num_negative]


def visualize_clicks(image, mask, positive_points, negative_points, save_path=None):
    """
    Visualize clicks on image for debugging
    
    Args:
        image: RGB image (H, W, 3)
        mask: Ground truth mask (H, W)
        positive_points: List of (y, x) for positive clicks
        negative_points: List of (y, x) for negative clicks
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Image with clicks
    axes[2].imshow(image)
    for y, x in positive_points:
        axes[2].plot(x, y, 'go', markersize=10, markeredgewidth=2, 
                    markeredgecolor='white', label='Positive' if (y, x) == positive_points[0] else '')
    for y, x in negative_points:
        axes[2].plot(x, y, 'rx', markersize=10, markeredgewidth=2, 
                    markeredgecolor='white', label='Negative' if (y, x) == negative_points[0] else '')
    axes[2].set_title(f'Clicks ({len(positive_points)}+, {len(negative_points)}-)')
    axes[2].legend()
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # Test the click generation
    print("Testing click generation...")
    
    # Create synthetic mask
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 1, -1)
    
    strategies = ['random', 'centroid', 'boundary', 'mixed']
    
    for strategy in strategies:
        pos, neg = sample_points_from_mask(mask, num_positive=5, num_negative=5, strategy=strategy)
        print(f"\n{strategy.upper()} strategy:")
        print(f"  Positive clicks: {len(pos)}")
        print(f"  Negative clicks: {len(neg)}")
        print(f"  Example pos: {pos[:2]}")
        print(f"  Example neg: {neg[:2]}")

