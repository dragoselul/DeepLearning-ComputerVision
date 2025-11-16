"""
Point-based loss functions for weakly supervised segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointSupervisionLoss(nn.Module):
    """
    Loss function for point-level supervision
    Only supervises at clicked points, not full mask
    """
    def __init__(self, loss_type='bce'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, y_pred, point_labels, point_coords):
        """
        Args:
            y_pred: Model predictions (B, 1, H, W) - logits
            point_labels: Point labels (B, N) - 0 or 1
            point_coords: Point coordinates (B, N, 2) - (y, x) normalized to [0, 1]
        
        Returns:
            loss: Scalar loss value
        """
        B, _, H, W = y_pred.shape
        N = point_labels.shape[1]
        
        # Sample predictions at point locations
        # Convert normalized coords to pixel coords
        point_coords_pixel = point_coords.clone()
        point_coords_pixel[:, :, 0] = point_coords_pixel[:, :, 0] * H
        point_coords_pixel[:, :, 1] = point_coords_pixel[:, :, 1] * W
        
        # Use grid_sample for differentiable sampling
        # grid_sample expects coords in [-1, 1]
        grid = point_coords.clone()
        grid = grid * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)
        
        # Permute y_pred to (B, C, H, W) and sample
        sampled_logits = F.grid_sample(
            y_pred, 
            grid, 
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (B, 1, N, 1)
        
        sampled_logits = sampled_logits.squeeze(-1).squeeze(1)  # (B, N)
        
        # Compute loss at sampled points
        if self.loss_type == 'bce':
            # Binary cross entropy with logits
            loss = F.binary_cross_entropy_with_logits(
                sampled_logits, 
                point_labels.float()
            )
        elif self.loss_type == 'focal':
            # Focal loss for handling imbalance
            p = torch.sigmoid(sampled_logits)
            ce_loss = F.binary_cross_entropy_with_logits(
                sampled_logits, 
                point_labels.float(), 
                reduction='none'
            )
            p_t = p * point_labels + (1 - p) * (1 - point_labels)
            focal_weight = (1 - p_t) ** 2.0
            loss = (focal_weight * ce_loss).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class PointSupervisionWithRegularizationLoss(nn.Module):
    """
    Point supervision + regularization for better segmentation
    Combines point loss with spatial smoothness
    """
    def __init__(self, point_weight=1.0, reg_weight=0.1, reg_type='tv'):
        super().__init__()
        self.point_loss = PointSupervisionLoss(loss_type='bce')
        self.point_weight = point_weight
        self.reg_weight = reg_weight
        self.reg_type = reg_type
        
    def forward(self, y_pred, point_labels, point_coords):
        """
        Args:
            y_pred: Model predictions (B, 1, H, W) - logits
            point_labels: Point labels (B, N)
            point_coords: Point coordinates (B, N, 2)
        """
        # Point supervision loss
        point_loss = self.point_loss(y_pred, point_labels, point_coords)
        
        # Regularization
        if self.reg_type == 'tv':
            # Total variation (encourages smoothness)
            tv_h = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]).mean()
            tv_w = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]).mean()
            reg_loss = tv_h + tv_w
        elif self.reg_type == 'entropy':
            # Entropy regularization (encourages confident predictions)
            p = torch.sigmoid(y_pred)
            entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
            reg_loss = entropy.mean()
        else:
            reg_loss = 0.0
        
        total_loss = self.point_weight * point_loss + self.reg_weight * reg_loss
        
        return total_loss


class PartialCrossEntropyLoss(nn.Module):
    """
    Alternative: Use partial annotations with ignore label
    Creates soft labels around clicks
    """
    def __init__(self, sigma=5.0, ignore_label=-1):
        super().__init__()
        self.sigma = sigma
        self.ignore_label = ignore_label
        
    def create_soft_labels(self, point_labels, point_coords, H, W):
        """
        Create soft labels from clicks using Gaussian kernels
        
        Args:
            point_labels: (B, N) point labels
            point_coords: (B, N, 2) normalized coords
            H, W: Output size
        
        Returns:
            soft_labels: (B, H, W) with ignore_label where no annotation
        """
        B, N = point_labels.shape
        device = point_labels.device
        
        # Initialize with ignore label
        soft_labels = torch.full((B, H, W), self.ignore_label, 
                                dtype=torch.float32, device=device)
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        y_grid = y_grid.unsqueeze(0)  # (1, H, W)
        x_grid = x_grid.unsqueeze(0)  # (1, H, W)
        
        # For each batch
        for b in range(B):
            for n in range(N):
                label = point_labels[b, n].item()
                y_coord = point_coords[b, n, 0].item() * H
                x_coord = point_coords[b, n, 1].item() * W
                
                # Create Gaussian kernel around point
                dist_sq = (y_grid[0] - y_coord) ** 2 + (x_grid[0] - x_coord) ** 2
                gaussian = torch.exp(-dist_sq / (2 * self.sigma ** 2))
                
                # Apply label with Gaussian weights
                mask = gaussian > 0.1  # Only where significant
                soft_labels[b][mask] = label
        
        return soft_labels
    
    def forward(self, y_pred, point_labels, point_coords):
        """
        Args:
            y_pred: (B, 1, H, W) logits
            point_labels: (B, N) 
            point_coords: (B, N, 2)
        """
        B, _, H, W = y_pred.shape
        
        # Create soft labels
        soft_labels = self.create_soft_labels(point_labels, point_coords, H, W)
        
        # Compute loss only where labels are not ignored
        y_pred_flat = y_pred.squeeze(1)  # (B, H, W)
        
        # Mask for valid labels
        valid_mask = (soft_labels != self.ignore_label)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        
        # BCE loss only on valid pixels
        loss = F.binary_cross_entropy_with_logits(
            y_pred_flat[valid_mask],
            soft_labels[valid_mask],
            reduction='mean'
        )
        
        return loss


if __name__ == '__main__':
    # Test losses
    print("Testing point supervision losses...")
    
    B, H, W = 2, 128, 128
    N = 10  # 10 clicks per image
    
    # Dummy data
    y_pred = torch.randn(B, 1, H, W)
    point_labels = torch.randint(0, 2, (B, N)).float()
    point_coords = torch.rand(B, N, 2)  # normalized [0, 1]
    
    # Test losses
    loss1 = PointSupervisionLoss()
    l1 = loss1(y_pred, point_labels, point_coords)
    print(f"PointSupervisionLoss: {l1.item():.4f}")
    
    loss2 = PointSupervisionWithRegularizationLoss()
    l2 = loss2(y_pred, point_labels, point_coords)
    print(f"PointSupervisionWithRegularizationLoss: {l2.item():.4f}")
    
    loss3 = PartialCrossEntropyLoss()
    l3 = loss3(y_pred, point_labels, point_coords)
    print(f"PartialCrossEntropyLoss: {l3.item():.4f}")
    
    print("\n✓ All losses working!")

