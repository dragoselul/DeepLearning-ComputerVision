import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = torch.sum(y_pred * y_true)
        dice = (2. * intersection) / (torch.sum(y_pred) + torch.sum(y_true) + 1e-8)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        p = torch.sigmoid(y_pred)
        ce_loss = y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred))
        p_t = p * y_true + (1 - p) * (1 - y_true)
        focal_weight = self.alpha * ((1 - p_t) ** self.gamma)
        loss = focal_weight * ce_loss
        return torch.mean(loss)

class BCELossTotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        tv_h = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        tv_w = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        regularization = torch.mean(tv_h) + torch.mean(tv_w)
        return loss + 0.1*regularization


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.dice = DiceLoss()

    def forward(self, y_pred, y_true):
        # Dice loss
        dice_loss = self.dice(y_pred, y_true)

        # Weighted BCE loss
        if self.pos_weight is not None:
            # Apply positive class weighting
            bce_loss = torch.mean(
                self.pos_weight * y_true * torch.log(torch.sigmoid(y_pred) + 1e-8) +
                (1 - y_true) * torch.log(1 - torch.sigmoid(y_pred) + 1e-8)
            )
            bce_loss = -bce_loss
        else:
            bce_loss = torch.mean(
                y_pred - y_true * y_pred + torch.log(1 + torch.exp(-y_pred))
            )

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
