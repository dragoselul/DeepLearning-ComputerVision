"""
Weak Supervision Module for Segmentation

This module provides tools for training segmentation models with weak supervision
using point clicks instead of full pixel-level masks.
"""

from .generate_clicks import sample_points_from_mask, visualize_clicks
from .point_losses import (
    PointSupervisionLoss,
    PointSupervisionWithRegularizationLoss,
    PartialCrossEntropyLoss
)
from .weak_dataset import WeakPH2Dataset
from .train_weak import train_weak_model, evaluate_weak_model

__all__ = [
    'sample_points_from_mask',
    'visualize_clicks',
    'PointSupervisionLoss',
    'PointSupervisionWithRegularizationLoss',
    'PartialCrossEntropyLoss',
    'WeakPH2Dataset',
    'train_weak_model',
    'evaluate_weak_model',
]

