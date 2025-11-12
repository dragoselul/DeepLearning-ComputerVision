import os
import numpy as np
import glob
from PIL import Image
import argparse


def load_mask(path):
    mask = np.array(Image.open(path).convert('L'))
    mask = (mask > 127).astype(np.uint8)
    return mask


def dice_coefficient(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)


def iou(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-8)


def accuracy(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    return np.sum(pred == gt) / len(gt)


def sensitivity(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp / (tp + fn + 1e-8)


def specificity(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    return tn / (tn + fp + 1e-8)


def evaluate_all_metrics(pred, gt):
    return {
        'dice': dice_coefficient(pred, gt),
        'iou': iou(pred, gt),
        'accuracy': accuracy(pred, gt),
        'sensitivity': sensitivity(pred, gt),
        'specificity': specificity(pred, gt)
    }


def evaluate_dataset(pred_dir, gt_dir, pred_pattern='*.png', gt_pattern='*.png'):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, pred_pattern)))

    if len(pred_files) == 0:
        raise ValueError(f"No prediction files found in {pred_dir} with pattern {pred_pattern}")

    all_metrics = []
    per_image_metrics = {}

    print(f"Evaluating {len(pred_files)} predictions...")

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {filename}, skipping...")
            continue

        pred_mask = load_mask(pred_path)
        gt_mask = load_mask(gt_path)
        metrics = evaluate_all_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        per_image_metrics[filename] = metrics

        print(f"{filename}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
              f"Acc={metrics['accuracy']:.4f}, Sens={metrics['sensitivity']:.4f}, "
              f"Spec={metrics['specificity']:.4f}")

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    std_metrics = {k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

    print("\n" + "="*60)
    print("AVERAGE METRICS:")
    print("="*60)
    for metric_name in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
        print(f"{metric_name.capitalize():15s}: {avg_metrics[metric_name]:.4f} ± {std_metrics[metric_name]:.4f}")
    print("="*60)

    return {
        'average': avg_metrics,
        'std': std_metrics,
        'per_image': per_image_metrics,
        'num_images': len(all_metrics)
    }