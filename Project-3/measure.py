import os
import numpy as np
import glob
from PIL import Image
import argparse
import json


def load_mask(path, size=None):
    mask_img = Image.open(path).convert('L') # Use a new variable name for clarity

    if size is not None:
        if isinstance(size, tuple) and len(size) == 2:
            h, w = size  # height, width (from the pred_mask.shape)
            mask_size = (w, h)  # convert to (width, height) for PIL

            # 2. Perform the resize while it is still a PIL Image object
            mask_img = mask_img.resize(mask_size, Image.Resampling.NEAREST)
        else:
            raise ValueError("size must be a tuple of 2 integers (height, width)")

    # 3. Convert the resized PIL Image object to a NumPy array
    mask = np.array(mask_img)
    
    # 4. Final processing (thresholding)
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



def evaluate_dataset(pred_dir, gt_files, dataset_name, model_name, pred_pattern='*.png'):
    """
    Evaluate predictions against a list of ground-truth masks, match by shared prefix,
    and save results to metrics/model_name/dataset_name/results.json

    Args:
        pred_dir (str): Directory containing predicted masks.
        gt_files (list of str): List of full paths to ground-truth masks.
        dataset_name (str): Name of the dataset (used for saving results).
        model_name (str): Name of the model (used for saving results).
        pred_pattern (str): Glob pattern to match prediction files.
    """

    pred_files = sorted(glob.glob(os.path.join(pred_dir, pred_pattern)))

    if len(pred_files) == 0:
        raise ValueError(f"No prediction files found in {pred_dir} with pattern {pred_pattern}")

    all_metrics = []
    per_image_metrics = {}

    print(f"Evaluating {len(pred_files)} predictions...")

    # Create GT dictionary for matching based on prefix
    gt_dict = {}
    for p in gt_files:
        base = os.path.basename(p)
        name, _ = os.path.splitext(base)
        # Remove common suffixes like _mask or _lesion
        if name.endswith('_mask'):
            name = name[:-5]
        elif name.endswith('_lesion'):
            name = name[:-7]
        gt_dict[name] = p

    for pred_path in pred_files:
        pred_name = os.path.basename(pred_path)
        pred_base, _ = os.path.splitext(pred_name)

        if pred_base not in gt_dict:
            print(f"Warning: No ground-truth found matching prediction {pred_name}, skipping...")
            continue

        gt_path = gt_dict[pred_base]

        pred_mask = load_mask(pred_path)
        # Resize GT to match prediction size
        gt_mask = load_mask(gt_path, size=pred_mask.shape)
        metrics = evaluate_all_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        per_image_metrics[pred_name] = metrics

        print(f"{pred_name}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
              f"Acc={metrics['accuracy']:.4f}, Sens={metrics['sensitivity']:.4f}, "
              f"Spec={metrics['specificity']:.4f}")

    if not all_metrics:
        raise ValueError("No valid predictions matched ground-truth masks.")

    # Compute average and std metrics
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
    std_metrics = {k: float(np.std([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}

    print("\n" + "="*60)
    print("AVERAGE METRICS:")
    print("="*60)
    for metric_name in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']:
        print(f"{metric_name.capitalize():15s}: {avg_metrics[metric_name]:.4f} ± {std_metrics[metric_name]:.4f}")
    print("="*60)

    results = {
        'average': avg_metrics,
        'std': std_metrics,
        'per_image': per_image_metrics,
        'num_images': len(all_metrics)
    }

    # --- Save results to metrics/model_name/dataset_name/results.json ---
    metrics_dir = os.path.join("metrics", model_name, dataset_name)
    os.makedirs(metrics_dir, exist_ok=True)
    results_file = os.path.join(metrics_dir, "results.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {results_file}")

    return results
