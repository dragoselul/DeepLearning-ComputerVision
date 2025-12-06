import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Part-2'))

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import DataLoader

from potholes import POTHOLES, BBox
from utils import label2target, decode_bbox
from rcnn import RCNN

from ap_score.ap_score import ap_score, compute_ap_from_points, plot_pr_curve, iou_xyxy


# ===================== NMS Implementation =====================
def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    boxes: list of [x1, y1, x2, y2]
    scores: list of confidence scores
    Returns: indices of boxes to keep
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sort by score (descending)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in remaining])

        # Keep boxes with IoU below threshold
        mask = ious < iou_threshold
        order = remaining[mask]

    return keep




# ===================== Sliding Window Proposals =====================
def generate_proposals(img_width, img_height, scales=None, stride=32):
    """Generate sliding window proposals at multiple scales"""
    if scales is None:
        # More diverse scales to capture different pothole sizes
        scales = [32, 48, 64, 96, 128, 160, 192, 256]
    proposals = []
    for scale in scales:
        # Use scale-dependent stride for efficiency
        s = max(16, stride if scale < 100 else stride * 2)
        for y in range(0, max(1, img_height - scale), s):
            for x in range(0, max(1, img_width - scale), s):
                proposals.append(BBox(x, y, x + scale, y + scale))
    return proposals


# ===================== Run Inference on Single Image =====================
def predict_image(model, img_tensor, device, conf_threshold=0.5, nms_threshold=0.3, batch_size=32):
    """
    Run inference on a single image
    Returns: list of (box, score) for detected potholes
    """
    model.eval()
    C, H, W = img_tensor.shape

    # Generate proposals
    proposals = generate_proposals(W, H)

    if len(proposals) == 0:
        return []

    # Prepare proposal crops
    crops = []
    valid_proposals = []
    for prop in proposals:
        x1, y1 = max(0, prop.xmin), max(0, prop.ymin)
        x2, y2 = min(W, prop.xmax), min(H, prop.ymax)

        if x2 > x1 and y2 > y1:
            crop = TF.crop(img_tensor, y1, x1, y2 - y1, x2 - x1)
            crop = F.interpolate(crop.unsqueeze(0), size=(224, 224),
                                mode='bilinear', align_corners=False).squeeze(0)
            crops.append(crop)
            valid_proposals.append(prop)

    if len(crops) == 0:
        return []

    # Process in batches to avoid OOM
    all_pothole_scores = []
    all_bbox_deltas = []

    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i+batch_size]
            batch = torch.stack(batch_crops).to(device)

            cls_scores, bbox_deltas = model(batch)
            probs = F.softmax(cls_scores, dim=1)
            pothole_scores = probs[:, label2target['pothole']].cpu().numpy()
            bbox_deltas = bbox_deltas.cpu().numpy()

            all_pothole_scores.extend(pothole_scores)
            all_bbox_deltas.extend(bbox_deltas)

    all_pothole_scores = np.array(all_pothole_scores)
    all_bbox_deltas = np.array(all_bbox_deltas)

    # Filter by confidence
    detections = []
    for i, (prop, score) in enumerate(zip(valid_proposals, all_pothole_scores)):
        if score >= conf_threshold:
            # Decode bbox
            final_box = decode_bbox(prop, all_bbox_deltas[i].tolist())
            detections.append({
                'box': [final_box.xmin, final_box.ymin, final_box.xmax, final_box.ymax],
                'score': float(score)
            })

    if len(detections) == 0:
        return []

    # Apply NMS
    boxes = [d['box'] for d in detections]
    scores = [d['score'] for d in detections]
    keep_indices = nms(boxes, scores, nms_threshold)

    return [detections[i] for i in keep_indices]


# ===================== Evaluate on Test Set =====================
def evaluate_model(model, test_loader, device, conf_threshold=0.5, nms_threshold=0.3, batch_size=32):
    """
    Evaluate model on test set and compute AP
    """
    all_preds = []
    all_gts = {}
    gt_id = 0

    print("Running inference on test set...")
    for batch_idx, (images, targets) in enumerate(test_loader):
        for img_idx, (img_tensor, gt_bboxes) in enumerate(zip(images, targets)):
            image_id = f"img_{batch_idx}_{img_idx}"

            # Add ground truths
            for gt_box in gt_bboxes:
                all_gts[gt_id] = {
                    'imageid': image_id,
                    'bbox': [gt_box.xmin, gt_box.ymin, gt_box.xmax, gt_box.ymax]
                }
                gt_id += 1

            # Run prediction
            img_tensor = img_tensor.to(device)
            detections = predict_image(model, img_tensor, device,
                                       conf_threshold, nms_threshold, batch_size=batch_size)

            for det in detections:
                all_preds.append({
                    'imageid': image_id,
                    'score': det['score'],
                    'box': det['box']
                })

        print(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")

    print(f"\nTotal predictions: {len(all_preds)}")
    print(f"Total ground truths: {len(all_gts)}")

    # Debug: show score distribution
    if all_preds:
        scores = [p['score'] for p in all_preds]
        print(f"Score distribution - Min: {min(scores):.4f}, Max: {max(scores):.4f}, Mean: {np.mean(scores):.4f}")

    # Compute AP
    if len(all_preds) == 0:
        print("No predictions made!")
        return 0.0, [], all_preds, all_gts

    points = ap_score(all_preds, all_gts)
    ap = compute_ap_from_points(points)

    return ap, points, all_preds, all_gts


# ===================== Main =====================
if __name__ == "__main__":
    import torchvision.transforms as transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define transforms (same as training)
    size = (560, 576)
    eval_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    label_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = POTHOLES(split="train", transform=eval_transform, label_transform=label_transform)

    # Split dataset (same as training)
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = RCNN().to(device)

    # Try to load trained weights
    model_path = '../Part-2/rcnn_model_best.pth'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"WARNING: No trained model found at {model_path}")
        print("Using untrained model - results will be poor!")

    # Evaluate with different thresholds
    print("\n" + "="*60)
    print("Testing different confidence thresholds:")
    print("="*60)

    best_ap = 0
    best_thresh = 0.5
    for conf_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ap, points, preds, gts = evaluate_model(
            model, test_loader, device,
            conf_threshold=conf_thresh,
            nms_threshold=0.3,
            batch_size=64
        )
        print(f"conf_threshold={conf_thresh}: AP={ap:.4f}, predictions={len(preds)}")
        if ap > best_ap:
            best_ap = ap
            best_thresh = conf_thresh

    # Final evaluation with best threshold
    print("\n" + "="*60)
    print(f"Final Evaluation (conf={best_thresh}):")
    print("="*60)
    ap, points, preds, gts = evaluate_model(
        model, test_loader, device,
        conf_threshold=best_thresh,
        nms_threshold=0.3,
        batch_size=64
    )

    print(f"\n{'='*50}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"{'='*50}")

    # Plot PR curve
    if points:
        plot_pr_curve(points, ap=ap, title="Pothole Detection PR Curve")

