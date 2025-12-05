def iou_xyxy(a, b, eps=1e-12):
    #for a (two corner points to make box)
    #Top-left: (ax1, ay1)
    #Bottom-right: (ax2, ay2)
    
    #then
    #Top-right: (ax2, ay1)
    #Bottom-left: (ax1, ay2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # intersection
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    # areas
    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - inter
    if union <= 0.0:   # covers degenerate boxes
        return 0.0
    return inter / (union + eps)

import matplotlib.pyplot as plt

def plot_pr_curve(points, ap=None, title="Precision–Recall Curve"):
    """
    points: list of (precision, recall) tuples
    ap: (optional) display AP in title
    """
    if not points:
        print("No points to plot.")
        return
    precisions, recalls = zip(*points)
    plt.figure(figsize=(6,4))
    plt.plot(recalls, precisions, marker='o', label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.xlim([0, 1.05])
    plt.grid(True)
    plt.title(f"{title}" + (f" (AP={ap:.3f})" if ap is not None else ""))
    plt.legend()
    plt.show()
    
    
import numpy as np
def compute_ap_from_points(points):
    if not points:
        return 0.0

    # Unzip points
    precisions, recalls = zip(*points)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Pad start and end
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Make precision non-increasing (right-to-left max)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    # Only add area where recall increases
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return ap

def ap_score(preds, gts, iou_thresh=0.5):
    """
    preds: list of dicts {'imageid': ..., 'score': ..., 'box': ...}
    gts: dict gt_id: {'imageid': ..., 'bbox': ...}
    Returns: list of (precision, recall) tuples after each pred
    """
    tp = 0
    fp = 0
    tp_plus_fn = len(gts)
    gts_used = set()  # Store matched gt_id
    preds = sorted(preds, key=lambda d: d['score'], reverse=True)  # Sort by score descending
    points = []

    for p in preds:
        pimageid, pbox = p['imageid'], p['box']

        # Find GTs for this image that haven't been matched
        image_gts = [
            (gt_id, gt['bbox']) for gt_id, gt in gts.items()
            if gt['imageid'] == pimageid and gt_id not in gts_used
        ]

        # If no GTs left for image, it's a FP
        if not image_gts:
            fp += 1
            points.append((tp / (tp + fp), tp / tp_plus_fn))
            continue

        # Find best match by IoU
        best_gt_id, best_gt_box, best_iou = None, None, 0.0
        for gt_id, gt_box in image_gts:
            iou = iou_xyxy(pbox, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
                best_gt_box = gt_box

        if best_iou >= iou_thresh:
            gts_used.add(best_gt_id)
            tp += 1
        else:
            fp += 1

        points.append((tp / (tp + fp), tp / tp_plus_fn))

    return points
