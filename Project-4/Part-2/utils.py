try:
    from .potholes import BBox
except ImportError:
    from potholes import BBox
import numpy as np
import random
from typing import List 
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from matplotlib import patches
import matplotlib.pyplot as plt
from typing import Callable, List

label2target = {'background': 0, 'pothole': 1}
target2label = {v: k for k, v in label2target.items()}


def decode(_clss):
    _, preds = _clss.max(-1)
    return preds

def encode_bbox(p_bbox: BBox, g_bbox: BBox) -> List[float]:
    """
    Encodes the transformation (tx, ty, tw, th) from a proposal box (p) 
    to a ground truth box (g).

    Inputs are BBox objects with coordinates scaled to the input image size (e.g., 560x576).
    """
    px, py, pw, ph = (p_bbox.xmin + p_bbox.xmax) / 2., \
                     (p_bbox.ymin + p_bbox.ymax) / 2., \
                     p_bbox.xmax - p_bbox.xmin, \
                     p_bbox.ymax - p_bbox.ymin
    
    gx, gy, gw, gh = (g_bbox.xmin + g_bbox.xmax) / 2., \
                     (g_bbox.ymin + g_bbox.ymax) / 2., \
                     g_bbox.xmax - g_bbox.xmin, \
                     g_bbox.ymax - g_bbox.ymin
    
    # Avoid division by zero for proposals with zero width/height
    eps = np.finfo(float).eps
    pw = max(pw, eps)
    ph = max(ph, eps)
    gw = max(gw, eps)
    gh = max(gh, eps)

    t_x = (gx - px) / pw
    t_y = (gy - py) / ph
    t_w = np.log(gw / pw)
    t_h = np.log(gh / ph)
    
    return [t_x, t_y, t_w, t_h]


def IoU(box_a: BBox, box_b: BBox) -> float:
    """Calculates Intersection over Union (IoU) of two BBox objects."""
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a.xmin, box_b.xmin)
    y_a = max(box_a.ymin, box_b.ymin)
    x_b = min(box_a.xmax, box_b.xmax)
    y_b = min(box_a.ymax, box_b.ymax)

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # compute the area of both the prediction and ground-truth rectangles
    box_a_area = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin)
    box_b_area = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin)

    # compute the intersection over union (handle zero area case)
    union_area = float(box_a_area + box_b_area - inter_area)
    if union_area <= 0:
        return 0.0
    iou = inter_area / union_area
    return iou


def prepare_rcnn_batch(images: List[torch.Tensor], targets: List[List[BBox]],
                       iou_threshold=0.5, neg_sample_iou_max=0.3, max_proposals_per_image=32):
    """
    Transforms a batch of images and ground truth bounding boxes into R-CNN training format.
    
    In a true R-CNN pipeline, 'proposals' would come from Selective Search/RPN.
    Here, we simplify by using Ground Truth boxes as positive proposals
    and randomly sampling regions as negative proposals.
    """
    all_inputs = []
    all_classes = []
    all_deltas = []
    
    device = images[0].device if images else torch.device("cpu")
    
    # Target size is needed for cropping and random negative sampling
    C, H, W = images[0].shape
    
    for img_tensor, gt_bboxes in zip(images, targets):
        
        # 1. Proposals: GT boxes (Positive) + Random Boxes (Negative)
        proposals = []
        proposal_labels = []
        proposal_deltas = []
        
        # --- A. Positive Proposals (Ground Truth Boxes + jittered versions) ---
        for gt_box in gt_bboxes:
            # Original GT box
            proposals.append((gt_box, label2target['pothole'], encode_bbox(gt_box, gt_box)))

            # Add jittered versions of GT boxes (data augmentation for positives)
            for _ in range(2):
                jitter = 0.1
                bw = gt_box.xmax - gt_box.xmin
                bh = gt_box.ymax - gt_box.ymin
                dx = int(random.uniform(-jitter, jitter) * bw)
                dy = int(random.uniform(-jitter, jitter) * bh)
                jittered = BBox(
                    max(0, gt_box.xmin + dx),
                    max(0, gt_box.ymin + dy),
                    min(W, gt_box.xmax + dx),
                    min(H, gt_box.ymax + dy)
                )
                if IoU(jittered, gt_box) > 0.7:
                    proposals.append((jittered, label2target['pothole'], encode_bbox(jittered, gt_box)))

        # --- B. Hard Negative Proposals (partial overlap with GT) ---
        hard_neg_count = min(len(gt_bboxes) * 3, max_proposals_per_image // 3)
        hard_neg_attempts = 0
        hard_negs_added = 0
        while hard_negs_added < hard_neg_count and hard_neg_attempts < hard_neg_count * 10:
            hard_neg_attempts += 1
            if len(gt_bboxes) == 0:
                break
            # Pick a random GT box and create a shifted version
            gt_box = random.choice(gt_bboxes)
            bw = gt_box.xmax - gt_box.xmin
            bh = gt_box.ymax - gt_box.ymin
            shift_x = int(random.uniform(0.3, 0.7) * bw * random.choice([-1, 1]))
            shift_y = int(random.uniform(0.3, 0.7) * bh * random.choice([-1, 1]))

            hard_box = BBox(
                max(0, gt_box.xmin + shift_x),
                max(0, gt_box.ymin + shift_y),
                min(W, gt_box.xmax + shift_x),
                min(H, gt_box.ymax + shift_y)
            )

            # Check IoU - should be between 0.1 and 0.4 (hard negative range)
            max_iou = max(IoU(hard_box, gb) for gb in gt_bboxes)
            if 0.1 < max_iou < 0.4:
                proposals.append((hard_box, label2target['background'], [0.0] * 4))
                hard_negs_added += 1

        # --- C. Easy Negative Proposals (Random Background Sampling) ---
        neg_count = max(0, max_proposals_per_image - len(proposals))
        neg_attempts = 0
        while len(proposals) < max_proposals_per_image and neg_attempts < neg_count * 5:
            neg_attempts += 1
            # Generate a random box with varying sizes
            min_size = 30
            max_size = min(W, H) // 2
            box_size = random.randint(min_size, max_size)
            xmin = random.randint(0, max(1, W - box_size))
            ymin = random.randint(0, max(1, H - box_size))
            xmax = min(W, xmin + box_size)
            ymax = min(H, ymin + box_size)

            rand_box = BBox(xmin, ymin, xmax, ymax)
            
            # Check for max IoU with all ground truth boxes
            max_iou = 0.0
            for gt_box in gt_bboxes:
                max_iou = max(max_iou, IoU(rand_box, gt_box))
            
            # Only use as a negative sample if IoU is very low
            if max_iou <= neg_sample_iou_max:
                proposals.append((rand_box, label2target['background'], [0.0] * 4))

        # 2. Crop and Resize Proposals - only add labels/deltas for valid crops
        batch_crops = []
        batch_labels = []
        batch_deltas = []
        for prop, label, delta in proposals:
            # Clamp coordinates to ensure they are within the image boundaries
            x1 = max(0, prop.xmin)
            y1 = max(0, prop.ymin)
            x2 = min(W, prop.xmax)
            y2 = min(H, prop.ymax)
            
            if x2 > x1 and y2 > y1:
                # Use torchvision.transforms.functional.crop
                cropped_region = TF.crop(img_tensor, y1, x1, y2 - y1, x2 - x1)
                
                # Resize to R-CNN fixed input size (224x224 for VGG)
                resized_crop = F.interpolate(
                    cropped_region.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                batch_crops.append(resized_crop)
                batch_labels.append(label)
                batch_deltas.append(delta)

        # 3. Final Batch Assembly
        if batch_crops:
            all_inputs.append(torch.stack(batch_crops).to(device))
            all_classes.append(torch.tensor(batch_labels).long().to(device))
            all_deltas.append(torch.tensor(batch_deltas).float().to(device))

    # Concatenate all lists into single tensors for the batch
    if not all_inputs:
        # Return dummy empty tensors if the batch is empty (e.g., due to filtering)
        return torch.empty((0, C, 224, 224), device=device), \
               torch.empty((0,), dtype=torch.long, device=device), \
               torch.empty((0, 4), dtype=torch.float, device=device)
               
    final_inputs = torch.cat(all_inputs)
    final_classes = torch.cat(all_classes)
    final_deltas = torch.cat(all_deltas)
    
    return final_inputs, final_classes, final_deltas

def decode_bbox(p_bbox: BBox, t_encoded: List[float]) -> BBox:
    """
    Decodes the predicted transformation (t) back into a final bounding box (g).
    p_bbox is the original proposal box used to generate the prediction.
    """
    px, py, pw, ph = (p_bbox.xmin + p_bbox.xmax) / 2., \
                     (p_bbox.ymin + p_bbox.ymax) / 2., \
                     p_bbox.xmax - p_bbox.xmin, \
                     p_bbox.ymax - p_bbox.ymin
    
    tx, ty, tw, th = t_encoded
    
    # Calculate predicted center and size
    g_x = tx * pw + px
    g_y = ty * ph + py
    g_w = pw * np.exp(tw)
    g_h = ph * np.exp(th)
    
    # Convert center/size back to (xmin, ymin, xmax, ymax)
    g_xmin = int(g_x - g_w / 2.)
    g_ymin = int(g_y - g_h / 2.)
    g_xmax = int(g_x + g_w / 2.)
    g_ymax = int(g_y + g_h / 2.)
    
    return BBox(g_xmin, g_ymin, g_xmax, g_ymax)


def visualize_predictions(model, data_loader, device, num_images=2):
    """
    Fetches a batch, gets predictions, and visualizes them.
    """
    model.eval()
    
    # Get one batch of raw data (normalized image tensor list and BBox list)
    images_list, targets_list = next(iter(data_loader))
    
    for i in range(min(num_images, len(images_list))):
        img_tensor = images_list[i].to(device)
        gt_bboxes = targets_list[i] # Ground Truth Boxes
        
        # 1. Generate Proposals (Same process as validation)
        # We need the proposals and their IoU to decode only positive ones
        
        # For visualization, let's use all original proposals (GT boxes) as 'test' proposals.
        # This is a simplification; a full test involves an RPN.
        test_proposals = [b for b in gt_bboxes]
        
        # Prepare the single image for the R-CNN model
        test_inputs, _, _ = prepare_rcnn_batch([img_tensor], [test_proposals])
        
        if test_inputs.numel() == 0:
            print(f"No test proposals generated for image {i}. Skipping visualization.")
            continue
            
        # 2. Get Model Predictions
        with torch.no_grad():
            cls_scores, pred_deltas = model(test_inputs)
        
        # Predicted Classes (Pothole or Background)
        pred_labels = decode(cls_scores) 
        
        # 3. Decode and Draw Bounding Boxes
        
        # Un-normalize the image tensor for display (using VGG mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        display_img = (img_tensor * std) + mean
        display_img = display_img.permute(1, 2, 0).cpu().numpy().clip(0, 1) # H, W, C
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(display_img)
        ax.set_title(f"Image {i+1} Predictions")
        
        for j, p_bbox in enumerate(test_proposals):
            pred_label = target2label[pred_labels[j].item()]
            
            # We only draw the predicted box if the model classifies it as a 'pothole' (1)
            if pred_labels[j].item() == label2target['pothole']:
                # Decode the predicted deltas using the original proposal box
                final_bbox = decode_bbox(p_bbox, pred_deltas[j].cpu().tolist())
                
                # Draw the predicted box (Red for Pothole)
                rect = patches.Rectangle(
                    (final_bbox.xmin, final_bbox.ymin), 
                    final_bbox.xmax - final_bbox.xmin, 
                    final_bbox.ymax - final_bbox.ymin, 
                    linewidth=2, 
                    edgecolor='r', 
                    facecolor='none',
                    linestyle='-',
                    label=f'Predicted: {pred_label}'
                )
                ax.add_patch(rect)
                
            # Optional: Draw Ground Truth box (Green for comparison)
            gt_rect = patches.Rectangle(
                (p_bbox.xmin, p_bbox.ymin), 
                p_bbox.xmax - p_bbox.xmin, 
                p_bbox.ymax - p_bbox.ymin, 
                linewidth=1, 
                edgecolor='g', 
                facecolor='none',
                linestyle='--',
                label='Ground Truth'
            )
            ax.add_patch(gt_rect)

        plt.show()