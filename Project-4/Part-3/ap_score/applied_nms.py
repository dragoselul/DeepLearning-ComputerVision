import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np 

from potholes import POTHOLES, BBox
from rcnn import RCNN
from utils import prepare_rcnn_batch, decode_bbox, nms, label2target, IoU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nms(boxes, scores, iou_threshold): 
    """
    Non-Maximum Suppression (NMS) 
    boxes: list of BBox 
    scores: list of floats
    """
    # Converting boxes and scores to arrays
    boxes_array = np.array([[b.xmin, b.ymin, b.xmax, b.ymax] for b in boxes])
    scores_array = np.array(scores)

    order = scores_array.argsort()[::-1] # Order by descending score 
    keep = []

    while len(order) > 0: 
        i = order[0]
        keep.append(i)

        # Compute IoU with all other boxes 
        rest_boxes = order[1:]
        ious = [] 
        for j in rest_boxes: 
            ious.append( IoU (
                BBox(*boxes_array[i]), 
                BBox(*boxes_array[j])
            ))
        ious = np.array(ious) 

        # Keep only boxes with IoU < threshold 
        order = rest_boxes[ious < iou_threshold]

    return keep 

# Load dataset

size = (560, 576)

train_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# label_transform is no longer directly used in __getitem__ but kept for completeness
label_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])


dataset = POTHOLES(split="test", transform=train_transform, label_transform=None)

batch_size = 8
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Load model
rcnn = RCNN().to(device)
rcnn.eval()

# Inference and NMS

# NMS threshold under which BBoxes will be discarded
iou_threshold = 0.5  

for images, targets_list in loader:

    images = [img.to(device) for img in images]

    for img_tensor, gt_bboxes in zip(images, targets_list):

        # prepare proposals 
        proposals = gt_bboxes 
        inputs, _, _ = prepare_rcnn_batch([img_tensor], [proposals])

        if inputs.numel() == 0:
            continue

        # model forward
        cls_scores, pred_deltas = rcnn(inputs)

        # decode predictions
        pred_labels = torch.argmax(cls_scores, dim=1)
        probs = torch.softmax(cls_scores, dim=1)[:, label2target['pothole']].cpu().numpy()

        pothole_preds = []

        for i, lbl in enumerate(pred_labels):
            if lbl.item() == label2target['pothole']:
                pothole_preds.append((proposals[i], pred_deltas[i].cpu(), probs[i]))

        if not pothole_preds:
            continue

        boxes, deltas, scores = zip(*pothole_preds)

        # apply NMS (cf. utils.py, also available here)
        keep_indices = nms(list(boxes), list(scores), iou_threshold=iou_threshold)
        all_indices = set(range(len(boxes)))
        removed_indices = all_indices - set(keep_indices) #kept for illustration purposes

        # visualize final boxes
        fig, ax = plt.subplots(1, figsize=(8, 8))
        # display image
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
        display_img = (img_tensor * std + mean).permute(1,2,0).cpu().numpy().clip(0,1)
        ax.imshow(display_img)

        # draw boxes kept by NMS
        for idx in keep_indices:
            p_bbox = boxes[idx]
            delta = deltas[idx]
            final_bbox = decode_bbox(p_bbox, delta.tolist())
            rect = patches.Rectangle(
                (final_bbox.xmin, final_bbox.ymin),
                final_bbox.xmax - final_bbox.xmin,
                final_bbox.ymax - final_bbox.ymin,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

        # draw boxes discarded by NMS 
        for idx in removed_indices:
            p_bbox = boxes[idx]
            delta = deltas[idx]
            final_bbox = decode_bbox(p_bbox, delta.tolist())
            rect = patches.Rectangle(
                (final_bbox.xmin, final_bbox.ymin),
                final_bbox.xmax - final_bbox.xmin,
                final_bbox.ymax - final_bbox.ymin,
                linewidth=2, edgecolor='b', facecolor='none', linestyle=':'
            )
            ax.add_patch(rect)

        # draw ground truth boxes for reference
        for gt_bbox in gt_bboxes:
            rect = patches.Rectangle(
                (gt_bbox.xmin, gt_bbox.ymin),
                gt_bbox.xmax - gt_bbox.xmin,
                gt_bbox.ymax - gt_bbox.ymin,
                linewidth=1, edgecolor='g', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)

        plt.show()
