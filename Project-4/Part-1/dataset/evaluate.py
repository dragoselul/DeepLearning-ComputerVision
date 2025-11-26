import torch
import cv2
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from potholes_dataset import PotholeDataset


# --- Worker Function ---
def extract_and_label(data_pack):
    """
    Runs SS, generates 2000 boxes, and labels them based on GT IoU.
    Returns: (filename, proposals, labels)
    """
    filename, img_tensor, gt_boxes, num_proposals, iou_thresh, mean, std = data_pack

    # 1. Prepare Image for OpenCV
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * std + mean)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 2. Run Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_cv)
    ss.switchToSelectiveSearchQuality()  # Stick to Fast as decided
    rects = ss.process()

    # 3. Process Proposals
    if len(rects) == 0:
        # Fallback if SS fails completely (rare)
        proposals = np.zeros((0, 4), dtype=np.float32)
    else:
        rects = rects[:num_proposals]
        # Convert (x, y, w, h) -> (xmin, ymin, xmax, ymax)
        proposals = np.zeros((len(rects), 4), dtype=np.float32)
        proposals[:, 0] = rects[:, 0]
        proposals[:, 1] = rects[:, 1]
        proposals[:, 2] = rects[:, 0] + rects[:, 2]
        proposals[:, 3] = rects[:, 1] + rects[:, 3]

    # 4. Assign Labels (Vectorized)
    # Default everything to Background (0)
    labels = np.zeros(len(proposals), dtype=np.int64)

    if len(gt_boxes) > 0 and len(proposals) > 0:
        # Calculate IoU for every proposal against every GT box
        # Expand dims to broadcast: (N, 1, 4) vs (1, M, 4)
        p = proposals[:, np.newaxis, :]
        g = gt_boxes[np.newaxis, :, :]

        xA = np.maximum(p[..., 0], g[..., 0])
        yA = np.maximum(p[..., 1], g[..., 1])
        xB = np.minimum(p[..., 2], g[..., 2])
        yB = np.minimum(p[..., 3], g[..., 3])

        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        pArea = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
        gArea = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])

        iou = interArea / (pArea + gArea - interArea + 1e-6)

        # Get max IoU for each proposal (did it hit ANY pothole?)
        max_ious = np.max(iou, axis=1)

        # Label 1 if IoU >= Threshold (usually 0.5)
        labels[max_ious >= iou_thresh] = 1

    return filename, proposals, labels


def generate_training_data(dataset, output_path, num_proposals=2000, workers=10):
    print(f"Generating training data from {len(dataset)} images...")
    print(f"Settings: Top-{num_proposals} proposals, IoU Threshold 0.5")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Prepare tasks
    tasks = []
    for i in range(len(dataset)):
        img, target = dataset[i]
        fname = dataset.image_files[i]  # Assuming dataset has this attribute
        gt = target['boxes'].numpy()
        tasks.append((fname, img, gt, num_proposals, 0.5, mean, std))

    # Run Pool
    saved_data = []
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(extract_and_label, tasks, chunksize=4), total=len(tasks)))

    # Structure the data for saving
    pos_count = 0
    neg_count = 0

    for fname, props, labs in results:
        saved_data.append({
            'filename': fname,
            'proposals': torch.tensor(props, dtype=torch.float32),
            'labels': torch.tensor(labs, dtype=torch.long)
        })
        pos_count += np.sum(labs == 1)
        neg_count += np.sum(labs == 0)

    # Save to disk
    print(f"\nSaving {len(saved_data)} entries to {output_path}...")
    torch.save(saved_data, output_path)

    print("--- Statistics ---")
    print(f"Total Proposals: {pos_count + neg_count}")
    print(f"Positives (Potholes): {pos_count}")
    print(f"Negatives (Background): {neg_count}")
    print(f"Class Imbalance: 1 positive for every {neg_count / pos_count:.1f} negatives")


if __name__ == "__main__":
    # Paths
    ROOT = os.path.join(os.path.dirname(__file__), "../../", "potholes_processed")
    SAVE_PATH = "proposals_train_3k.pt"

    if os.path.exists(ROOT):
        # Load only the TRAIN split
        train_loader, _, _ = PotholeDataset.create_dataloaders(ROOT)
        train_ds = train_loader.dataset

        generate_training_data(train_ds, SAVE_PATH, num_proposals=3000, workers=10)
