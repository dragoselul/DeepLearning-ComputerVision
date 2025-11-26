import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from potholes_dataset import PotholeDataset
import os


# --- Worker Function ---
def process_single_image(data_pack):
    """
    Worker function: Runs Selective Search on one image.
    Returns (found_count, total_gt_boxes)
    """
    img_tensor, gt_boxes, iou_threshold, num_proposals, mean, std = data_pack

    if len(gt_boxes) == 0:
        return 0, 0

    # 1. Un-normalize & Convert to OpenCV
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * std + mean)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 2. Run Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_cv)
    ss.switchToSelectiveSearchQuality()  # Fast mode is usually sufficient for analysis
    rects = ss.process()

    if len(rects) == 0:
        return 0, len(gt_boxes)

    # 3. Filter Proposals
    rects = rects[:num_proposals]

    # Convert xywh -> xmin, ymin, xmax, ymax
    proposals = np.zeros((len(rects), 4))
    proposals[:, 0] = rects[:, 0]
    proposals[:, 1] = rects[:, 1]
    proposals[:, 2] = rects[:, 0] + rects[:, 2]
    proposals[:, 3] = rects[:, 1] + rects[:, 3]

    # 4. Check Overlap (IoU)
    found_count = 0
    for gt in gt_boxes:
        # Vectorized IoU Calculation
        xA = np.maximum(gt[0], proposals[:, 0])
        yA = np.maximum(gt[1], proposals[:, 1])
        xB = np.minimum(gt[2], proposals[:, 2])
        yB = np.minimum(gt[3], proposals[:, 3])

        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        boxAArea = (gt[2] - gt[0]) * (gt[3] - gt[1])
        boxBArea = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])

        iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)

        if np.max(iou) >= iou_threshold:
            found_count += 1

    return found_count, len(gt_boxes)


# --- The Evaluator Class ---
class ProposalEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def evaluate(self, num_proposals=2000, iou_threshold=0.5, workers=10):
        """
        Returns the Recall score (float).
        """
        # Prepare tasks
        tasks = []
        for i in range(len(self.dataset)):
            img, target = self.dataset[i]
            gt_boxes = target['boxes'].numpy()
            tasks.append((img, gt_boxes, iou_threshold, num_proposals, self.mean, self.std))

        # Run Pool
        total_found, total_gt = 0, 0
        with Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap(process_single_image, tasks, chunksize=4),
                                total=len(tasks),
                                desc=f"Evaluating Top-{num_proposals}"))

        for found, total in results:
            total_found += found
            total_gt += total

        recall = total_found / total_gt if total_gt > 0 else 0
        print(f"-> Recall for {num_proposals} proposals: {recall:.2%}")
        return recall

    @staticmethod
    def plot_results(results_dict, save_path='recall_analysis.png'):
        """
        Plots the dictionary {num_proposals: recall_score}.
        """
        sorted_data = sorted(results_dict.items())
        proposals = [x[0] for x in sorted_data]
        recalls = [x[1] * 100 for x in sorted_data]  # to percent

        plt.figure(figsize=(10, 6))
        plt.plot(proposals, recalls, marker='o', linewidth=2, label='Selective Search')

        # Annotate points
        for x, y in zip(proposals, recalls):
            plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.title("Recall vs. Number of Proposals")
        plt.xlabel("Proposals per Image")
        plt.ylabel("Recall (%)")
        plt.grid(True, alpha=0.5)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")


def evaluate_and_plot(path):
    if os.path.exists(ROOT):
        # Load Data
        train_loader, _, _ = PotholeDataset.create_dataloaders(ROOT)
        dataset = train_loader.dataset
        evaluator = ProposalEvaluator(dataset)

        # 1. Define experiments
        proposal_counts = [100, 500, 1000, 2000, 3000]  # Compare these settings
        results = {}

        # 2. Run Loop
        cpu_cores_for_eval = 12
        print(f"Starting evaluation on {len(dataset)} images with {cpu_cores_for_eval} workers...")
        for k in proposal_counts:
            score = evaluator.evaluate(num_proposals=k, workers=cpu_cores_for_eval)
            results[k] = score

        # 3. Plot
        print("\nFinal Results:", results)
        ProposalEvaluator.plot_results(results)

    else:
        print(f"Dataset not found at {ROOT}")

# --- Usage ---
if __name__ == "__main__":
    # Adjust path as needed
    ROOT = os.path.join(os.path.dirname(__file__), "../../", "potholes_processed")
    evaluate_and_plot(ROOT)


