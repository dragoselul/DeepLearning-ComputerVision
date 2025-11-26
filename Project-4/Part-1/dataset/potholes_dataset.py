import os
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PotholeDataset(Dataset):
    def __init__(self, image_files, root_dir, transform=None):
        self.image_files = image_files
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = {'pothole': 1}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, "images", img_name)
        xml_path = os.path.join(self.root_dir, "annotations", os.path.splitext(img_name)[0] + ".xml")

        # 1. Load Image & Boxes
        image = np.array(Image.open(img_path).convert("RGB"))
        boxes, labels = self._load_annotations(xml_path)

        # 2. Apply Transforms (with empty box handling)
        if self.transform:
            # Albumentations requires at least one box. If empty, add a dummy (0,0,1,1) with label 0
            if len(boxes) == 0:
                boxes = [[0.0, 0.0, 1.0, 1.0]]
                labels = [0]

            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)

            # Remove dummy box if it exists
            if len(boxes) == 1 and labels[0] == 0:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)

        # 3. Format for PyTorch Detection Models
        # shape [N, 4] and [N]
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([0.]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }

        return image, target

    def _load_annotations(self, xml_path):
        boxes, labels = [], []
        if not os.path.exists(xml_path):
            return boxes, labels

        tree = ET.parse(xml_path)
        for obj in tree.findall("object"):
            name = obj.find("name").text
            if name in self.class_map:
                b = obj.find("bndbox")
                xmin = float(b.find("xmin").text)
                ymin = float(b.find("ymin").text)
                xmax = float(b.find("xmax").text)
                ymax = float(b.find("ymax").text)

                # --- SAFETY CHECK START ---
                # 1. Ensure coordinates are valid (min < max)
                # 2. Ensure box has at least 1 pixel width/height
                if xmax <= xmin + 1 or ymax <= ymin + 1:
                    print(f"Warning: Ignored invalid box in {xml_path}: {[xmin, ymin, xmax, ymax]}")
                    continue
                # --- SAFETY CHECK END ---

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_map[name])
        return boxes, labels

    @staticmethod
    def collate_fn(batch):
        """Tuple of zipped batch (required for detection)."""
        return tuple(zip(*batch))

    @staticmethod
    def create_dataloaders(root_dir, batch_size=4, split=(0.7, 0.15, 0.15)):
        """Creates Train/Val/Test loaders."""
        img_dir = os.path.join(root_dir, "images")
        files = [f for f in os.listdir(img_dir)]

        random.seed(42)
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * split[0])
        n_val = int(n_total * split[1])

        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:]
        }

        # Standard transform: Resize -> Tensor
        tfm = A.Compose([
            A.Resize(500, 500),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        loaders = {}
        for phase, file_list in splits.items():
            ds = PotholeDataset(file_list, root_dir, transform=tfm)
            loaders[phase] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(phase == 'train'),
                collate_fn=PotholeDataset.collate_fn,
                num_workers=2  # Safe to use now
            )

        print(f"Dataset Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        return loaders['train'], loaders['val'], loaders['test']


# --- Usage ---
if __name__ == "__main__":
    # Adjust path to your actual dataset location
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", "potholes")

    if os.path.exists(ROOT):
        train_loader, _, _ = PotholeDataset.create_dataloaders(ROOT)
        img, target = next(iter(train_loader))
        print(f"Loaded batch of {len(img)} images")
        print(f"Image dimensions: {img[0].shape}")
    else:
        print(f"Dataset not found at {ROOT}. Please check path.")
