# Data loader for retinal blood vessel segmentation dataset DRIVE
import os
import lxml
from lxml import etree
import glob
import torch
from PIL import Image
from typing import Callable
import torchvision.transforms.functional as TF
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'potholes')

@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclass
class DetectedObject:
    name: str
    pose: str
    truncated: int
    occluded: int
    difficult: int
    bbox: BBox


def rotate_bbox(bbox, img_width, img_height, angle_deg):
    """
    Rotate bbox around image center by angle_deg.
    The resulting box is axis-aligned.
    """
    xmin, ymin, xmax, ymax = unpack_bbox(bbox)

    # image center
    cx = img_width / 2
    cy = img_height / 2

    # bbox corner points
    corners = [
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymin),
        (xmax, ymax),
    ]

    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    rotated = []
    for x, y in corners:
        # translate to origin
        x0 = x - cx
        y0 = y - cy

        # rotate
        xr = x0 * cos_t - y0 * sin_t
        yr = x0 * sin_t + y0 * cos_t

        # translate back
        xr += cx
        yr += cy
        rotated.append((xr, yr))

    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]

    new_xmin = min(xs)
    new_xmax = max(xs)
    new_ymin = min(ys)
    new_ymax = max(ys)

    return new_xmin, new_ymin, new_xmax, new_ymax

def unpack_bbox(bbox):
    return bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

class POTHOLES(torch.utils.data.Dataset):
    def __init__(self, transform: Callable, split="train", label_transform=None, augment=False):
        'Initialization'
        self.transform = transform
        self.label_transform = label_transform
        self.augment = augment and split == "train"  # Only augment training data



        # --- Image Paths ---
        # Adjust extension if your images are not .tif
        IMAGE_EXT = '*.png'
        image_glob = os.path.join(DATA_PATH, 'images', IMAGE_EXT)
        self.image_paths = sorted(glob.glob(image_glob))

        MASK_EXT = '*.xml'
        label_glob = os.path.join(DATA_PATH, 'annotations', MASK_EXT)
        self.label_paths = sorted(glob.glob(label_glob))

        # --- DEBUG CHECKS (Run these lines outside the class to verify paths) ---
        print(f"--- {DATA_PATH} SET PATH CHECK ---")
        print(f"Expected Image Glob: {image_glob}")
        print(f"Found Images: {len(self.image_paths)}")
        print(f"Expected Label Glob: {label_glob}")
        print(f"Found Labels: {len(self.label_paths)}")
        if self.image_paths:
            print(f"Example Image Path: {self.image_paths[0]}")
        # ------------------------------------------------------------------------

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found! Check DATA_PATH and file extension. Expected glob: {image_glob}")
        if len(self.image_paths) != len(self.label_paths):
            print(
                f"Warning: Count mismatch in {DATA_PATH} set: Images={len(self.image_paths)}, Masks={len(self.label_paths)}")

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert('RGB')
        tree = ET.parse(label_path)
        root = tree.getroot()

        labels = []

        for obj in root.findall("object"):
            name = obj.findtext("name")
            pose = obj.findtext("pose")
            truncated = int(obj.findtext("truncated"))
            occluded = int(obj.findtext("occluded"))
            difficult = int(obj.findtext("difficult"))
            b = obj.find("bndbox")

            bbox = BBox(
                xmin=int(b.findtext("xmin")),
                ymin=int(b.findtext("ymin")),
                xmax=int(b.findtext("xmax")),
                ymax=int(b.findtext("ymax")),
            )

            labels.append(
                DetectedObject(
                    name=name,
                    pose=pose,
                    truncated=truncated,
                    occluded=occluded,
                    difficult=difficult,
                    bbox=bbox,
                )
            )
            
        image_width, image_height = image.size

        # Apply synchronized augmentation BEFORE transforms
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                for label in labels:
                    xmin, ymin, xmax, ymax = unpack_bbox(label.bbox)
                    new_xmin = image_width - xmax
                    new_xmax = image_width - xmin
                    label.bbox = BBox(new_xmin, ymin, new_xmax, ymax)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                for label in labels:
                    xmin, ymin, xmax, ymax = unpack_bbox(label.bbox)
                    new_ymin = image_height - ymax
                    new_ymax = image_height - ymin
                    label.bbox = BBox(xmin, new_ymin, xmax, new_ymax)

            # Random rotation
            #if random.random() > 0.5:
                #angle = random.uniform(-15, 15)
                #image = TF.rotate(image, angle)
                #for label in labels:
                #    label.bbox = rotate_bbox(label.bbox, image_width, image_height, angle)

            # Color jitter (only for image)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        # Apply image transform (with normalization)
        X = self.transform(image)

        # Apply label transform (without normalization)
        target_size = self.transform.transforms[0].size
        width_scale = target_size[0] / image_width
        height_scale = target_size[1] / image_height
        for label in labels:
            label.bbox.xmin = label.bbox.xmin * width_scale
            label.bbox.xmax = label.bbox.xmax * width_scale
            label.bbox.ymin = label.bbox.ymin * height_scale
            label.bbox.ymax = label.bbox.ymax * height_scale
        Y = labels

        return X, Y
