import math
import random
from dataclasses import dataclass
import glob
import os
import xml.etree.ElementTree as ET
from typing import Callable
from PIL import Image
import torch
import torchvision.transforms.functional as TF

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
    # (Implementation remains the same)
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

    return BBox(new_xmin, new_ymin, new_xmax, new_ymax)

def unpack_bbox(bbox):
    return bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax


script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, '../potholes')

class POTHOLES(torch.utils.data.Dataset):
    def __init__(self, transform: Callable, split="train", label_transform=None, augment=False):
        'Initialization'
        self.transform = transform
        self.label_transform = label_transform
        self.augment = augment and split == "train"

        IMAGE_EXT = '*.png'
        image_glob = os.path.join(DATA_PATH, 'images', IMAGE_EXT)
        self.image_paths = sorted(glob.glob(image_glob))

        MASK_EXT = '*.xml'
        label_glob = os.path.join(DATA_PATH, 'annotations', MASK_EXT)
        self.label_paths = sorted(glob.glob(label_glob))

        if len(self.image_paths) == 0:
            
            print(f"No images found at {image_glob}. Using dummy data path.")
            self.image_paths = ['dummy_path.png']
            self.label_paths = ['dummy_path.xml']
            
    
        if not os.path.exists(DATA_PATH) or len(self.image_paths) == 0:
             print("WARNING: Using dummy image/label paths because real data was not found. Cannot train properly.")
             self.image_paths = ['dummy.png']
             self.label_paths = ['dummy.xml']

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

        try:
            image = Image.open(image_path).convert('RGB')
            tree = ET.parse(label_path)
            root = tree.getroot()
        except FileNotFoundError:
            # Handle dummy data case gracefully if data files are missing
            if 'dummy' in image_path:
                print("Returning dummy data item.")
                image = Image.new('RGB', (560, 576), color = 'red')
                labels = [DetectedObject(name='pothole', pose='Unspecified', truncated=0, occluded=0, difficult=0, bbox=BBox(100, 100, 200, 200))]
            else:
                raise
        except ET.ParseError:
            # Handle XML parsing error
            print(f"Error parsing XML file: {label_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self)) # Load next item
        
        labels = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            pose = obj.findtext("pose")
            truncated = int(obj.findtext("truncated") or 0)
            occluded = int(obj.findtext("occluded") or 0)
            difficult = int(obj.findtext("difficult") or 0)
            b = obj.find("bndbox")

            # Check if bndbox exists and contains all required elements
            if b is not None and all(b.findtext(tag) is not None for tag in ["xmin", "ymin", "xmax", "ymax"]):
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
                    new_xmin = int(image_width - xmax)
                    new_xmax = int(image_width - xmin)
                    label.bbox = BBox(new_xmin, ymin, new_xmax, ymax)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                for label in labels:
                    xmin, ymin, xmax, ymax = unpack_bbox(label.bbox)
                    new_ymin = int(image_height - ymax)
                    new_ymax = int(image_height - ymin)
                    label.bbox = BBox(xmin, new_ymin, xmax, new_ymax)

        # Apply image transform (with normalization)
        X = self.transform(image)

        # Apply label transform (scaling bounding boxes)
        target_size = self.transform.transforms[0].size
        # The image tensor size is [C, H_target, W_target]
        W_target, H_target = target_size 
        width_scale = W_target / image_width
        height_scale = H_target / image_height
        
        # Create a list of BBox objects scaled to the target image size
        scaled_bboxes = []
        for label in labels:
            scaled_bboxes.append(
                BBox(
                    xmin=int(label.bbox.xmin * width_scale),
                    ymin=int(label.bbox.ymin * height_scale),
                    xmax=int(label.bbox.xmax * width_scale),
                    ymax=int(label.bbox.ymax * height_scale),
                )
            )

        # Return the transformed image tensor and a list of SCALED BBox objects
        return X, scaled_bboxes