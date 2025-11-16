# Data loader for retinal blood vessel segmentation dataset DRIVE
import os
import glob
import torch
from PIL import Image
from typing import Callable
import torchvision.transforms.functional as TF
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'DRIVE')


class DRIVE(torch.utils.data.Dataset):
    def __init__(self, transform: Callable, split="train", label_transform=None, augment=False):
        'Initialization'
        self.transform = transform
        self.label_transform = label_transform
        self.augment = augment and split == "train"  # Only augment training data

        split_dir = 'training' if split=="train" else 'test'

        data_path = os.path.join(DATA_PATH, split_dir)

        # --- Image Paths ---
        # Adjust extension if your images are not .tif
        IMAGE_EXT = '*.tif'
        image_glob = os.path.join(data_path, 'images', IMAGE_EXT)
        self.image_paths = sorted(glob.glob(image_glob))

        MASK_EXT = '*.gif'
        if split == 'train':
            label_glob = os.path.join(data_path, '1st_manual', MASK_EXT)
        else:
            label_glob = os.path.join(data_path, 'mask', MASK_EXT)


        self.label_paths = sorted(glob.glob(label_glob))

        # --- DEBUG CHECKS (Run these lines outside the class to verify paths) ---
        print(f"--- {split_dir.upper()} SET PATH CHECK ---")
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
                f"Warning: Count mismatch in {split_dir} set: Images={len(self.image_paths)}, Masks={len(self.label_paths)}")

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        # Apply synchronized augmentation BEFORE transforms
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)

            # Color jitter (only for image)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        # Apply image transform (with normalization)
        X = self.transform(image)

        # Apply label transform (without normalization)
        if self.label_transform is not None:
            Y = self.label_transform(label)
        else:
            # Default: just resize and convert to tensor
            Y = TF.resize(label, X.shape[-2:])
            Y = TF.to_tensor(Y)

        # Binarize label: threshold at 0.5 to get 0/1 binary mask
        Y = (Y > 0.5).float()

        return X, Y
