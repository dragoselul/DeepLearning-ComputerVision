# Data loader for retinal blood vessel segmentation dataset DRIVE
import os
import glob
import torch
from PIL import Image
from typing import Callable

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'DRIVE')


class DRIVE(torch.utils.data.Dataset):
    def __init__(self, transform: Callable, split="train"):
        'Initialization'
        self.transform = transform

        split_dir = 'training' if split=="train" else 'test'

        data_path = os.path.join(DATA_PATH, split_dir)

        # --- Image Paths ---
        # Adjust extension if your images are not .tif
        IMAGE_EXT = '*.tif'
        image_glob = os.path.join(data_path, 'images', IMAGE_EXT)
        self.image_paths = sorted(glob.glob(image_glob))

        # --- Mask Paths ---
        # Adjust extension if your masks are not .gif
        MASK_EXT = '*.gif'
        if split=="train":
            # Training masks are in '1st_manual'
            label_glob = os.path.join(data_path, 'mask', MASK_EXT)
        else:
            # Test masks are in 'mask'
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

        # Apply the transformations
        X = self.transform(image)
        Y = self.transform(label)

        return X, Y
