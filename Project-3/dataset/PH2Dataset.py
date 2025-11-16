import os
import glob
import torch
import random
from PIL import Image
from typing import Callable, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'PH2_Dataset_images')

# Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Subfolder/Filename structure (based on your image)
IMAGE_SUBFOLDER = 'Dermoscopic_Image'
MASK_SUBFOLDER = 'lesion' 
IMAGE_FILENAME = '*.bmp' 
# ---------------------

# --- Global Cache for Data Paths ---
# This dictionary will store the splits once they are calculated.
_PH2_DATA_SPLITS = {
    'train': None,
    'val': None,
    'test': None
}

def get_all_image_paths(data_root: str) -> List[str]:
    """
    Collects the full path for all image files in the dataset.
    """
    patient_dirs = sorted(glob.glob(os.path.join(data_root, 'IMD*')))
    
    image_paths = []
    for p_dir in patient_dirs:
        # e.g., /.../IMD002/IMD002_Dermoscopic_Image/IMD002.bmp
        image_glob = os.path.join(p_dir, f'*{IMAGE_SUBFOLDER}', IMAGE_FILENAME)
        found_images = glob.glob(image_glob)
        image_paths.extend(found_images)
        
    return sorted(image_paths)

def split_data_paths(all_image_paths: List[str], train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits the list of image paths into train, validation, and test sets.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum up to 1.0")

    # Ensure reproducibility
    random.seed(42)
    random.shuffle(all_image_paths)
    
    total_count = len(all_image_paths)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    train_paths = all_image_paths[:train_count]
    val_paths = all_image_paths[train_count:train_count + val_count]
    test_paths = all_image_paths[train_count + val_count:]
    
    return train_paths, val_paths, test_paths

# --------------------------------------------------------------------------
# --- MODIFIED PH2Dataset CLASS ---
# --------------------------------------------------------------------------

class PH2Dataset(Dataset):
    """
    Dataset class for the PH2 skin lesion dataset, handling internal splits.
    """
    def __init__(self, split: Union[str, bool], transform: Callable, label_transform=None, augment=False):
        'Initialization'
        self.transform = transform
        self.label_transform = label_transform

        # --- Determine the required split ---
        if isinstance(split, str):
            split_key = split.lower()
        elif isinstance(split, bool):
            # Backwards compatibility/standard train/test
            split_key = 'train' if split else 'test'
        else:
            raise ValueError("The 'split' argument must be a string ('train', 'val', 'test') or a boolean.")

        self.augment = augment and split_key == 'train'  # Only augment training data

        if split_key not in _PH2_DATA_SPLITS:
            raise ValueError(f"Invalid split key: {split_key}. Must be 'train', 'val', or 'test'.")

        # --- Perform Split and Cache if not already done (lazy loading) ---
        if _PH2_DATA_SPLITS['train'] is None:
            print("--- PH2 Data Split Initialization ---")
            all_paths = get_all_image_paths(DATA_PATH)
            
            if not all_paths:
                raise FileNotFoundError(f"No image files found in {DATA_PATH}. Check DATA_PATH.")

            train_paths, val_paths, test_paths = split_data_paths(
                all_paths, 
                TRAIN_RATIO, 
                VAL_RATIO, 
                TEST_RATIO
            )
            
            _PH2_DATA_SPLITS['train'] = train_paths
            _PH2_DATA_SPLITS['val'] = val_paths
            _PH2_DATA_SPLITS['test'] = test_paths
            print(f"Dataset split completed. Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
            print("-------------------------------------")

        # --- Assign paths for this instance ---
        self.image_paths = _PH2_DATA_SPLITS[split_key]
        
        # Determine mask paths from image paths
        self.label_paths = self._get_mask_paths(self.image_paths)

        if len(self.image_paths) != len(self.label_paths):
            raise RuntimeError(f"Count mismatch in {split_key} set: Images={len(self.image_paths)}, Masks={len(self.label_paths)}")
            
    def _get_mask_paths(self, image_paths: List[str]) -> List[str]:
        """
        Derives the corresponding mask path for each image path.
        """
        mask_paths = []
        for img_path in image_paths:
            # 1. Get the patient folder path (e.g., /.../IMD002)
            patient_dir = os.path.dirname(os.path.dirname(img_path))
            # 2. Extract the base image ID (e.g., 'IMD002')
            base_id = os.path.basename(patient_dir) 
            
            # 3. Construct the expected mask path
            # Expected structure: PATIENT_DIR / {ID}_lesion / {ID}_lesion.bmp
            mask_filename = f'{base_id}_lesion.bmp'
            mask_path = os.path.join(patient_dir, f'{base_id}_{MASK_SUBFOLDER}', mask_filename)
            
            mask_paths.append(mask_path)
            
        return mask_paths

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
            import torchvision.transforms.functional as TF
            import random

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
            import torchvision.transforms.functional as TF
            Y = TF.resize(label, X.shape[-2:])
            Y = TF.to_tensor(Y)

        # Binarize label: threshold at 0.5 to get 0/1 binary mask
        Y = (Y > 0.5).float()

        return X, Y