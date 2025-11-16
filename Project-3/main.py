import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from time import time

import models as mod
from losses import *
import dataset as ds
from train import train
from predict import predict_and_save
from measure import evaluate_dataset

# LOAD DATASET
size = (560, 576)  # Near original DRIVE size (565x584), divisible by 16
batch_size = 2  # Small batch for large images to fit in GPU memory

train_transform_drive = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_drive = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_transform_drive = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])

size = (256,256)

train_transform_ph2 = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_ph2 = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_transform_ph2 = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])

from torch.utils.data import random_split

# Load datasets
full_trainset_drive = ds.DRIVE(split="train", transform=train_transform_drive, label_transform=label_transform_drive, augment=True)
testset_drive = ds.DRIVE(split="test", transform=test_transform_drive, label_transform=label_transform_drive, augment=False)

full_trainset_ph2 = ds.PH2Dataset(split="train", transform=train_transform_ph2, label_transform=label_transform_ph2, augment=True)
testset_ph2 = ds.PH2Dataset(split="test", transform=test_transform_ph2, label_transform=label_transform_ph2, augment=False)

# Create validation splits (80% train, 20% validation)
def create_train_val_split(dataset, val_ratio=0.2):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    return random_split(dataset, [train_size, val_size])

trainset_drive, valset_drive = create_train_val_split(full_trainset_drive)
trainset_ph2, valset_ph2 = create_train_val_split(full_trainset_ph2)

print(f"DRIVE: {len(trainset_drive)} train, {len(valset_drive)} val, {len(testset_drive)} test")
print(f"PH2: {len(trainset_ph2)} train, {len(valset_ph2)} val, {len(testset_ph2)} test")

datasets = {
    'drive': (trainset_drive, valset_drive, testset_drive),
    'ph2': (trainset_ph2, valset_ph2, testset_ph2)
}

models = {
    'encdec': mod.EncDec,
    'unet': mod.UNet
}

# TRAINING SETUP

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = DiceBCELoss(
    dice_weight=0.7,
    bce_weight=0.3,
    pos_weight=torch.tensor([15.0]).to(device)
)
learning_rate = 1e-4  # Slightly higher initial LR
epochs = 100  # More epochs for better convergence


# ITERATE OVER MODELS AND DATASETS

for model_name, model_class in models.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    for dataset_name, (trainset, valset, testset) in datasets.items():
        print(f"\nDataset: {dataset_name}")

        model = model_class().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                num_workers=3)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                                num_workers=3)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                num_workers=3)
        
        # START TRAINING with early stopping
        trained_model = train(model, train_loader, loss_fn, opt, device,
                            save_name=f'{dataset_name}_{model_name}_model.pth',
                            epochs=epochs,
                            val_loader=val_loader,
                            patience=10,
                            min_delta=0.001)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        trained_model_path = os.path.join(script_dir, trained_model)
        predictions_output_dir = os.path.join(script_dir, f'predictions/{dataset_name}_{model_name}')


        # PREDICT ON TEST SET
        predict_and_save(trained_model_path, test_loader, predictions_output_dir, model_type=f'{model_name}', device=device, threshold=0.5)

        gt_files = testset.label_paths

        # EVALUATE PREDICTIONS
        evaluate_dataset(predictions_output_dir, gt_files, dataset_name=dataset_name, model_name=f'{model_name}', pred_pattern='*.png')

