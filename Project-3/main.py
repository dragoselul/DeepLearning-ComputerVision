import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from time import time

import models as mod
from losses import BCELoss, DiceLoss, FocalLoss, BCELossTotalVariation
import dataset as ds
from train import train
from predict import predict_and_save
from measure import evaluate_dataset

# LOAD DATASET
size = 128
batch_size = 48

train_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])

trainset_drive = ds.DRIVE(split="train", transform=train_transform)
testset_drive = ds.DRIVE(split="test", transform=test_transform)

trainset_ph2 = ds.PH2Dataset(split="train", transform=train_transform)
testset_ph2 = ds.PH2Dataset(split="test", transform=test_transform)

datasets = {
    'drive': (trainset_drive, testset_drive),
    'ph2': (trainset_ph2, testset_ph2)
}

models = {
    'encdec': mod.EncDec,
    'unet': mod.UNet
}

# TRAINING SETUP

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = BCELoss()
learning_rate = 0.001
epochs = 20


# ITERATE OVER MODELS AND DATASETS

for model_name, model_class in models.items():
    print(f"Model: {model_name}")
    model = model_class().to(device)
    opt = optim.AdamW(model.parameters(), learning_rate)

    for dataset_name, (trainset, testset) in datasets.items():

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                num_workers=3)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                num_workers=3)
        
        # START TRAINING
        trained_model = train(model, train_loader, loss_fn, opt, device, save_name=f'{dataset_name}_{model_name}_model.pth', epochs=epochs)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        trained_model_path = os.path.join(script_dir, trained_model)
        predictions_output_dir = os.path.join(script_dir, f'predictions/{dataset_name}_{model_name}')


        # PREDICT ON TEST SET
        predict_and_save(trained_model_path, test_loader, predictions_output_dir, model_type=f'{model_name}', device=device, threshold=0.5)

        gt_files = testset.label_paths

        # EVALUATE PREDICTIONS
        evaluate_dataset(predictions_output_dir, gt_files, dataset_name=dataset_name, model_name=f'{model_name}', pred_pattern='*.png')

