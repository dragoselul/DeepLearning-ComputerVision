import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T 
from torchvision.models.video import r3d_18
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import FrameVideoDataset
import os

#config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/dragoselul/git/DeepLearning-ComputerVision/Project-2/ucf101_noleakage"
work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

batch_size = 64 #careful with the RAM
epochs = 50
num_classes = 10
lr = 1e-3 
save_path = "C:/Users/Elio/Documents/ecole/S9/ComputerVision/project2/models/r3d18_video_baseline.pth"


#training 
def train_one_epoch(model, loader, optimizer, criterion): 
    model.train() 
    running_loss, correct, total = 0.0, 0, 0
    for videos, labels in tqdm(loader, desc="Train", leave=False): 
        videos, labels = videos.to(device), labels.to(device) 
        optimizer.zero_grad() 
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() * videos.size(0) 
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item() 
        total += labels.size(0) 

    return running_loss/total, correct/total 

@torch.no_grad() 
def evaluate(model, loader, criterion): 
    model.eval() 
    running_loss, correct, total = 0.0, 0, 0
    for videos, labels in tqdm(loader, desc="Val", leave=False):
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * videos.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss/total, correct/total


if __name__ == "__main__":

    #transforms
    transform = T.Compose([
        T.Resize((64, 64)), #frame size
        T.ToTensor(),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    #dataset, loaders
    train_ds = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    val_ds   = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Loaded {len(train_ds)} training videos and {len(val_ds)} validation frames.")


    #model 
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # torch.save(model.state_dict(), save_path)
            print(f"Saved best model (val acc: {val_acc:.3f})")

    print("Training finished.")
    print(f"Best validation accuracy: {best_acc:.3f}")

    metrics_df = pd.DataFrame({
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs
    })

    #save to csv
    metrics_path = f"{work_dir}/metrics/metrics_3cnn_aggreg_pf_noleakage.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
