import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import FrameVideoDataset
import torch.nn.functional as F
import os

#config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_dir = f"{work_dir}\\ucf10".replace("\\", "/")

batch_size = 256
epochs = 50
num_classes = 10
lr = 1e-3 
save_path = f"{work_dir}\\models\\2d_early_fusion.pth".replace("\\", "/")


#training 
def train_one_epoch(model, loader, optimizer, criterion): 
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        imgs = imgs.view(imgs.size(0), -1, imgs.size(-2), imgs.size(-1))
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss/total, correct/total 

@torch.no_grad() 
def evaluate(model, loader, criterion): 
    model.eval() 
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = imgs.view(imgs.size(0), -1, imgs.size(-2), imgs.size(-1))
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


class EarlyFusion2DCNN(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()
        in_channels = 3 * num_frames
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: [batch, num_frames*3, H, W]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":

    #transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    #dataset, loaders
    train_ds = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform)
    val_ds   = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Loaded {len(train_ds)} training frames and {len(val_ds)} validation frames.")


    #model 
    model = EarlyFusion2DCNN()
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
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
    metrics_path = f"{work_dir}/metrics/metrics_2cnn_early_fusion_pf.csv".replace("\\", "/")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
