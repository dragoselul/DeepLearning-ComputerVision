import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from tqdm import tqdm
import pandas as pd
import os

class ModelWrapper:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        

    def _train_one_epoch(self, train_loader): 

        self.model.train() 
        running_loss, correct, total = 0.0, 0, 0
        for videos, labels in tqdm(train_loader, desc="Train", leave=False): 
            videos, labels = videos.to(self.device), labels.to(self.device) 
            self.optimizer.zero_grad() 
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            loss.backward() 
            self.optimizer.step() 

            running_loss += loss.item() * videos.size(0) 
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item() 
            total += labels.size(0) 

        return running_loss/total, correct/total 
    

    @torch.no_grad() 
    def _evaluate(self, val_loader): 

        self.model.eval() 
        running_loss, correct, total = 0.0, 0, 0
        for videos, labels in tqdm(val_loader, desc="Val", leave=False):
            videos, labels = videos.to(self.device), labels.to(self.device)
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        return running_loss/total, correct/total


    def train(self, save_path, epochs, train_loader, val_loader):

        train_losses, train_accs = [], []
        val_losses, val_accs = [], []

        best_acc = 0
        for epoch in range(1, epochs + 1):
           
            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._evaluate(val_loader)

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
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model (val acc: {val_acc:.3f})")

        print("Training finished.")
        print(f"Best validation accuracy: {best_acc:.3f}")

        metrics_df = pd.DataFrame({
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs
        })

        return metrics_df
