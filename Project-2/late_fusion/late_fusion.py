import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from torchvision.models.video import r3d_18
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import FrameVideoDataset 
import os
from cnn_utils import ModelWrapper
from collections import OrderedDict

class ThreeDLateFusion(nn.Module):
    def __init__(self, backbone_3d, num_classes, aggregation_mode='average'):
        super().__init__()
     
        self.feature_extractor = nn.Sequential(*list(backbone_3d.children())[:-1]) 
        self.feature_dim = backbone_3d.fc.in_features 
        
        if aggregation_mode == 'average':
            # This aggregates features across the clips (T_clips dimension)
            self.aggregate = lambda x: torch.mean(x, dim=1) 
        elif aggregation_mode == 'max':
            self.aggregate = lambda x: torch.max(x, dim=1)[0]
        else:
            raise ValueError("Unsupported aggregation mode.")

        # 3. Final classifier (FC layer)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, clip_length=8):
        
        B, C, T_full, H, W = x.shape
        
        T_new = (T_full // clip_length) * clip_length 
         # Truncate the tensor along the Time dimension (index 2)
        if T_full != T_new:
             # Truncate the video tensor
             x = x[:, :, :T_new, :, :] 
             T_full = T_new # Update T_full

        T_clips = T_full // clip_length
        
        x = x.transpose(1, 2) # (B, T_full, C, H, W)
        x = x.contiguous().view(B * T_clips, clip_length, C, H, W)
        x = x.transpose(1, 2) # Restore to (B * T_clips, C, clip_length, H, W)
        features = self.feature_extractor(x).squeeze()
        if features.dim() == 1:
            features = features.unsqueeze(0)

        features = features.view(B, T_clips, self.feature_dim) 
        aggregated_features = self.aggregate(features) 
        output = self.classifier(aggregated_features)  
        return output

# --- Model Definition for Late Fusion ---
class ResNetLateFusion(nn.Module):
    def __init__(self, backbone, num_classes, aggregation_mode='average'):
        super().__init__()
     
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1]) 

        self.feature_dim = backbone.fc.in_features 
        
        # 2. Aggregation mode
        if aggregation_mode == 'average':
            self.aggregate = lambda x: torch.mean(x, dim=1)
        elif aggregation_mode == 'max':
            self.aggregate = lambda x: torch.max(x, dim=1)[0]
        else:
            raise ValueError("Unsupported aggregation mode.")

        # 3. Final classifier (FC layer)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) 
        
        features = self.feature_extractor(x)  
        
        features = features.view(B * T, self.feature_dim) 
        
        features = features.view(B, T, self.feature_dim) 
        
        aggregated_features = self.aggregate(features) 
 
        output = self.classifier(aggregated_features)  
        return output
# -----------------------------------------------------------------

#config 
current_dir = os.path.dirname(os.path.abspath(__file__))
# Parent directory
project_dir = os.path.dirname(current_dir)

# Paths relative to project root
save_path = os.path.join(project_dir, "checkpoints", "late_fusion.pth")
metrics_path_2d = os.path.join(project_dir, "metrics", "late_fusion_2d_metrics.csv")
metrics_path_3d = os.path.join(project_dir, "metrics", "late_fusion_3d_metrics.csv")

# Ensure the directories exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)
os.makedirs(os.path.dirname(metrics_path_2d), exist_ok=True)
os.makedirs(os.path.dirname(metrics_path_3d), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/home/dragoselul/git/DeepLearning-ComputerVision/Project-2/ufc10"



batch_size = 64
epochs = 50
num_classes = 10
lr = 1e-3 


if __name__ == "__main__":

    #transforms
    transform = T.Compose([
        T.Resize((112, 112)), # Use slightly larger size for better ResNet features
        T.ToTensor(),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

   
    train_ds = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True) 
    val_ds   = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)

   
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Loaded {len(train_ds)} training videos and {len(val_ds)} validation videos.")


    #model 
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    r3d_18 = r3d_18(weights=None)
    # Instantiate the Late Fusion Model
    # model = ThreeDLateFusion(backbone_3d=r3d_18, num_classes=num_classes, aggregation_mode='average')
    model = ResNetLateFusion(backbone=resnet18, num_classes=num_classes, aggregation_mode='average')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    modelWrapper = ModelWrapper(model, optimizer, criterion, device)
    metrics_df = modelWrapper.train(save_path, epochs, train_loader, val_loader)

    #save to csv
    metrics_path = metrics_path_2d
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")