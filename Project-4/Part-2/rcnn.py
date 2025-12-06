
import torch
from dataclasses import dataclass
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.models as models
from torch_snippets import Report
import numpy as np
from potholes import POTHOLES

from utils import prepare_rcnn_batch, decode, label2target, visualize_predictions


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_backbone = models.vgg16(weights=True)
vgg_backbone.classifier = nn.Sequential(*list(vgg_backbone.classifier.children())[:-1]) # Removing the final FC layer (25088 -> 4096)
# Correct feature dim is 4096 after removing the last layer, not 25088 (which is the output of the flatten layer)
feature_dim = 4096 
for param in vgg_backbone.parameters():
    param.requires_grad = False
vgg_backbone.eval().to(device)


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # feature_dim is 4096 from the VGG16 backbone's penultimate layer
        self.backbone = vgg_backbone
        self.cls_score = nn.Linear(feature_dim, len(label2target))
        self.bbox = nn.Sequential(
              nn.Linear(feature_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 4),
              nn.Tanh(),
            )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()

    def forward(self, input):
        feat = self.backbone(input)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox
    
    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        
        # Only calculate regression loss for non-background labels (label != 0)
        ixs, = torch.where(labels != label2target['background'])
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        
        if len(ixs) > 0:
            # Ensure deltas are normalized between -1 and 1 as per Tanh output
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            # Handle case where all proposals are background
            regression_loss = torch.tensor(0.0, device=probs.device)
            return detection_loss, detection_loss.detach(), regression_loss.detach()


def train_batch(inputs, model, optimizer, criterion):
    # inputs is now (input_crops, clss_labels, delta_targets)
    input, clss, deltas = inputs
    
    # Skip batch if no samples were generated (e.g., all samples filtered out)
    if input.numel() == 0:
        return torch.tensor(0.0, device=device), 0.0, 0.0, np.array([True])

    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    
    # Check for NaNs before backprop
    if torch.isnan(loss):
        print("Warning: Loss is NaN. Skipping backward pass.")
        return torch.tensor(0.0, device=device), loc_loss.item(), regr_loss.item(), np.array([True])
        
    accs = clss == decode(_clss)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()

@torch.no_grad()
def validate_batch(inputs, model, criterion):
    # inputs is now (input_crops, clss_labels, delta_targets)
    input, clss, deltas = inputs
    
    if input.numel() == 0:
        return None, None, torch.tensor(0.0, device=device), 0.0, 0.0, np.array([True])

    with torch.no_grad():
        model.eval()
        _clss_score, _deltas = model(input)
        loss, loc_loss, regr_loss = criterion(_clss_score, _deltas, clss, deltas)
        
        _, _clss = _clss_score.max(-1)
        accs = clss == _clss
        
    return _clss, _deltas, loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()




size = (560, 576)

train_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# label_transform is no longer directly used in __getitem__ but kept for completeness
label_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])


potholes = POTHOLES(split="train", transform=train_transform, label_transform=label_transform, augment=True)

n = len(potholes)
train_size = int(0.7 * n)
val_size   = int(0.15 * n)
test_size  = n - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    potholes,
    [train_size, val_size, test_size]
)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets

# Reduced batch size to avoid GPU memory overflow (each image generates multiple proposals)
batch_size = 8
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


rcnn = RCNN().to(device)
criterion = rcnn.calc_loss
optimizer = torch.optim.SGD(rcnn.parameters(), lr=1e-3)
n_epochs = 5
log = Report(n_epochs)

print("Starting Training...")
print(f"Total dataset size: {n}")
print(f"Training on device: {device}")

for epoch in range(n_epochs):
    print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
    
    # 1. Training Phase
    _n = len(train_loader)
    for ix, (images, targets) in enumerate(train_loader):
        # Move images to device before preparing the batch
        images = [img.to(device) for img in images]
        # NEW STEP: Prepare the batch for R-CNN training
        inputs = prepare_rcnn_batch(images, targets)
        
        # Check if the batch generation was successful
        if inputs[0].numel() == 0:
            print(f"Skipping training batch {ix}/{_n}: No valid proposals generated.")
            continue
            
        loss, loc_loss, regr_loss, accs = train_batch(inputs, rcnn, 
                                                      optimizer, criterion)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss, 
                   trn_regr_loss=regr_loss, 
                   trn_acc=accs.mean(), end='\r')
        
    # 2. Validation Phase
    _n = len(test_loader)
    for ix, (images, targets) in enumerate(test_loader):
        # Move images to device before preparing the batch
        images = [img.to(device) for img in images]
        # NEW STEP: Prepare the batch for R-CNN validation
        inputs = prepare_rcnn_batch(images, targets)
        
        if inputs[0].numel() == 0:
            print(f"Skipping validation batch {ix}/{_n}: No valid proposals generated.")
            continue
            
        _clss, _deltas, loss, \
        loc_loss, regr_loss, accs = validate_batch(inputs, 
                                                rcnn, criterion)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss, 
                val_regr_loss=regr_loss, 
                val_acc=accs.mean(), end='\r')
        

visualize_predictions(rcnn, test_loader, device, num_images=2)