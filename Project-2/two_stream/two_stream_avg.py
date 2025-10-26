from glob import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision import models
import torch.nn as nn
from glob import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

import random


class TwoStreamDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='./ufc10', split='train', transform=None, flow_transform=None, stack_flows=True):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.n_sampled_frames = 10
        self.transform = transform
        self.stack_flows = stack_flows
        self.flow_transform = flow_transform
    
    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        #We need only a single frame and flows for the two-stream
        frames = self.get_single_frame(idx)
        flows_npy = self.get_flows(idx)

        #Flip augmentation, but must be appiled for pairs
        do_flip = torch.rand(()) < 0.5
        if do_flip:
            frames = torch.flip(frames, dims=[2])

            if self.stack_flows:
                flows_npy = torch.flip(flows_npy, dims=[2])
                flows_npy[0::2] = -flows_npy[0::2]
            else:
                flows_npy = [torch.flip(f, dims=[2]) for f in flows_npy]
                for f in flows_npy:
                    f[0] = -f[0]
                    
        return {
            'frames': frames,
            'flows_npy': flows_npy,
            'label': label,
            'video_name': video_name
        }

    def get_flows(self, idx):
        flow_dir = self.video_paths[idx].replace('videos', 'flows').replace('.avi', '')
        flows = self.load_flows(flow_dir) #List of (2,H,W) flows
        if self.flow_transform:
            flows = [self.flow_transform(f) for f in flows]

        # Stack for 2D
        if self.stack_flows:
            flows = torch.stack(flows) # (T-1, 2, H, W)
            flows = flows.flatten(0, 1) # (2*(T-1), H, W)
        return flows

    def load_flows(self, frames_dir):
        flows = []
        for i in range(1, self.n_sampled_frames):
            fpath = os.path.join(frames_dir, f"flow_{i}_{i+1}.npy")
            arr = np.load(fpath).astype(np.float32)
            flows.append(torch.from_numpy(arr).contiguous())
        return flows

    def get_single_frame(self, idx):
        #Random frame
        frame_dir = self.video_paths[idx].replace('videos','frames').replace('.avi','')
        n = self.n_sampled_frames
        k = random.randint(1, n) if self.split == 'train' else (n + 1)//2
        img = Image.open(os.path.join(frame_dir, f"frame_{k}.jpg")).convert("RGB")
        return self.transform(img) if self.transform else T.ToTensor()(img)
    
#Backbone, you can change in_ch as they are different for each stream
def make_vgg16(num_classes, in_ch=3):
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    if in_ch != 3:
        old = m.features[0]
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding)
        m.features[0] = new
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m

class TwoStreamFusion(nn.Module):
    def __init__(self, spatial, temporal, num_classes, fusion='weighted'):
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal
        self.fusion = fusion
        if fusion == 'mlp':
            self.fuse_head = nn.Sequential(
                nn.Linear(2 * num_classes, num_classes)
            )
        elif fusion == 'weighted':
            # learnable positive weights; start equal
            self.w_rgb  = nn.Parameter(torch.tensor(0.0))  # softplus -> ~1.0
            self.w_flow = nn.Parameter(torch.tensor(0.0))
        else:
            self.fuse_head = None

    def forward(self, frame, flows):
        logit_rgb  = self.spatial(frame)     # (B, C)
        logit_flow = self.temporal(flows)    # (B, C)

        if self.fusion == 'avg':
            return (logit_rgb + logit_flow) / 2
        elif self.fusion == 'logitsum':
            return logit_rgb + logit_flow
        elif self.fusion == 'mlp':
            x = torch.cat([logit_rgb, logit_flow], dim=1)
            return self.fuse_head(x)
        elif self.fusion == 'weighted':
            # positive weights, normalized to sum=1
            w1 = torch.nn.functional.softplus(self.w_rgb)
            w2 = torch.nn.functional.softplus(self.w_flow)
            s  = (w1 + w2).clamp_min(1e-6)
            return (w1/s) * logit_rgb + (w2/s) * logit_flow
        else:
            raise ValueError(self.fusion)
        
frame_tf = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def flow_tf(flow, out_size=(224,224), clip=20.0):
    H0,W0 = flow.shape[1:]
    if (H0,W0) != out_size:
        flow = F.interpolate(flow.unsqueeze(0), size=out_size, mode='bilinear',
                             align_corners=False).squeeze(0)
        sx, sy = out_size[1]/W0, out_size[0]/H0
        flow[0] *= sx; flow[1] *= sy
    return torch.clamp(flow, -clip, clip) / clip

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for b in loader:
        x_rgb = b['frame'].to(device, non_blocking=True)
        x_flow = b['flows'].to(device, non_blocking=True)
        y = b['label'].to(device, non_blocking=True).long()
        logits = model(x_rgb, x_flow)
        loss_sum += ce(logits, y).item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum/total, correct/total

def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for b in loader:
        x_rgb  = b['frame'].to(device, non_blocking=True)
        x_flow = b['flows'].to(device, non_blocking=True)
        y      = b['label'].to(device, non_blocking=True).long()

        opt.zero_grad(set_to_none=True)
        logits = model(x_rgb, x_flow)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # accumulate metrics
        bs = y.size(0)
        running_loss += loss.item() * bs
        correct      += (logits.argmax(1) == y).sum().item()
        total        += bs

    epoch_loss = running_loss / max(total, 1)
    epoch_acc  = correct / max(total, 1)
    return epoch_loss, epoch_acc

#Combine a row from each to make a training example
def collate_two_stream(batch):
    frames = torch.stack([b['frames'] for b in batch]) # (B, 3, 224, 224)
    flows  = torch.stack([b['flows_npy'] for b in batch]) # (B, 2*(T-1), 224, 224)
    labels = torch.tensor([b['label'] for b in batch])
    names  = [b['video_name'] for b in batch]
    return {'frame': frames, 'flows': flows, 'label': labels, 'video_name': names}
    
if __name__ == '__main__':
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'
    train_dataset = TwoStreamDataset(root_dir, split='train', transform=frame_tf, flow_transform=flow_tf, stack_flows=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_two_stream)


    val_dataset   = TwoStreamDataset(root_dir, split='val', transform=frame_tf, flow_transform=flow_tf, stack_flows=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_two_stream)

    test_dataset = TwoStreamDataset(root_dir, split='test',transform=frame_tf,flow_transform=flow_tf,stack_flows=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_two_stream)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Frames = 10
    num_classes = 10
    spatial_model = make_vgg16(num_classes, in_ch=3)
    temporal_model = make_vgg16(num_classes, in_ch=2*(Frames-1))
    model = TwoStreamFusion(spatial_model, temporal_model, num_classes=num_classes, fusion='avg').to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.5e-4)
    EPOCHS = 50
    best = {'acc': 0.0, 'state': None}
    metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in range(1, EPOCHS + 1):
        # ---- train
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device, criterion)

        # ---- validate
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"[Two-stream][epoch {epoch:02d}] train_loss={train_loss:.4f}  train_acc={train_acc:.3f} val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        # ---- save best
        if val_acc > best['acc']:
            best['acc'] = val_acc
            best['state'] = {k: v.cpu() for k, v in model.state_dict().items()}

        # ---- log metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    for step, batch in enumerate(train_loader, 1):
        x_rgb = batch['frame'].to(device)
        x_flow = batch['flows'].to(device)
        y = batch['label'].to(device).long()
        opt.zero_grad()
        logits = model(x_rgb, x_flow)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

    # 3) Evaluate once on test
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"TEST: loss={test_loss:.4f}  acc={test_acc:.3f}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    metrics_path = os.path.join(project_dir, "183656/metrics", "two_stream_metrics_avg.csv")
    metrics_df = pd.DataFrame({
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs
    })
    metrics_df.to_csv(metrics_path, index=False)

