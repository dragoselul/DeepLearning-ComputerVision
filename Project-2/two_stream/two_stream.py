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

        frames = self.get_frames(idx)
        flows_npy = self.get_flows(idx)

        return {
            'frames': frames,
            'flows_npy': flows_npy,
            'label': label,
            'video_name': video_name
        }

    def get_flows(self, idx):
        flow_dir = self.video_paths[idx].replace('videos', 'flows').replace('.avi', '')
        flows = self.load_flows(flow_dir)                  # list of (2,H,W) tensors length T-1
        if self.flow_transform:
            flows = [self.flow_transform(f) for f in flows]

        if self.stack_flows:
            f = torch.stack(flows).contiguous()         # (T-1, 2, H, W)
            f = f.view(-1, f.shape[2], f.shape[3])      # (2*(T-1), H, W) = u1,v1,u2,v2,...
            return f
        return flows

    def load_flows(self, frames_dir):
        flows = []
        for i in range(1, self.n_sampled_frames):
            fpath = os.path.join(frames_dir, f"flow_{i}_{i+1}.npy")
            arr = np.load(fpath).astype(np.float32)
            flows.append(torch.from_numpy(arr).contiguous())
        return flows

    def get_frames(self, idx):
        frame_dir = self.video_paths[idx].replace('videos','frames').replace('.avi','')
        mid = (self.n_sampled_frames + 1)//2
        img = Image.open(os.path.join(frame_dir, f"frame_{mid}.jpg")).convert("RGB")
        return self.transform(img) if self.transform else T.ToTensor()(img)
    
def random_hflip_flow(flow, p=0.5):
    # flow: (2,H,W)
    if torch.rand(1).item() < p:
        flow = torch.flip(flow, dims=[2])  # flip width
        flow[0] = -flow[0]                 # negate u
    return flow

def flow_tf_train(flow_2hw: torch.Tensor, out_size=(224,224), clip=20.0):
    H0, W0 = flow_2hw.shape[1:]
    if (H0, W0) != out_size:
        flow = F.interpolate(flow_2hw.unsqueeze(0), size=out_size,
                            mode='bilinear', align_corners=False).squeeze(0)
        sx, sy = out_size[1]/W0, out_size[0]/H0
        flow[0] *= sx; flow[1] *= sy
    else:
        flow = flow_2hw
    # train aug
    if torch.rand(1).item() < 0.5:
        flow = torch.flip(flow, dims=[2]); flow[0] = -flow[0]
    flow = torch.clamp(flow, -clip, clip) / clip
    return flow

def flow_tf_eval(flow_2hw: torch.Tensor, out_size=(224,224), clip=20.0):
    H0, W0 = flow_2hw.shape[1:]
    if (H0, W0) != out_size:
        flow = F.interpolate(flow_2hw.unsqueeze(0), size=out_size,
                            mode='bilinear', align_corners=False).squeeze(0)
        sx, sy = out_size[1]/W0, out_size[0]/H0
        flow[0] *= sx; flow[1] *= sy
    else:
        flow = flow_2hw
    return torch.clamp(flow, -clip, clip) / clip

def collate_two_stream(batch):
    frames = torch.stack([b['frames'] for b in batch])          # (B, 3, 224, 224)
    flows  = torch.stack([b['flows_npy'] for b in batch])       # (B, 2*(T-1), 224, 224)
    labels = torch.tensor([b['label'] for b in batch])
    names  = [b['video_name'] for b in batch]
    return {'frame': frames, 'flows': flows, 'label': labels, 'video_name': names}


def make_vgg16(num_classes, in_ch=3, weights=None):
    m = models.vgg16(weights=weights)
    # Replace first conv if needed
    if in_ch != 3:
        old = m.features[0]
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            w = old.weight
            w_mean = w.mean(1, keepdim=True)
            new.weight.copy_(w_mean.repeat(1, in_ch, 1, 1) * (3.0 / in_ch))
        m.features[0] = new
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m

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
    for b in loader:
        x_rgb = b['frame'].to(device, non_blocking=True)
        x_flow = b['flows'].to(device, non_blocking=True)
        y = b['label'].to(device, non_blocking=True).long()
        opt.zero_grad(set_to_none=True)
        logits = model(x_rgb, x_flow)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

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
        
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    frame_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'

    train_dataset= TwoStreamDataset(root_dir, split='train',transform=frame_tf, flow_transform=flow_tf_train, stack_flows=True)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_two_stream)
    val_dataset = TwoStreamDataset(root_dir, split='val',
                                transform=frame_tf, flow_transform=flow_tf_eval, stack_flows=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_two_stream)

    batch = next(iter(train_loader))
    print('frame:', batch['frame'].shape)  # (B, 3, 224, 224)
    print('flows:', batch['flows'].shape)  # (B, 18, 224, 224) if T=10
    assert batch['frame'].shape[1] == 3, "RGB must have 3 channels"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Frames = 10
    num_classes = 10
    spatial_model = make_vgg16(num_classes, in_ch=3, weights=models.VGG16_Weights.IMAGENET1K_V1)
    temporal_model = make_vgg16(num_classes, in_ch=2*(Frames-1), weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TwoStreamFusion(spatial_model, temporal_model, num_classes=num_classes, fusion='weighted').to(device)



    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    EPOCHS = 10
    best = {'acc': 0.0, 'state': None}
    for epoch in range(1, EPOCHS + 1):
        # ---- train
        train_one_epoch(model, train_loader, opt, device, criterion)

        # ---- validate
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[Fusion][epoch {epoch:02d}] val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        # ---- save best
        if val_acc > best['acc']:
            best['acc'] = val_acc
            best['state'] = {k: v.cpu() for k, v in model.state_dict().items()}
    for step, batch in enumerate(train_loader, 1):
        x_rgb = batch['frame'].to(device)
        x_flow = batch['flows'].to(device)
        y = batch['label'].to(device).long()
        opt.zero_grad()
        logits = model(x_rgb, x_flow)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

