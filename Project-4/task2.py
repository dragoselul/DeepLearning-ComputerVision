import dataset.POTHOLESDataset as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch

size = (560, 576)

train_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])
potholes = ds.POTHOLES(split="train", transform=train_transform, label_transform=label_transform, augment=True)

n = len(potholes)
train_size = int(0.7 * n)
val_size   = int(0.15 * n)
test_size  = n - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    potholes,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)