import dataset.POTHOLESDataset as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
from matplotlib import patches

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
    [train_size, val_size, test_size]
)
def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

image, masks = next(iter(train_loader))
for im, mask in zip(image, masks):
    fig, ax = plt.subplots()
    ax.imshow(im.permute(1, 2, 0).numpy())
    for mas in mask:
        rect = patches.Rectangle((mas.bbox.xmin, mas.bbox.ymin), mas.bbox.xmax - mas.bbox.xmin, mas.bbox.ymax - mas.bbox.ymin, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()