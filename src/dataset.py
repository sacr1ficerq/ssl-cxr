from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

MEDMNIST_MEAN = (0.5,)
MEDMNIST_STD = (0.5,)


def invert(img):
    return ImageOps.invert(img)


def get_medmnist_transforms(size: int = 224, augment: str | None = None) -> transforms.Compose:
    # SSL augmentations for SimCLR/VICReg pretraining
    assert augment is None or augment in ["ssl", "sft", "ref"]
    if augment == "ssl":
        return transforms.Compose([
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.25),
                antialias=True
            ),
            v2.RandomHorizontalFlip(p=0.5),  # for Pretraining OK
            v2.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            v2.RandomApply([
                v2.GaussianNoise(mean=0.0, sigma=0.05)
            ], p=0.8),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
# Legacy augmentations
# return transforms.Compose([
#     transforms.Resize((size, size)),
#     transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#     ], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(MEDMNIST_MEAN, MEDMNIST_STD)
# ])
    elif augment == "sft":
        return transforms.Compose([
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.5, 1.0),
                ratio=(0.9, 1.1),
                antialias=True
            ),
            # v2.RandomHorizontalFlip(p=0.5),  # for Pretraining OK
            v2.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            v2.RandomApply([
                v2.GaussianNoise(mean=0.0, sigma=0.05)
            ], p=0.5),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif augment == "ref":
        # I asked Gemini 3 to rewrite transforms from task 3
        return transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(degrees=(-10.0, 10.0), interpolation=transforms.InterpolationMode.NEAREST, expand=False, fill=0)
            ], p=0.5), # Assuming p=0.5 for RandomApply if not specified, usually wrapper for optionality
            transforms.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2))
            ], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.Lambda(invert)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEDMNIST_MEAN, MEDMNIST_STD)
        ])


class CustomNPZDataset(Dataset):
    """
    expects keys: 'train_images', 'train_labels', 'val_images', 'val_labels'...
    """

    def __init__(self, file_path: str, split: str, transform=None):
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        self.transform = transform

        # might need to use mmap_mode='r' if the file is too large for memory, otherwise load fully
        data = np.load(file_path)

        self.images = data[f'{split}_images']
        self.labels = data[f'{split}_labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        img = self.images[idx]
        target = self.labels[idx]

        # numpy -> PIL
        assert img.ndim == 2
        img = Image.fromarray(img, mode='L')

        if self.transform:
            img = self.transform(img)

        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)

        # flatten label if it's shape (1,)
        if target.ndim > 0 and target.shape[0] == 1:
            target = target.squeeze()
        assert target.ndim == 0

        return img, target


# At first I wanted to use HF-like training pipeline like in first HW but decided against it
# I will just leave this here for reference as legacy code
class HFDataset(Dataset):
    def __init__(self, dataset, for_ssl: bool = False):
        self.dataset = dataset
        self.for_ssl = for_ssl

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        assert isinstance(self.dataset[idx], tuple)
        image, label = self.dataset[idx]
        return {"pixel_values": image, "labels": label}


class SSLDataset:
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, label


def plot_ssl_views(view1, view2, label):
    mean, std = 0.5, 0.5
    view1 = view1 * std + mean
    view2 = view2 * std + mean

    # (C, H, W) -> (H, W, C) for matplotlib
    view1 = view1.permute(1, 2, 0)
    view2 = view2.permute(1, 2, 0)

    # handle grayscale vs RGB
    # if channel dim is 1, squeeze it out for cleaner plotting
    cmap = None
    if view1.shape[-1] == 1:
        view1 = view1.squeeze(-1)
        view2 = view2.squeeze(-1)
        cmap = 'gray'

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # View 1
    axes[0].imshow(view1, cmap=cmap)
    axes[0].set_title("View 1 (Augmentation A)")
    axes[0].axis('off')

    # View 2
    axes[1].imshow(view2, cmap=cmap)
    axes[1].set_title("View 2 (Augmentation B)")
    axes[1].axis('off')

    plt.suptitle(f"Label: {label}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_views(item):
    img = item['pixel_values']
    label = item['labels']

    mean, std = 0.5, 0.5
    img = img * std + mean

    # (C, H, W) -> (H, W, C) for matplotlib
    img = img.permute(1, 2, 0)

    # handle grayscale vs RGB
    # if channel dim is 1, squeeze it out for cleaner plotting
    cmap = None
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
        cmap = 'gray'

    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.imshow(img, cmap=cmap)
    axes.set_title("View 1 (Augmentation A)")
    axes.axis('off')

    plt.suptitle(f"Label: {label}", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # This probably doesn't work because I created this in the beggining back when I had different APIs
    data_path = '../data/pneumoniamnist_224.npz'
    size = 224
    # For SSL
    print("SSL")

    # SSLDataset wrapper handles the transforms (generating 2 views).
    train_base = CustomNPZDataset(data_path, split='train', transform=None)
    val_base = CustomNPZDataset(data_path, split='val', transform=None)
    test_base = CustomNPZDataset(data_path, split='test', transform=None)

    # Wrap with SSL dataset
    transform = get_medmnist_transforms(size=size, augment="ssl")

    train_set = SSLDataset(train_base, transform)
    val_set = SSLDataset(val_base, transform)  # Usually SSL is only trained on train_set
    test_set = SSLDataset(test_base, transform)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(train_set[0])

    # For SFT
    print("\n\nSFT:")
    transform = get_medmnist_transforms(size=size, augment="sft")

    train_set = CustomNPZDataset(data_path, split='train', transform=transform)
    val_set = CustomNPZDataset(data_path, split='val', transform=transform)
    test_set = CustomNPZDataset(data_path, split='test', transform=transform)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(train_set[0])

