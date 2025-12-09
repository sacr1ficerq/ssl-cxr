from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms import v2

import os

import matplotlib.pyplot as plt
import torch

# grayscale assumed
MEDMNIST_MEAN = (0.5,)
MEDMNIST_STD = (0.5,)


def get_medmnist_transforms(size: int = 224, augment: bool = False) -> transforms.Compose:
    if augment:
        # SSL augmentations for SimCLR/VICReg pretraining
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEDMNIST_MEAN, MEDMNIST_STD),

            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),

            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.5, 1.0),
                ratio=(0.8, 1.2),
                antialias=True
            ),
            v2.RandomHorizontalFlip(p=0.5),  # for Pretraining OK
            v2.RandomAffine(
                degrees=12,
                translate=(0.1, 0.1),
                scale=(0.95, 1.05)
            ),

            v2.RandomApply([
                v2.GaussianNoise(mean=0.0, sigma=0.05)
            ], p=0.2),

            v2.Lambda(lambda x: x.clamp(0.0, 1.0)),
            v2.Normalize(mean=[0.5], std=[0.5])
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
    else:
        # Standard transforms for linear probing/evaluation
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


class SSLDataset(Dataset):
    """
    returns two augmented views of the same image
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        # We access the underlying dataset's __getitem__ without transform
        # by accessing the internal data directly to avoid double transform

        assert isinstance(self.base_dataset, CustomNPZDataset)
        # manually get PIL image to apply different transforms
        img_array = self.base_dataset.images[idx]
        label = self.base_dataset.labels[idx]

        assert img_array.ndim == 2
        image = Image.fromarray(img_array, mode='L')

        # flatten label if needed
        label = torch.tensor(label, dtype=torch.long)
        if label.ndim > 0 and label.shape[0] == 1:
            label = label.squeeze()
        assert label.ndim == 0

        view1 = self.transform(image)
        view2 = self.transform(image)

        return {
            "view1": view1,
            "view2": view2,
            "labels": label
        }


class HFDataset(Dataset):
    """Dataset wrapper to make datasets compatible with HF Trainer"""

    def __init__(self, dataset, for_ssl: bool = False):
        self.dataset = dataset
        self.for_ssl = for_ssl

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Any]:
        if self.for_ssl:
            assert isinstance(self.dataset[idx], dict)
            return self.dataset[idx]
        else:
            image, label = self.dataset[idx]
            return {"pixel_values": image, "labels": label}


def plot_ssl_views(item):
    view1 = item['view1']
    view2 = item['view2']
    label = item['labels']

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
    data_path = '../data/pneumoniamnist_224.npz'
    size = 224
    # For SSL
    print("SSL")

    # SSLDataset wrapper handles the transforms (generating 2 views).
    train_base = CustomNPZDataset(data_path, split='train', transform=None)
    val_base = CustomNPZDataset(data_path, split='val', transform=None)
    test_base = CustomNPZDataset(data_path, split='test', transform=None)

    # Wrap with SSL dataset
    transform = get_medmnist_transforms(size=size, augment=True)

    train_set = SSLDataset(train_base, transform)
    val_set = SSLDataset(val_base, transform)  # Usually SSL is only trained on train_set
    test_set = SSLDataset(test_base, transform)

    train_set = HFDataset(train_set, for_ssl=True)
    val_set = HFDataset(val_set, for_ssl=True)
    test_set = HFDataset(test_set, for_ssl=True)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(train_set[0].keys())

    # For SFT
    print("\n\nSFT:")
    transform = get_medmnist_transforms(size=size, augment=False)

    train_base = CustomNPZDataset(data_path, split='train', transform=transform)
    val_base = CustomNPZDataset(data_path, split='val', transform=transform)
    test_base = CustomNPZDataset(data_path, split='test', transform=transform)

    train_set = HFDataset(train_base, for_ssl=False)
    val_set = HFDataset(val_base, for_ssl=False)
    test_set = HFDataset(test_base, for_ssl=False)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(train_set[0].keys())
