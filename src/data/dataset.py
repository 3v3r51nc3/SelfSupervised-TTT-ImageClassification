"""
Data module (CIFAR-10 and CIFAR-10-C).

Responsibilities:
- load train/val/test splits,
- build DataLoaders for SSL and supervised stages,
- expose CIFAR-10-C loaders for robustness/TTT evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from src.data.transforms import TransformFactory


class CIFARDataModule:

    def __init__(
        self,
        data_root="./data",
        image_size=32,
        batch_size_ssl=128,
        batch_size_sup=128,
        num_workers=2,
        val_fraction=0.1,
        seed=42,
        augment_supervised: bool = True,
        randaugment_n: int = 2,
        randaugment_m: int = 9,
    ):
        self.data_root = data_root
        self.batch_size_ssl = batch_size_ssl
        self.batch_size_sup = batch_size_sup
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

        factory = TransformFactory(
            image_size=image_size,
            randaug_n=randaugment_n,
            randaug_m=randaugment_m,
        )
        self.simclr_tf = factory.build_simclr()
        self.sup_train_tf = factory.build_supervised_train(augment=augment_supervised)
        self.eval_tf = factory.build_eval()

        self.ssl_train_dataset = None
        self.ssl_val_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # download if not already there
        CIFAR10(root=self.data_root, train=True, download=True)
        CIFAR10(root=self.data_root, train=False, download=True)

    def setup(self):
        ssl_full = CIFAR10(self.data_root, train=True, transform=self.simclr_tf)
        sup_train_full = CIFAR10(self.data_root, train=True, transform=self.sup_train_tf)
        sup_val_full = CIFAR10(self.data_root, train=True, transform=self.eval_tf)
        total_examples = len(sup_val_full)
        n_val = int(total_examples * self.val_fraction)
        n_train = total_examples - n_val
        if n_val <= 0 or n_train <= 0:
            raise ValueError("val_fraction must leave at least one example in both train and val splits.")
        gen = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(total_examples, generator=gen).tolist()
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        self.ssl_train_dataset = Subset(ssl_full, train_indices)
        self.ssl_val_dataset = Subset(ssl_full, val_indices)
        self.train_dataset = Subset(sup_train_full, train_indices)
        self.val_dataset = Subset(sup_val_full, val_indices)
        self.test_dataset = CIFAR10(self.data_root, train=False, transform=self.eval_tf)

    def ssl_loaders(self):
        train_loader = DataLoader(
            self.ssl_train_dataset,
            batch_size=self.batch_size_ssl,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
        val_loader = DataLoader(
            self.ssl_val_dataset,
            batch_size=self.batch_size_ssl,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
        return train_loader, val_loader

    def train_ssl_loader(self):
        # label is ignored during SSL — SimCLR only uses the two views
        # drop_last=True: NT-Xent compares pairs within the batch,
        # a partial last batch breaks the loss computation
        train_loader, _ = self.ssl_loaders()
        return train_loader

    def supervised_loaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_sup,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_sup,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        return train_loader, val_loader

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_sup,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def cifar10c_loader(self, corruption: str, severity: int) -> DataLoader:
        """
        Load a specific corruption and severity from CIFAR-10-C.

        Each corruption file contains 50k images (5 severities x 10k images).
        severity 1 -> images[0:10000], severity 5 -> images[40000:50000].
        """
        if not 1 <= severity <= 5:
            raise ValueError(f"severity must be in [1, 5], got {severity}.")

        path = Path(self.data_root) / "CIFAR-10-C"
        images = np.load(path / f"{corruption}.npy")
        labels = np.load(path / "labels.npy")

        start = (severity - 1) * 10000
        end = severity * 10000
        images = images[start:end]
        labels = labels[start:end]

        dataset = CIFAR10CDataset(images, labels, transform=self.eval_tf)
        return DataLoader(
            dataset,
            batch_size=self.batch_size_sup,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def split_sizes(self) -> dict[str, int]:
        return {
            "ssl_train": len(self.ssl_train_dataset),
            "ssl_val": len(self.ssl_val_dataset),
            "supervised_train": len(self.train_dataset),
            "supervised_val": len(self.val_dataset),
            "test": len(self.test_dataset),
        }

    @staticmethod
    def cifar10c_corruptions() -> list[str]:
        return [
            "gaussian_noise", "shot_noise", "impulse_noise",
            "defocus_blur", "glass_blur", "motion_blur",
            "snow", "frost", "fog", "brightness",
            "contrast", "elastic_transform", "pixelate", "jpeg_compression",
        ]


class CIFAR10CDataset(Dataset):
    """Dataset wrapper for one (corruption, severity) slice of CIFAR-10-C."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])
        img = self.transform(img)
        return img, int(self.labels[idx])
