"""
Data module (CIFAR-10).

Expected responsibilities:
- load train/val/test splits,
- build DataLoaders for SSL and supervised stages,
- optionally support CIFAR-10-C for robustness/TTT.
"""

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10

from src.data.transforms import TransformFactory


class CIFARDataModule:

    def __init__(self, data_root="./data", image_size=32, batch_size=128,
                 num_workers=2, val_fraction=0.1, seed=42):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

        factory = TransformFactory(image_size)
        self.simclr_tf = factory.build_simclr()
        self.eval_tf = factory.build_eval()

        self.ssl_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # download if not already there
        CIFAR10(root=self.data_root, train=True, download=True)
        CIFAR10(root=self.data_root, train=False, download=True)

    def setup(self):
        self.ssl_dataset = CIFAR10(self.data_root, train=True, transform=self.simclr_tf)

        full = CIFAR10(self.data_root, train=True, transform=self.eval_tf)
        n_val = int(len(full) * self.val_fraction)
        n_train = len(full) - n_val

        gen = torch.Generator().manual_seed(self.seed)
        train_idx, val_idx = random_split(range(len(full)), [n_train, n_val], generator=gen)

        self.train_dataset = Subset(full, train_idx.indices)
        self.val_dataset = Subset(full, val_idx.indices)
        self.test_dataset = CIFAR10(self.data_root, train=False, transform=self.eval_tf)

    def train_ssl_loader(self):
        # label is ignored during SSL — SimCLR only uses the two views
        # drop_last=True: NT-Xent compares pairs within the batch,
        # a partial last batch breaks the loss computation
        return DataLoader(self.ssl_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)

    def supervised_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers,
                                pin_memory=True)
        return train_loader, val_loader

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)
