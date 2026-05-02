"""
Data module (CIFAR-10 and CIFAR-10-C).

Responsibilities:
- load train/val/test splits,
- build DataLoaders for SSL and supervised stages,
- expose CIFAR-10-C loaders for robustness/TTT evaluation.
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from src.data.transforms import TransformFactory

_LOG = logging.getLogger(__name__)

# torchvision pins the Toronto mirror, which intermittently returns 503.
# We pre-stage the archive from a working mirror first; torchvision then
# verifies the MD5 and skips its own download.
_CIFAR10_ARCHIVE_NAME = "cifar-10-python.tar.gz"
_CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"
_CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
# Mirrors are tried in order; first one that returns content matching the
# expected MD5 wins. We rely on the Internet Archive Wayback Machine because
# (a) PyTorch's ossci-datasets bucket only mirrors MNIST, not CIFAR, and
# (b) HuggingFace hosts CIFAR-10 only as parquet, not the canonical tarball.
# The `id_` modifier on Wayback URLs returns the original payload bytes
# (no toolbar wrapping), and the bare-year form lets Wayback redirect to any
# available snapshot for that year.
_CIFAR10_MIRRORS: tuple[str, ...] = (
    "https://web.archive.org/web/2024id_/https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://web.archive.org/web/20241225200100id_/https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
)


def _md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _ensure_cifar10_archive(data_root: str) -> None:
    """Pre-stage the CIFAR-10 tarball from a working mirror.

    No-op if the extracted dataset directory already exists, or if the archive
    is already present with the expected MD5. On failure across all mirrors,
    returns silently and lets torchvision attempt its own download (which may
    still succeed if its mirror has recovered).
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    if (root / _CIFAR10_EXTRACTED_DIR).is_dir():
        return

    archive = root / _CIFAR10_ARCHIVE_NAME
    if archive.is_file() and _md5_of_file(archive) == _CIFAR10_MD5:
        _LOG.info("CIFAR-10 archive already staged at %s", archive)
        return

    if archive.is_file():
        archive.unlink()

    for mirror in _CIFAR10_MIRRORS:
        for attempt in (1, 2):
            try:
                _LOG.info("Downloading CIFAR-10 from %s (attempt %d)", mirror, attempt)
                with urllib.request.urlopen(mirror, timeout=30) as resp, archive.open("wb") as out:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                _LOG.warning("Mirror %s failed (attempt %d): %s", mirror, attempt, exc)
                if archive.exists():
                    archive.unlink()
                time.sleep(2 * attempt)
                continue

            digest = _md5_of_file(archive)
            if digest == _CIFAR10_MD5:
                _LOG.info("CIFAR-10 staged from %s (md5 verified)", mirror)
                return
            _LOG.warning(
                "Mirror %s returned content with bad MD5 %s (expected %s); discarding",
                mirror,
                digest,
                _CIFAR10_MD5,
            )
            archive.unlink()

    _LOG.warning(
        "All CIFAR-10 mirrors failed; falling through to torchvision's own downloader."
    )


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
        # Stage the archive from a reliable mirror first (no-op if already
        # extracted/staged); torchvision then just verifies + extracts.
        _ensure_cifar10_archive(self.data_root)
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

    def test_loader_subsample(self, n_samples: int, seed: int) -> DataLoader:
        """Stratified random subsample of the clean test set."""
        labels = np.asarray(self.test_dataset.targets, dtype=np.int64)
        indices = _stratified_indices(labels, n_samples=n_samples, seed=seed)
        subset = Subset(self.test_dataset, indices.tolist())
        return DataLoader(
            subset,
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

    def cifar10c_loader_subsample(
        self,
        corruption: str,
        severity: int,
        n_samples: int,
        seed: int,
    ) -> DataLoader:
        """Stratified random subsample of one (corruption, severity) cell."""
        if not 1 <= severity <= 5:
            raise ValueError(f"severity must be in [1, 5], got {severity}.")

        path = Path(self.data_root) / "CIFAR-10-C"
        images = np.load(path / f"{corruption}.npy")
        labels = np.load(path / "labels.npy").astype(np.int64)

        start = (severity - 1) * 10000
        end = severity * 10000
        images = images[start:end]
        labels = labels[start:end]

        # Seed includes severity so each (corruption, severity) gets a
        # distinct deterministic subsample.
        cell_seed = seed + 1000 * severity + hash(corruption) % 1000
        indices = _stratified_indices(labels, n_samples=n_samples, seed=cell_seed)

        dataset = CIFAR10CDataset(images[indices], labels[indices], transform=self.eval_tf)
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


def _stratified_indices(labels: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """
    Pick `n_samples` indices from `labels` with one quota per class.

    Quota is `n_samples // num_classes`; the remainder is distributed
    one-per-class across the first `n_samples % num_classes` classes so
    the returned size equals `n_samples` exactly. Indices are sorted so
    DataLoader iteration is deterministic.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}.")
    if n_samples > len(labels):
        raise ValueError(
            f"n_samples={n_samples} exceeds available {len(labels)} samples."
        )

    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    base, remainder = divmod(n_samples, len(classes))

    chosen: list[int] = []
    for i, cls in enumerate(classes):
        cls_idx = np.flatnonzero(labels == cls)
        quota = base + (1 if i < remainder else 0)
        if quota > len(cls_idx):
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} samples but quota is {quota}."
            )
        picked = rng.choice(cls_idx, size=quota, replace=False)
        chosen.extend(picked.tolist())

    return np.sort(np.asarray(chosen, dtype=np.int64))
