"""
Data loading utilities for AVHRR Pathfinder SST imagery.

Reads NetCDF files, applies the preprocessing described in
Section 4.1 of the paper, and splits the dataset into train / test
subsets based on a fixed set of ``MM-DD`` date strings.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import albumentations as A
import numpy as np
import xarray as xr
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# ── Fixed 20 % test split (73 out of 365 days) ──────────────────────────────
# These dates are kept identical across all years in multi-year experiments
# to ensure a consistent, reproducible train/test partition.
DEFAULT_TEST_DATES: List[str] = [
    "01-02","01-09","01-18","01-24","01-29","02-11","02-18","02-22",
    "02-26","03-03","03-04","03-05","03-06","03-11","03-20","03-23",
    "03-25","03-31","04-01","04-04","04-09","04-10","04-17","04-18",
    "04-21","04-26","04-30","05-05","05-08","05-10","05-14","05-15",
    "05-17","05-22","05-23","05-24","06-09","06-12","06-20","06-25",
    "06-26","07-08","07-09","07-12","07-13","07-14","07-20","07-21",
    "07-23","07-28","08-02","08-14","08-15","08-16","08-23","08-28",
    "09-04","09-14","09-28","09-29","10-05","10-13","10-14","10-17",
    "10-25","11-05","11-11","11-21","11-27","12-10","12-11","12-17","12-28",
]

# Land / invalid pixel sentinel values in the Default_qual4 convention.
# They are remapped to 0 (sea class) since Mask-Net performs binary
# cloud / sea discrimination only.
_LAND_VALUES = {253, 254}


# ── Transforms ──────────────────────────────────────────────────────────────

def make_transform(image_size: int) -> A.Compose:
    """Build the Albumentations preprocessing pipeline.

    Applied identically to training and test samples.

    Parameters
    ----------
    image_size : int
        Target square resolution (288 for 9 km, 144 for 18 km).
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])


# ── Preprocessing ────────────────────────────────────────────────────────────

def _preprocess_mask(raw: np.ndarray) -> np.ndarray:
    """Clean a raw cloud-mask array.

    - Fills NaN values with 0 (sea class).
    - Replaces land-mask sentinels (253, 254) with 0.
    - Casts to uint8.
    """
    mask = np.nan_to_num(raw, nan=0.0).astype(np.uint8)
    mask[np.isin(mask, list(_LAND_VALUES))] = 0
    return mask


def _preprocess_sst(raw: np.ndarray) -> np.ndarray:
    """Clean a raw SST field: replace NaN with 0, add channel axis."""
    return np.nan_to_num(raw, nan=0.0).astype(np.float32)[:, :, np.newaxis]


# ── Dataset ──────────────────────────────────────────────────────────────────

class SSTDataset(Dataset):
    """Thin Dataset wrapper around a list of (image_tensor, mask_tensor)."""

    def __init__(self, samples: list) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ── Loading ──────────────────────────────────────────────────────────────────

def _load_one_year(
    image_path: str,
    mask_path: str,
    transform: A.Compose,
    test_dates: Sequence[str],
) -> Tuple[list, list, list]:
    """Load one year of SST + cloud-mask data and split into train / test.

    Parameters
    ----------
    image_path, mask_path : str
        NetCDF files for SST (variable ``sst``) and cloud mask
        (variable ``dc``), one per year.
    transform : A.Compose
        Preprocessing pipeline.
    test_dates : Sequence[str]
        ``MM-DD`` strings identifying test days; all others go to
        training.

    Returns
    -------
    train_samples, test_samples : list of (image, mask) tuples
    test_timestamps : list of raw xarray time values (one per test day)
    """
    test_set = set(test_dates)

    sst_ds  = xr.open_dataset(image_path, drop_variables="bounds_time")
    mask_ds = xr.open_dataset(mask_path,  drop_variables="bounds_time")

    train_samples, test_samples, test_timestamps = [], [], []

    for i in range(len(sst_ds.sst)):
        timestamp = sst_ds.sst[i].time.data
        day_key   = str(timestamp).split("T")[0][5:]   # "MM-DD"

        sst   = _preprocess_sst(sst_ds.sst[i].data)
        mask  = _preprocess_mask(mask_ds.dc[i].fillna(0).data)

        aug = transform(image=sst, mask=mask)
        sample = (aug["image"], aug["mask"])

        if day_key in test_set:
            test_samples.append(sample)
            test_timestamps.append(timestamp)
        else:
            train_samples.append(sample)

    sst_ds.close()
    mask_ds.close()

    return train_samples, test_samples, test_timestamps


def build_dataloaders(
    image_paths : Sequence[str],
    mask_paths  : Sequence[str],
    resolution_km: int = 9,
    batch_size  : int = 8,
    test_dates  : Sequence[str] = DEFAULT_TEST_DATES,
    num_workers : int = 0,
) -> Tuple[DataLoader, DataLoader, list]:
    """End-to-end helper: load data and return train / test DataLoaders.

    Parameters
    ----------
    image_paths, mask_paths : Sequence[str]
        Per-year NetCDF paths in matching order.  Use a single-element
        list for a single-year dataset.
    resolution_km : {9, 18}, default 9
        Determines the target image size (288 or 144).
    batch_size : int, default 8
    test_dates : Sequence[str], default ``DEFAULT_TEST_DATES``
    num_workers : int, default 0

    Returns
    -------
    train_loader, test_loader : DataLoader
    test_timestamps : list
        Original xarray timestamps for each test sample, useful for
        per-day evaluation (reproducing Figs. 6–8 of the paper).
    """
    image_size = 288 if resolution_km == 9 else 144
    transform  = make_transform(image_size)

    all_train, all_test, all_ts = [], [], []
    for img_p, msk_p in zip(image_paths, mask_paths):
        train, test, ts = _load_one_year(img_p, msk_p, transform, test_dates)
        all_train.extend(train)
        all_test.extend(test)
        all_ts.extend(ts)

    train_loader = DataLoader(
        SSTDataset(all_train), batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        SSTDataset(all_test), batch_size=batch_size,
        shuffle=False, pin_memory=True, num_workers=num_workers,
    )

    print(
        f"[data] {len(all_train)} train  |  {len(all_test)} test  "
        f"({resolution_km} km / {image_size}×{image_size})"
    )
    return train_loader, test_loader, all_ts
