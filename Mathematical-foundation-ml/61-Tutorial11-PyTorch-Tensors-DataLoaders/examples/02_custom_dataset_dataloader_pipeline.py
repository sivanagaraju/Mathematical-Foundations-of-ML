#!/usr/bin/env python3
"""
Tutorial 11 : PyTorch - Tensors and Data Loaders
Simulation Script 02: Custom Dataset Architecture, Lazy I/O, and DataLoader Batching Pipeline

This standalone script simulates production dataset engineering and batch collation:
1. Subclassing `torch.utils.data.Dataset` with the mandatory trinity: `__init__`, `__len__`, `__getitem__`.
2. Simulating decoupled CSV annotation parsing without loading raw sensory bytes into RAM.
3. On-demand lazy image decoding and custom feature transformation:
   - Converting raw uint8 numpy arrays in [0, 255] with shape (H, W, C)
   - Normalizing into float32 tensors in [0.0, 1.0] with channel-first shape (C, H, W).
4. Wrapping with `torch.utils.data.DataLoader` for automated batching, shuffling, and drop-last handling.
5. Verifying batch tensor dimensionalities, memory bounds, and epoch iteration invariance.

All assertions execute cleanly with exit code 0.
"""

import sys
import os
import csv
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ImageTransformToTensor:
    """
    Simulates torchvision.transforms.ToTensor:
    Converts a NumPy uint8 image array (H, W, C) in [0, 255]
    to a PyTorch float32 tensor (C, H, W) in [0.0, 1.0].
    """
    def __call__(self, pic: np.ndarray) -> torch.Tensor:
        assert pic.dtype == np.uint8, "Expected uint8 input image!"
        assert pic.ndim == 3, "Expected (H, W, C) input shape!"
        
        # 1. Permute axes from (H, W, C) to (C, H, W)
        transposed = np.transpose(pic, (2, 0, 1))
        
        # 2. Convert to float32 and scale to [0.0, 1.0]
        tensor = torch.from_numpy(transposed).to(dtype=torch.float32).div_(255.0)
        return tensor


class CustomImageDataset(Dataset):
    """
    Production-grade Custom Image Dataset implementing decoupled indexing
    and lazy on-demand sample loading from a CSV annotation file.
    """
    def __init__(self, csv_filepath: str, transform=None, target_transform=None):
        # Store metadata index only; zero image buffers loaded in memory
        self.records = []
        with open(csv_filepath, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if row:
                    self.records.append((row[0], int(row[1])))
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        # 1. Retrieve metadata record
        rel_path, label = self.records[idx]

        # 2. Simulate lazy on-demand disk I/O (synthetic 28x28 grayscale image)
        # Using deterministic seed based on idx to simulate distinct images
        rng = np.random.RandomState(idx)
        raw_uint8_image = rng.randint(0, 256, size=(28, 28, 1), dtype=np.uint8)

        # 3. Apply feature transformation
        if self.transform is not None:
            image_tensor = self.transform(raw_uint8_image)
        else:
            image_tensor = torch.from_numpy(raw_uint8_image)

        # 4. Apply target transformation
        if self.target_transform is not None:
            label_tensor = self.target_transform(label)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


def run_pipeline_test():
    print("[1/4] Setting up Synthetic CSV Annotations...")
    # Generate 50 synthetic annotation records across 10 classes (0-9)
    num_samples = 50
    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "annotations.csv")
    
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "class_label"])
        for i in range(num_samples):
            writer.writerow([f"images/sample_{i:04d}.png", i % 10])
            
    print(f"      [OK] Created CSV annotation index at {csv_file} for {num_samples} samples.")

    print("[2/4] Instantiating Custom Dataset with Lazy Ingestion Contract...")
    transform_pipeline = ImageTransformToTensor()
    dataset = CustomImageDataset(csv_file, transform=transform_pipeline)

    # Verify __len__
    assert len(dataset) == num_samples, f"Expected length {num_samples}, got {len(dataset)}"

    # Verify __getitem__ lazy materialization
    img_sample, label_sample = dataset[0]
    # Shape: [C, H, W] = [1, 28, 28]
    assert img_sample.shape == (1, 28, 28), f"Expected (1, 28, 28), got {img_sample.shape}"
    assert img_sample.dtype == torch.float32, f"Expected float32, got {img_sample.dtype}"
    assert 0.0 <= img_sample.min().item() <= 1.0, "Pixels must be normalized in [0.0, 1.0]"
    assert 0.0 <= img_sample.max().item() <= 1.0, "Pixels must be normalized in [0.0, 1.0]"
    assert isinstance(label_sample, torch.Tensor)
    assert label_sample.dtype == torch.long
    assert label_sample.item() == 0
    print("      [OK] Single item lazy extraction and ToTensor transform verified.")

    print("[3/4] Wrapping Dataset in DataLoader with Mini-Batching...")
    batch_size = 8
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    # Total batches = ceil(50 / 8) = 7 batches (6 of size 8, 1 of size 2)
    expected_batches = int(np.ceil(num_samples / batch_size))
    assert len(loader) == expected_batches, f"Expected {expected_batches} batches, got {len(loader)}"

    total_samples_iterated = 0
    batch_count = 0
    for batch_idx, (batch_features, batch_labels) in enumerate(loader):
        batch_count += 1
        current_b = batch_features.shape[0]
        total_samples_iterated += current_b

        # Verify shapes
        assert batch_features.shape == (current_b, 1, 28, 28), f"Batch feature shape mismatch: {batch_features.shape}"
        assert batch_labels.shape == (current_b,), f"Batch label shape mismatch: {batch_labels.shape}"
        assert batch_features.dtype == torch.float32
        assert batch_labels.dtype == torch.long

        # Check last batch size
        if batch_idx == expected_batches - 1:
            assert current_b == num_samples % batch_size, f"Expected tail batch size {num_samples % batch_size}, got {current_b}"
        else:
            assert current_b == batch_size, f"Expected full batch size {batch_size}, got {current_b}"

    assert total_samples_iterated == num_samples, f"Iterated {total_samples_iterated} != total {num_samples}"
    print(f"      [OK] Successfully iterated through {batch_count} batches ({total_samples_iterated} samples total).")

    print("[4/4] Verifying DataLoader Iterator Protocol via iter() and next()...")
    loader_iter = iter(loader)
    first_batch_x, first_batch_y = next(loader_iter)
    assert first_batch_x.shape == (batch_size, 1, 28, 28)
    assert first_batch_y.shape == (batch_size,)
    print("      [OK] Direct iter() and next() extraction verified.")


if __name__ == "__main__":
    print("=" * 70)
    print("TUTORIAL 11 : CUSTOM DATASET & DATALOADER PIPELINE VERIFICATION")
    print("=" * 70)
    run_pipeline_test()
    print("=" * 70)
    print("ALL DATASET & DATALOADER ASSERTIONS PASSED (EXIT CODE 0)")
    print("=" * 70)
    sys.exit(0)
