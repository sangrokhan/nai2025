"""
Integration tests for full training pipeline
"""

import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data import CellularDataPreprocessor, HierarchicalDataset
from src.data.dataset import collate_fn
from src.models import HierarchicalModel
from src.training.losses import HierarchicalLoss


@pytest.fixture
def small_dataset():
    """Create small dataset for integration testing"""
    n_samples = 200
    data = {}

    # Physical features
    for i in range(4):
        for j in range(16):
            data[f'CQI_{i}_{j}'] = np.random.rand(n_samples) * 15

    for i in range(2):
        for j in range(20):
            data[f'SINR_{i}_{j}'] = np.random.randn(n_samples) * 10

    for i in range(4):
        data[f'RI_{i}'] = np.random.rand(n_samples)

    # MCS features
    for layer in ['ONE_LAYER', 'TWO_LAYER', 'THREE_LAYER', 'FOUR_LAYER']:
        for mimo in ['SU_MIMO', 'MU_MIMO']:
            for i in range(32):
                data[f'MCS_{layer}_{mimo}_MCS{i}'] = np.random.rand(n_samples) * 100

    # Context features
    data['UE_COUNT'] = np.random.randint(1, 50, n_samples)
    data['PRB_UTILIZATION'] = np.random.rand(n_samples)

    # Target
    data['THROUGHPUT'] = np.random.rand(n_samples) * 100_000
    data['SPECTRAL_EFFICIENCY'] = np.random.rand(n_samples) * 10

    return pd.DataFrame(data)


def test_full_pipeline(small_dataset):
    """Test full data pipeline → model → loss"""
    # Preprocess data
    preprocessor = CellularDataPreprocessor()
    processed = preprocessor.fit_transform(small_dataset)

    # Create dataset and dataloader
    dataset = HierarchicalDataset(processed)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Create model
    model = HierarchicalModel(channel_dim=64)

    # Create loss function
    loss_fn = HierarchicalLoss()

    # Get a batch and run through model
    batch = next(iter(dataloader))
    outputs = model(batch)
    losses = loss_fn(outputs, batch)

    # Check everything works
    assert losses['total'].item() > 0
    assert not torch.isnan(losses['total'])
    assert not torch.isinf(losses['total'])


def test_overfitting_small_batch(small_dataset):
    """
    Test model can overfit small batch (capacity check)

    This validates that the model has sufficient capacity.
    """
    # Use very small subset
    small_df = small_dataset.iloc[:32]

    # Preprocess
    preprocessor = CellularDataPreprocessor()
    processed = preprocessor.fit_transform(small_df)

    # Create dataset
    dataset = HierarchicalDataset(processed)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn,
    )

    # Create model
    model = HierarchicalModel(channel_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = HierarchicalLoss()

    # Train for many epochs
    initial_loss = None
    final_loss = None

    model.train()
    for epoch in range(100):
        for batch in dataloader:
            # Forward
            outputs = model(batch)
            losses = loss_fn(outputs, batch)

            if initial_loss is None:
                initial_loss = losses['throughput_mse'].item()

            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

            final_loss = losses['throughput_mse'].item()

    # Check that loss decreased significantly
    assert final_loss < initial_loss * 0.5, \
        f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    print(f"\nOverfitting test: {initial_loss:.4f} -> {final_loss:.4f}")


def test_training_stability(small_dataset):
    """Test that training is stable (no NaN/Inf)"""
    # Preprocess
    preprocessor = CellularDataPreprocessor()
    processed = preprocessor.fit_transform(small_dataset)

    # Create dataset
    dataset = HierarchicalDataset(processed)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Create model
    model = HierarchicalModel(channel_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = HierarchicalLoss()

    # Train for a few epochs
    model.train()
    for epoch in range(3):
        for batch in dataloader:
            outputs = model(batch)
            losses = loss_fn(outputs, batch)

            # Check for NaN/Inf
            assert not torch.isnan(losses['total']), \
                f"NaN loss at epoch {epoch}"
            assert not torch.isinf(losses['total']), \
                f"Inf loss at epoch {epoch}"

            optimizer.zero_grad()
            losses['total'].backward()

            # Check gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), \
                        f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), \
                        f"Inf gradient in {name}"

            optimizer.step()

    print("\nTraining stability test passed!")
