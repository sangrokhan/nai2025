"""
Tests for data processing
"""

import pytest
import numpy as np
import pandas as pd
from src.data import CellularDataPreprocessor, HierarchicalDataset


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing"""
    n_samples = 100
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

    return pd.DataFrame(data)


def test_preprocessor_fit(sample_dataframe):
    """Test preprocessor fitting"""
    preprocessor = CellularDataPreprocessor()

    # Fit
    preprocessor.fit(sample_dataframe)

    # Check that scalers are fitted
    assert preprocessor.is_fitted
    assert len(preprocessor.scalers) > 0

    # Check that all groups have scalers
    expected_groups = (
        [f'cqi_{i}' for i in range(4)] +
        [f'sinr_{i}' for i in range(2)] +
        ['ri']
    )

    for group in expected_groups:
        assert group in preprocessor.scalers


def test_preprocessor_transform(sample_dataframe):
    """Test preprocessor transformation"""
    preprocessor = CellularDataPreprocessor()
    preprocessor.fit(sample_dataframe)

    # Transform
    processed = preprocessor.transform(sample_dataframe)

    # Check all required keys present
    assert 'cqi_0' in processed
    assert 'sinr_0' in processed
    assert 'ri' in processed
    assert 'mcs_one_layer_su_mimo' in processed
    assert 'ue_count' in processed
    assert 'prb_util' in processed
    assert 'throughput' in processed

    # Check shapes
    n_samples = len(sample_dataframe)
    assert processed['cqi_0'].shape == (n_samples, 16)
    assert processed['sinr_0'].shape == (n_samples, 20)
    assert processed['ri'].shape == (n_samples, 4)
    assert processed['mcs_one_layer_su_mimo'].shape == (n_samples, 32)

    # Check normalization (should have roughly zero mean and unit std)
    assert abs(processed['cqi_0'].mean()) < 0.5
    assert abs(processed['cqi_0'].std() - 1.0) < 0.5


def test_dataset_creation(sample_dataframe):
    """Test dataset creation"""
    preprocessor = CellularDataPreprocessor()
    processed = preprocessor.fit_transform(sample_dataframe)

    dataset = HierarchicalDataset(processed, include_targets=True)

    # Check length
    assert len(dataset) == len(sample_dataframe)

    # Get a sample
    sample = dataset[0]

    # Check all required keys
    assert 'cqi_0' in sample
    assert 'sinr_0' in sample
    assert 'ri' in sample
    assert 'mcs_one_layer_su_mimo' in sample
    assert 'ue_count' in sample
    assert 'prb_util' in sample
    assert 'throughput' in sample

    # Check types
    assert isinstance(sample['cqi_0'], torch.Tensor)
    assert isinstance(sample['throughput'], torch.Tensor)


def test_dataset_without_targets(sample_dataframe):
    """Test dataset without targets"""
    preprocessor = CellularDataPreprocessor()
    processed = preprocessor.fit_transform(sample_dataframe)

    # Remove target
    del processed['throughput']

    dataset = HierarchicalDataset(processed, include_targets=False)

    # Get a sample
    sample = dataset[0]

    # Target should not be present
    assert 'throughput' not in sample


def test_dataloader(sample_dataframe):
    """Test dataloader with collate function"""
    from torch.utils.data import DataLoader
    from src.data.dataset import collate_fn

    preprocessor = CellularDataPreprocessor()
    processed = preprocessor.fit_transform(sample_dataframe)

    dataset = HierarchicalDataset(processed)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_fn,
    )

    # Get a batch
    batch = next(iter(dataloader))

    # Check batch structure
    assert isinstance(batch, dict)
    assert 'cqi_0' in batch
    assert batch['cqi_0'].shape[0] == 16  # batch size


import torch  # Add import for test_dataset_creation
