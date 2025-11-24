"""
Tests for model components
"""

import pytest
import torch
from src.models import (
    PhysicalLayerEncoder,
    LinkAdaptationEncoder,
    AuxiliaryTasks,
    LAAuxiliaryTasks,
    HierarchicalModel,
)


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def channel_dim():
    return 64


@pytest.fixture
def dummy_batch(batch_size):
    """Create dummy batch for testing"""
    batch = {}

    # Physical features
    for i in range(4):
        batch[f'cqi_{i}'] = torch.randn(batch_size, 16)

    for i in range(2):
        batch[f'sinr_{i}'] = torch.randn(batch_size, 20)

    batch['ri'] = torch.randn(batch_size, 4)

    # MCS features
    for layer in ['one_layer', 'two_layer', 'three_layer', 'four_layer']:
        for mimo in ['su_mimo', 'mu_mimo']:
            batch[f'mcs_{layer}_{mimo}'] = torch.abs(torch.randn(batch_size, 32))

    # Context features
    batch['ue_count'] = torch.randn(batch_size, 1)
    batch['prb_util'] = torch.randn(batch_size, 1)

    # Target
    batch['throughput'] = torch.randn(batch_size)

    return batch


def test_physical_encoder(dummy_batch, channel_dim):
    """Test physical layer encoder"""
    encoder = PhysicalLayerEncoder(channel_dim=channel_dim)

    # Forward pass
    h_channel = encoder(dummy_batch)

    # Check output shape
    assert h_channel.shape == (len(dummy_batch['ri']), channel_dim)

    # Check gradient flow
    loss = h_channel.sum()
    loss.backward()

    for param in encoder.parameters():
        assert param.grad is not None


def test_la_encoder(dummy_batch, channel_dim):
    """Test link adaptation encoder"""
    encoder = LinkAdaptationEncoder(channel_dim=channel_dim)

    # Create dummy h_channel
    h_channel = torch.randn(len(dummy_batch['ri']), channel_dim)

    # Forward pass
    h_LA = encoder(dummy_batch, h_channel)

    # Check output shape
    assert h_LA.shape == (len(dummy_batch['ri']), channel_dim)

    # Check gradient flow
    loss = h_LA.sum()
    loss.backward()

    for param in encoder.parameters():
        assert param.grad is not None


def test_auxiliary_tasks(channel_dim):
    """Test auxiliary tasks"""
    batch_size = 16
    aux_tasks = AuxiliaryTasks(channel_dim=channel_dim)

    h_channel = torch.randn(batch_size, channel_dim)

    # Forward pass
    outputs = aux_tasks(h_channel)

    # Check outputs
    assert 'se_pred' in outputs
    assert 'ri_dist_pred' in outputs
    assert 'quality_pred' in outputs

    assert outputs['se_pred'].shape == (batch_size, 1)
    assert outputs['ri_dist_pred'].shape == (batch_size, 4)
    assert outputs['quality_pred'].shape == (batch_size, 1)

    # Check RI distribution sums to 1
    assert torch.allclose(
        outputs['ri_dist_pred'].sum(dim=1),
        torch.ones(batch_size),
        atol=1e-5,
    )

    # Check quality is in [0, 1]
    assert (outputs['quality_pred'] >= 0).all()
    assert (outputs['quality_pred'] <= 1).all()


def test_la_auxiliary_tasks(channel_dim):
    """Test LA auxiliary tasks"""
    batch_size = 16
    la_aux_tasks = LAAuxiliaryTasks(channel_dim=channel_dim)

    h_LA = torch.randn(batch_size, channel_dim)

    # Forward pass
    outputs = la_aux_tasks(h_LA)

    # Check outputs
    assert 'mcs_avg_pred' in outputs
    assert 'su_ratio_pred' in outputs

    assert outputs['mcs_avg_pred'].shape == (batch_size, 1)
    assert outputs['su_ratio_pred'].shape == (batch_size, 1)

    # Check SU ratio is in [0, 1]
    assert (outputs['su_ratio_pred'] >= 0).all()
    assert (outputs['su_ratio_pred'] <= 1).all()


def test_hierarchical_model(dummy_batch, channel_dim):
    """Test full hierarchical model"""
    model = HierarchicalModel(channel_dim=channel_dim)

    # Forward pass
    outputs = model(dummy_batch)

    # Check all outputs present
    assert 'throughput_pred' in outputs
    assert 'h_channel' in outputs
    assert 'h_LA' in outputs
    assert 'physical_aux' in outputs
    assert 'la_aux' in outputs

    # Check shapes
    batch_size = len(dummy_batch['throughput'])
    assert outputs['throughput_pred'].shape == (batch_size, 1)
    assert outputs['h_channel'].shape == (batch_size, channel_dim)
    assert outputs['h_LA'].shape == (batch_size, channel_dim)

    # Check gradient flow
    loss = outputs['throughput_pred'].sum()
    loss.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_model_parameter_count(channel_dim):
    """Test that model has reasonable parameter count"""
    model = HierarchicalModel(channel_dim=channel_dim)

    total_params = sum(p.numel() for p in model.parameters())

    # Should be in reasonable range (adjust based on expected size)
    assert total_params > 1000  # At least 1K parameters
    assert total_params < 50_000_000  # Less than 50M parameters

    print(f"Model has {total_params:,} parameters")
