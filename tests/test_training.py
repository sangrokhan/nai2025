"""
Tests for training components
"""

import pytest
import torch
from src.training.losses import (
    compute_auxiliary_losses,
    compute_la_auxiliary_losses,
    HierarchicalLoss,
)
from src.models import HierarchicalModel


@pytest.fixture
def dummy_batch():
    """Create dummy batch for testing"""
    batch_size = 16
    batch = {}

    # Physical features
    for i in range(4):
        batch[f'cqi_{i}'] = torch.randn(batch_size, 16)

    for i in range(2):
        batch[f'sinr_{i}'] = torch.randn(batch_size, 20)

    batch['ri'] = torch.abs(torch.randn(batch_size, 4))

    # MCS features
    for layer in ['one_layer', 'two_layer', 'three_layer', 'four_layer']:
        for mimo in ['su_mimo', 'mu_mimo']:
            batch[f'mcs_{layer}_{mimo}'] = torch.abs(torch.randn(batch_size, 32))

    # Context features
    batch['ue_count'] = torch.randn(batch_size, 1)
    batch['prb_util'] = torch.randn(batch_size, 1)

    # Target
    batch['throughput'] = torch.randn(batch_size)
    batch['spectral_efficiency'] = torch.abs(torch.randn(batch_size))

    return batch


def test_auxiliary_loss_computation(dummy_batch):
    """
    Test auxiliary loss computation

    CRITICAL: This ensures auxiliary losses are not zero!
    """
    # Create dummy auxiliary outputs
    batch_size = len(dummy_batch['throughput'])
    aux_outputs = {
        'se_pred': torch.randn(batch_size, 1),
        'ri_dist_pred': torch.softmax(torch.randn(batch_size, 4), dim=-1),
        'quality_pred': torch.sigmoid(torch.randn(batch_size, 1)),
    }

    weights = {'se': 0.25, 'ri_dist': 0.15, 'quality': 0.15}

    # Compute losses
    losses = compute_auxiliary_losses(aux_outputs, dummy_batch, weights)

    # Check all losses present
    assert 'se' in losses
    assert 'ri_dist' in losses
    assert 'quality' in losses
    assert 'total' in losses

    # CRITICAL: Check losses are not zero
    assert losses['se'].item() > 0
    assert losses['ri_dist'].item() >= 0  # KL can be 0 if distributions match
    assert losses['quality'].item() > 0
    assert losses['total'].item() > 0

    print("\nAuxiliary Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")


def test_la_auxiliary_loss_computation(dummy_batch):
    """Test LA auxiliary loss computation"""
    batch_size = len(dummy_batch['throughput'])
    la_aux_outputs = {
        'mcs_avg_pred': torch.randn(batch_size, 1),
        'su_ratio_pred': torch.sigmoid(torch.randn(batch_size, 1)),
    }

    weights = {'mcs_avg': 0.15, 'su_ratio': 0.08}

    # Compute losses
    losses = compute_la_auxiliary_losses(la_aux_outputs, dummy_batch, weights)

    # Check all losses present
    assert 'mcs_avg' in losses
    assert 'su_ratio' in losses
    assert 'total' in losses

    # CRITICAL: Check losses are not zero
    assert losses['mcs_avg'].item() > 0
    assert losses['su_ratio'].item() > 0
    assert losses['total'].item() > 0

    print("\nLA Auxiliary Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")


def test_hierarchical_loss(dummy_batch):
    """Test total hierarchical loss"""
    # Create model
    model = HierarchicalModel(channel_dim=64)

    # Forward pass
    outputs = model(dummy_batch)

    # Create loss function
    loss_fn = HierarchicalLoss()

    # Compute losses
    losses = loss_fn(outputs, dummy_batch)

    # Check all components present
    assert 'throughput_mse' in losses
    assert 'throughput_mae' in losses
    assert 'physical_total_aux' in losses
    assert 'la_total_aux' in losses
    assert 'total' in losses

    # Check total loss is sum of components
    expected_total = (
        losses['throughput_mse'] +
        losses['physical_total_aux'] +
        losses['la_total_aux']
    )

    assert torch.allclose(losses['total'], expected_total, atol=1e-5)

    print("\nHierarchical Losses:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.6f}")


def test_training_step(dummy_batch):
    """Test a single training step"""
    # Create model
    model = HierarchicalModel(channel_dim=64)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create loss function
    loss_fn = HierarchicalLoss()

    # Training step
    model.train()

    # Forward
    outputs = model(dummy_batch)
    losses = loss_fn(outputs, dummy_batch)

    # Backward
    optimizer.zero_grad()
    losses['total'].backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    # Update
    optimizer.step()

    # Check that parameters changed
    # (we'll just check that we can do another forward pass)
    outputs2 = model(dummy_batch)
    assert outputs2['throughput_pred'].shape == outputs['throughput_pred'].shape
