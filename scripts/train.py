#!/usr/bin/env python
"""
Training Script

Trains the hierarchical model on cellular network data.

Usage:
    python scripts/train.py --config configs/medium_model.yaml
    python scripts/train.py --config configs/medium_model.yaml --debug
    python scripts/train.py --config configs/medium_model.yaml --device cpu
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data import CellularDataPreprocessor, HierarchicalDataset
from src.data.dataset import collate_fn
from src.models import HierarchicalModel
from src.training.losses import HierarchicalLoss
from src.training.trainer import Trainer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='Train hierarchical model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)',
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_parquet(config['paths']['train_data'])
    val_df = pd.read_parquet(config['paths']['val_data'])

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")

    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = CellularDataPreprocessor()
    train_processed = preprocessor.fit_transform(train_df)
    val_processed = preprocessor.transform(val_df)

    # Create datasets
    train_dataset = HierarchicalDataset(train_processed, include_targets=True)
    val_dataset = HierarchicalDataset(val_processed, include_targets=True)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = HierarchicalModel(
        channel_dim=config['model']['channel_dim'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        predictor_hidden_dim=config['model']['predictor_hidden_dim'],
        predictor_num_layers=config['model']['predictor_num_layers'],
        dropout=config['model']['dropout'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Create loss function
    loss_fn = HierarchicalLoss(
        main_weight=config['loss']['main_weight'],
        physical_aux_weights=config['loss']['physical_aux_weights'],
        la_aux_weights=config['loss']['la_aux_weights'],
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['scheduler']['mode'],
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        min_lr=config['scheduler']['min_lr'],
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        log_dir=config['paths']['log_dir'],
        checkpoint_dir=config['paths']['save_dir'],
        gradient_clip_norm=config['training']['gradient_clip_norm'],
        debug=args.debug,
    )

    # Train
    print("\n" + "="*80)
    print("Starting training")
    print("="*80 + "\n")

    trainer.train(num_epochs=config['training']['num_epochs'])

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
