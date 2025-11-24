"""
Training Loop

Handles model training with logging, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

from .losses import HierarchicalLoss
from .logger import TensorBoardLogger
from ..utils.metrics import compute_metrics
from ..utils.early_stopping import EarlyStopping


class Trainer:
    """
    Training loop for hierarchical model

    Features:
    - TensorBoard logging
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Debug mode
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: HierarchicalLoss,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cuda',
        log_dir: str = './runs',
        checkpoint_dir: str = './checkpoints',
        gradient_clip_norm: float = 1.0,
        debug: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        self.debug = debug

        # Logger
        self.logger = TensorBoardLogger(log_dir)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=15,
            min_delta=1e-4,
            checkpoint_dir=checkpoint_dir,
        )

        # Tracking
        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}

        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch)
                losses = self.loss_fn(outputs, batch)

                # Backward pass
                self.optimizer.zero_grad()
                losses['total'].backward()

                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm,
                    )

                self.optimizer.step()

                # Accumulate losses
                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = []
                    epoch_losses[k].append(v.item())

                # Update progress bar
                pbar.set_postfix({'loss': losses['total'].item()})

                # Debug: print first batch details
                if self.debug and batch_idx == 0 and self.current_epoch == 0:
                    self._debug_batch(batch, outputs, losses)

                self.global_step += 1

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        return avg_losses

    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        epoch_losses = {}

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch)
                losses = self.loss_fn(outputs, batch)

                # Accumulate losses
                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = []
                    epoch_losses[k].append(v.item())

                # Collect predictions for metrics
                all_preds.append(outputs['throughput_pred'].cpu())
                all_targets.append(batch['throughput'].cpu())

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0).squeeze(-1)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = compute_metrics(
            all_targets.numpy(),
            all_preds.numpy(),
        )

        # Add metrics to losses dict
        avg_losses.update(metrics)

        return avg_losses

    def train(self, num_epochs: int):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Log
            self.logger.log_epoch(
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                learning_rate=self.optimizer.param_groups[0]['lr'],
            )

            # Print summary
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Val R²: {val_losses.get('r2', 0):.4f}")
            print(f"  Val MAPE: {val_losses.get('mape', 0):.2f}%")

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()

            # Early stopping
            should_stop = self.early_stopping(
                val_loss=val_losses['total'],
                model=self.model,
                epoch=epoch,
            )

            if should_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print("\nTraining completed!")
        print(f"Best model saved at: {self.early_stopping.best_model_path}")

        # Close logger
        self.logger.close()

    def _debug_batch(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
    ):
        """Print debug information for first batch"""
        print("\n" + "="*80)
        print("=== DEBUG: First Batch Information ===")
        print("="*80)

        print("\n--- Batch Shapes ---")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")

        print("\n--- Model Outputs ---")
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv.shape}")

        print("\n--- Loss Values ---")
        for k, v in losses.items():
            print(f"  {k}: {v.item():.6f}")

        print("\n--- Gradients ---")
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: {grad_norm:.6f}")

        print("\n" + "="*80)
