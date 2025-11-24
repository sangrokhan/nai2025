"""
Early Stopping

Implements early stopping with patience and model checkpointing.
"""

import torch
import torch.nn as nn
from pathlib import Path


class EarlyStopping:
    """
    Early stopping with patience

    Monitors validation loss and stops training when no improvement
    is seen for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        checkpoint_dir: str = './checkpoints',
    ):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            checkpoint_dir: Directory to save best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_path = None

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        epoch: int,
    ) -> bool:
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss
            model: Model to checkpoint
            epoch: Current epoch

        Returns:
            True if training should stop, False otherwise
        """
        # Check if this is an improvement
        if val_loss < self.best_loss - self.min_delta:
            # Improved
            self.best_loss = val_loss
            self.counter = 0

            # Save best model
            self._save_checkpoint(model, epoch, val_loss)

            print(f"  Validation loss improved to {val_loss:.4f}, saving model")

        else:
            # No improvement
            self.counter += 1
            print(f"  No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                return True

        return False

    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        val_loss: float,
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }

        self.best_model_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, self.best_model_path)

        # Also save latest model
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
