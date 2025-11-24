"""
TensorBoard Logger

Handles logging of training metrics to TensorBoard.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional


class TensorBoardLogger:
    """
    TensorBoard logger for training metrics

    Logs:
    - Scalar metrics (losses, metrics, learning rate)
    - Histograms (optional: gradients, weights)
    - Embeddings (optional: h_channel, h_LA)
    """

    def __init__(self, log_dir: str = './runs'):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def log_epoch(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        learning_rate: float,
    ):
        """
        Log metrics for one epoch

        Args:
            epoch: Current epoch number
            train_losses: Dictionary of training losses
            val_losses: Dictionary of validation losses
            learning_rate: Current learning rate
        """
        # Log training losses
        for name, value in train_losses.items():
            self.writer.add_scalar(f'train/{name}', value, epoch)

        # Log validation losses and metrics
        for name, value in val_losses.items():
            self.writer.add_scalar(f'val/{name}', value, epoch)

        # Log learning rate
        self.writer.add_scalar('train/learning_rate', learning_rate, epoch)

    def log_histograms(
        self,
        model: torch.nn.Module,
        epoch: int,
    ):
        """
        Log weight and gradient histograms

        Args:
            model: Model to log
            epoch: Current epoch
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log weights
                self.writer.add_histogram(
                    f'weights/{name}',
                    param.data.cpu(),
                    epoch,
                )

                # Log gradients if available
                if param.grad is not None:
                    self.writer.add_histogram(
                        f'gradients/{name}',
                        param.grad.cpu(),
                        epoch,
                    )

    def log_embeddings(
        self,
        h_channel: torch.Tensor,
        h_LA: torch.Tensor,
        epoch: int,
        metadata: Optional[list] = None,
    ):
        """
        Log embeddings for visualization

        Args:
            h_channel: [N, channel_dim] - Physical representations
            h_LA: [N, channel_dim] - LA representations
            epoch: Current epoch
            metadata: Optional metadata for each sample
        """
        # Log h_channel
        self.writer.add_embedding(
            h_channel,
            metadata=metadata,
            tag='h_channel',
            global_step=epoch,
        )

        # Log h_LA
        self.writer.add_embedding(
            h_LA,
            metadata=metadata,
            tag='h_LA',
            global_step=epoch,
        )

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value"""
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Close the logger"""
        self.writer.close()
