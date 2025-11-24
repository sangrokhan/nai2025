"""
Auxiliary Task Modules

Auxiliary prediction heads for validating representation quality.
"""

import torch
import torch.nn as nn
from typing import Dict


class AuxiliaryTasks(nn.Module):
    """
    Physical layer auxiliary tasks

    Validates h_channel representation quality through:
    1. Spectral Efficiency prediction
    2. RI distribution prediction
    3. Channel quality prediction
    """

    def __init__(self, channel_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # Spectral Efficiency predictor
        self.se_predictor = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim // 2, 1),
        )

        # RI distribution predictor (4 ranks)
        self.ri_dist_predictor = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim // 2, 4),
            nn.Softmax(dim=-1),
        )

        # Channel quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_channel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary predictions from h_channel

        Args:
            h_channel: [B, channel_dim]

        Returns:
            Dictionary containing:
                - 'se_pred': [B, 1] - Spectral efficiency
                - 'ri_dist_pred': [B, 4] - RI distribution
                - 'quality_pred': [B, 1] - Channel quality [0, 1]
        """
        return {
            'se_pred': self.se_predictor(h_channel),
            'ri_dist_pred': self.ri_dist_predictor(h_channel),
            'quality_pred': self.quality_predictor(h_channel),
        }


class LAAuxiliaryTasks(nn.Module):
    """
    Link adaptation auxiliary tasks

    Validates h_LA representation quality through:
    1. MCS average prediction
    2. SU/MU MIMO ratio prediction
    """

    def __init__(self, channel_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # MCS average predictor
        self.mcs_avg_predictor = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim // 2, 1),
        )

        # SU/MU ratio predictor
        self.su_ratio_predictor = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_LA: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute LA auxiliary predictions from h_LA

        Args:
            h_LA: [B, channel_dim]

        Returns:
            Dictionary containing:
                - 'mcs_avg_pred': [B, 1] - Average MCS index
                - 'su_ratio_pred': [B, 1] - SU MIMO ratio [0, 1]
        """
        return {
            'mcs_avg_pred': self.mcs_avg_predictor(h_LA),
            'su_ratio_pred': self.su_ratio_predictor(h_LA),
        }
