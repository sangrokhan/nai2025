"""
Link Adaptation Encoder

Processes MCS statistics with channel-aware modulation to produce h_LA representation.
"""

import torch
import torch.nn as nn
from typing import Dict


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for channel-aware processing"""

    def __init__(self, feature_dim: int, conditioning_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(conditioning_dim, feature_dim)
        self.beta_net = nn.Linear(conditioning_dim, feature_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, feature_dim] - Features to modulate
            conditioning: [B, conditioning_dim] - Conditioning signal (h_channel)
        Returns:
            modulated: [B, feature_dim]
        """
        gamma = self.gamma_net(conditioning)
        beta = self.beta_net(conditioning)
        return gamma * x + beta


class MCSEncoder(nn.Module):
    """Encodes MCS statistics for a single layer/MIMO combination"""

    def __init__(self, channel_dim: int, dropout: float = 0.1):
        super().__init__()

        # MCS has 32 indices
        self.encoder = nn.Sequential(
            nn.Linear(32, channel_dim),
            nn.LayerNorm(channel_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, channel_dim),
            nn.LayerNorm(channel_dim),
        )

        # Channel-aware modulation
        self.film = FiLMLayer(channel_dim, channel_dim)

    def forward(self, mcs_stats: torch.Tensor, h_channel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mcs_stats: [B, 32] - MCS statistics
            h_channel: [B, channel_dim] - Channel representation
        Returns:
            encoded: [B, channel_dim]
        """
        encoded = self.encoder(mcs_stats)
        modulated = self.film(encoded, h_channel)
        return modulated


class LinkAdaptationEncoder(nn.Module):
    """
    Link Adaptation Encoder

    Processes MCS statistics with channel-aware modulation.

    Input:
    - MCS statistics: 8 groups (4 layers × 2 MIMO types × 32 indices)
    - h_channel: Physical layer representation

    Output:
    - h_LA: [B, channel_dim] - Link adaptation representation
    """

    def __init__(
        self,
        channel_dim: int = 128,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.channel_dim = channel_dim

        # MCS encoders for each layer/MIMO combination
        # 4 layers × 2 MIMO types = 8 encoders
        self.mcs_encoders = nn.ModuleDict()

        layers = ['one_layer', 'two_layer', 'three_layer', 'four_layer']
        mimo_types = ['su_mimo', 'mu_mimo']

        for layer in layers:
            for mimo_type in mimo_types:
                key = f'mcs_{layer}_{mimo_type}'
                self.mcs_encoders[key] = MCSEncoder(channel_dim, dropout)

        # Transformer for integrating MCS encodings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channel_dim,
            nhead=num_attention_heads,
            dim_feedforward=channel_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # Fusion with h_channel
        self.fusion = nn.Sequential(
            nn.Linear(channel_dim * 2, channel_dim * 2),
            nn.LayerNorm(channel_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim * 2, channel_dim),
            nn.LayerNorm(channel_dim),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        h_channel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through link adaptation encoder

        Args:
            batch: Dictionary containing MCS statistics:
                - 'mcs_one_layer_su_mimo', 'mcs_one_layer_mu_mimo', etc.
                Each: [B, 32]
            h_channel: [B, channel_dim] - Physical layer representation

        Returns:
            h_LA: [B, channel_dim] - Link adaptation representation
        """
        encoded_mcs = []

        # Encode all MCS groups with channel-aware modulation
        for key, encoder in self.mcs_encoders.items():
            if key in batch:
                mcs_encoded = encoder(batch[key], h_channel)
                encoded_mcs.append(mcs_encoded)

        # Stack for transformer: [B, 8, channel_dim]
        mcs_stack = torch.stack(encoded_mcs, dim=1)

        # Apply transformer
        transformed = self.transformer(mcs_stack)  # [B, 8, channel_dim]

        # Pool across MCS encodings
        mcs_pooled = transformed.mean(dim=1)  # [B, channel_dim]

        # Fuse with h_channel
        combined = torch.cat([mcs_pooled, h_channel], dim=1)  # [B, 2*channel_dim]
        h_LA = self.fusion(combined)  # [B, channel_dim]

        return h_LA
