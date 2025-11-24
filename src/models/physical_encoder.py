"""
Physical Layer Encoder

Processes physical layer features (CQI, SINR, RI) and produces h_channel representation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class FeatureEncoder(nn.Module):
    """Encodes a single feature group into a fixed-size representation"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            encoded: [batch_size, output_dim]
        """
        return self.encoder(x)


class PhysicalLayerEncoder(nn.Module):
    """
    Physical Layer Encoder

    Processes physical layer features through multiple encoders and transformer.

    Input features:
    - CQI (×4 groups): 16 features each
    - SINR (×2 groups): 20 features each
    - RI (×1 group): 4 features

    Total: 116 features → h_channel [channel_dim]
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

        # Feature encoders for each group
        # CQI: 4 groups of 16 features
        self.cqi_encoders = nn.ModuleList([
            FeatureEncoder(16, channel_dim, dropout) for _ in range(4)
        ])

        # SINR: 2 groups of 20 features
        self.sinr_encoders = nn.ModuleList([
            FeatureEncoder(20, channel_dim, dropout) for _ in range(2)
        ])

        # RI: 1 group of 4 features
        self.ri_encoder = FeatureEncoder(4, channel_dim, dropout)

        # Transformer for attention across feature groups
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

        # Aggregation MLP
        self.aggregation = nn.Sequential(
            nn.Linear(channel_dim * 7, channel_dim * 2),  # 7 feature groups
            nn.LayerNorm(channel_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim * 2, channel_dim),
            nn.LayerNorm(channel_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through physical layer encoder

        Args:
            batch: Dictionary containing:
                - 'cqi_0', 'cqi_1', 'cqi_2', 'cqi_3': [B, 16] each
                - 'sinr_0', 'sinr_1': [B, 20] each
                - 'ri': [B, 4]

        Returns:
            h_channel: [B, channel_dim] - Physical layer representation
        """
        encoded_features = []

        # Encode CQI features
        for i, encoder in enumerate(self.cqi_encoders):
            cqi_encoded = encoder(batch[f'cqi_{i}'])
            encoded_features.append(cqi_encoded)

        # Encode SINR features
        for i, encoder in enumerate(self.sinr_encoders):
            sinr_encoded = encoder(batch[f'sinr_{i}'])
            encoded_features.append(sinr_encoded)

        # Encode RI features
        ri_encoded = self.ri_encoder(batch['ri'])
        encoded_features.append(ri_encoded)

        # Stack for transformer: [B, 7, channel_dim]
        feature_stack = torch.stack(encoded_features, dim=1)

        # Apply transformer attention
        transformed = self.transformer(feature_stack)  # [B, 7, channel_dim]

        # Flatten and aggregate
        flattened = transformed.flatten(1)  # [B, 7 * channel_dim]
        h_channel = self.aggregation(flattened)  # [B, channel_dim]

        return h_channel
