"""
Hierarchical Model

Integrates physical layer and link adaptation encoders with auxiliary tasks
for throughput prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .physical_encoder import PhysicalLayerEncoder
from .la_encoder import LinkAdaptationEncoder
from .auxiliary_tasks import AuxiliaryTasks, LAAuxiliaryTasks


class ThroughputPredictor(nn.Module):
    """Predicts throughput from hierarchical representations"""

    def __init__(
        self,
        channel_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input: [h_channel, h_LA, ue_count, prb_util] = 2*channel_dim + 2
        input_dim = 2 * channel_dim + 2

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        # Final prediction layer
        layers.append(nn.Linear(in_dim, 1))

        self.predictor = nn.Sequential(*layers)

    def forward(
        self,
        h_channel: torch.Tensor,
        h_LA: torch.Tensor,
        ue_count: torch.Tensor,
        prb_util: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict throughput

        Args:
            h_channel: [B, channel_dim]
            h_LA: [B, channel_dim]
            ue_count: [B, 1]
            prb_util: [B, 1]

        Returns:
            throughput_pred: [B, 1]
        """
        combined = torch.cat([h_channel, h_LA, ue_count, prb_util], dim=1)
        return self.predictor(combined)


class HierarchicalModel(nn.Module):
    """
    Hierarchical Model for Cellular Network Throughput Prediction

    Architecture:
    1. Physical Layer Encoder: raw features → h_channel
    2. Link Adaptation Encoder: MCS + h_channel → h_LA
    3. Auxiliary Tasks: validate both h_channel and h_LA
    4. Throughput Predictor: [h_channel, h_LA, context] → throughput

    Args:
        channel_dim: Dimension of latent representations
        num_transformer_layers: Number of transformer layers in encoders
        num_attention_heads: Number of attention heads
        predictor_hidden_dim: Hidden dimension for throughput predictor
        predictor_num_layers: Number of layers in throughput predictor
        dropout: Dropout rate
    """

    def __init__(
        self,
        channel_dim: int = 128,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        predictor_hidden_dim: int = 256,
        predictor_num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.channel_dim = channel_dim

        # Physical layer encoder
        self.physical_encoder = PhysicalLayerEncoder(
            channel_dim=channel_dim,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

        # Link adaptation encoder
        self.la_encoder = LinkAdaptationEncoder(
            channel_dim=channel_dim,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

        # Auxiliary tasks
        self.aux_tasks = AuxiliaryTasks(channel_dim=channel_dim, dropout=dropout)
        self.la_aux_tasks = LAAuxiliaryTasks(channel_dim=channel_dim, dropout=dropout)

        # Throughput predictor
        self.throughput_predictor = ThroughputPredictor(
            channel_dim=channel_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            dropout=dropout,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical model

        Args:
            batch: Dictionary containing:
                Physical features:
                    - 'cqi_0', 'cqi_1', 'cqi_2', 'cqi_3': [B, 16] each
                    - 'sinr_0', 'sinr_1': [B, 20] each
                    - 'ri': [B, 4]
                MCS features:
                    - 'mcs_one_layer_su_mimo', etc.: [B, 32] each
                Context features:
                    - 'ue_count': [B, 1]
                    - 'prb_util': [B, 1]

        Returns:
            Dictionary containing:
                - 'throughput_pred': [B, 1] - Main prediction
                - 'h_channel': [B, channel_dim] - Physical representation
                - 'h_LA': [B, channel_dim] - LA representation
                - 'physical_aux': Dict of physical auxiliary predictions
                - 'la_aux': Dict of LA auxiliary predictions
        """
        # 1. Physical layer encoding
        h_channel = self.physical_encoder(batch)

        # 2. Link adaptation encoding
        h_LA = self.la_encoder(batch, h_channel)

        # 3. Auxiliary predictions
        physical_aux = self.aux_tasks(h_channel)
        la_aux = self.la_aux_tasks(h_LA)

        # 4. Throughput prediction
        throughput_pred = self.throughput_predictor(
            h_channel,
            h_LA,
            batch['ue_count'],
            batch['prb_util'],
        )

        return {
            'throughput_pred': throughput_pred,
            'h_channel': h_channel,
            'h_LA': h_LA,
            'physical_aux': physical_aux,
            'la_aux': la_aux,
        }

    def get_representations(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract representations without predictions (for analysis)

        Args:
            batch: Input batch

        Returns:
            h_channel: [B, channel_dim]
            h_LA: [B, channel_dim]
        """
        with torch.no_grad():
            h_channel = self.physical_encoder(batch)
            h_LA = self.la_encoder(batch, h_channel)
        return h_channel, h_LA
