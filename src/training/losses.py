"""
Loss Functions

Implements main throughput loss and auxiliary losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def compute_auxiliary_losses(
    aux_outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    Compute physical layer auxiliary losses

    Args:
        aux_outputs: Dictionary from AuxiliaryTasks.forward()
            - 'se_pred': [B, 1]
            - 'ri_dist_pred': [B, 4]
            - 'quality_pred': [B, 1]
        batch: Batch data containing targets
        weights: Loss weights for each auxiliary task

    Returns:
        Dictionary with individual and total auxiliary losses
    """
    losses = {}

    # 1. Spectral Efficiency loss (if target available)
    if 'spectral_efficiency' in batch:
        se_target = batch['spectral_efficiency'].unsqueeze(-1)
        losses['se'] = F.mse_loss(aux_outputs['se_pred'], se_target)
    else:
        # Compute from SINR if not available
        # SE ≈ log2(1 + SINR)
        sinr_avg = (batch['sinr_0'].mean(dim=1) + batch['sinr_1'].mean(dim=1)) / 2
        se_target = torch.log2(1 + sinr_avg.unsqueeze(-1))
        losses['se'] = F.mse_loss(aux_outputs['se_pred'], se_target)

    # 2. RI distribution loss
    # Compute RI distribution from RI features
    ri_values = batch['ri']  # [B, 4]
    ri_dist_target = F.softmax(ri_values, dim=-1)
    losses['ri_dist'] = F.kl_div(
        torch.log(aux_outputs['ri_dist_pred'] + 1e-8),
        ri_dist_target,
        reduction='batchmean',
    )

    # 3. Channel quality loss
    # Compute from CQI weighted average (normalized to [0, 1])
    cqi_avg = sum(batch[f'cqi_{i}'].mean(dim=1) for i in range(4)) / 4
    quality_target = torch.sigmoid(cqi_avg.unsqueeze(-1))
    losses['quality'] = F.mse_loss(aux_outputs['quality_pred'], quality_target)

    # Weighted total
    total = sum(weights.get(k, 0.0) * v for k, v in losses.items())
    losses['total'] = total

    return losses


def compute_la_auxiliary_losses(
    la_aux_outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    Compute link adaptation auxiliary losses

    Args:
        la_aux_outputs: Dictionary from LAAuxiliaryTasks.forward()
            - 'mcs_avg_pred': [B, 1]
            - 'su_ratio_pred': [B, 1]
        batch: Batch data
        weights: Loss weights for each auxiliary task

    Returns:
        Dictionary with individual and total auxiliary losses
    """
    losses = {}

    # 1. MCS average loss
    # Compute weighted average across all MCS statistics
    mcs_sum = 0.0
    mcs_count = 0

    for layer in ['one_layer', 'two_layer', 'three_layer', 'four_layer']:
        for mimo in ['su_mimo', 'mu_mimo']:
            key = f'mcs_{layer}_{mimo}'
            if key in batch:
                # MCS indices weighted by their counts
                mcs_indices = torch.arange(32, device=batch[key].device).float()
                weighted_avg = (batch[key] * mcs_indices).sum(dim=1) / (batch[key].sum(dim=1) + 1e-8)
                mcs_sum += weighted_avg
                mcs_count += 1

    mcs_avg_target = (mcs_sum / mcs_count).unsqueeze(-1)
    losses['mcs_avg'] = F.mse_loss(la_aux_outputs['mcs_avg_pred'], mcs_avg_target)

    # 2. SU/MU ratio loss
    # Compute SU ratio: SU_total / (SU_total + MU_total)
    su_total = 0.0
    mu_total = 0.0

    for layer in ['one_layer', 'two_layer', 'three_layer', 'four_layer']:
        su_key = f'mcs_{layer}_su_mimo'
        mu_key = f'mcs_{layer}_mu_mimo'

        if su_key in batch:
            su_total += batch[su_key].sum(dim=1)
        if mu_key in batch:
            mu_total += batch[mu_key].sum(dim=1)

    su_ratio_target = su_total / (su_total + mu_total + 1e-8)
    losses['su_ratio'] = F.mse_loss(
        la_aux_outputs['su_ratio_pred'].squeeze(-1),
        su_ratio_target,
    )

    # Weighted total
    total = sum(weights.get(k, 0.0) * v for k, v in losses.items())
    losses['total'] = total

    return losses


class HierarchicalLoss(nn.Module):
    """
    Total loss for hierarchical model

    L_total = L_throughput + α·L_physical_aux + β·L_la_aux

    where:
    - L_throughput: MSE in log space + MAE in original space
    - L_physical_aux: Physical layer auxiliary losses
    - L_la_aux: Link adaptation auxiliary losses
    """

    def __init__(
        self,
        main_weight: float = 1.0,
        physical_aux_weights: Dict[str, float] = None,
        la_aux_weights: Dict[str, float] = None,
    ):
        super().__init__()

        self.main_weight = main_weight

        # Default auxiliary weights
        self.physical_aux_weights = physical_aux_weights or {
            'se': 0.25,
            'ri_dist': 0.15,
            'quality': 0.15,
        }

        self.la_aux_weights = la_aux_weights or {
            'mcs_avg': 0.15,
            'su_ratio': 0.08,
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss

        Args:
            outputs: Model outputs from HierarchicalModel.forward()
            batch: Batch data including targets

        Returns:
            Dictionary with all loss components
        """
        losses = {}

        # 1. Main throughput loss (log space MSE)
        throughput_pred = outputs['throughput_pred'].squeeze(-1)
        throughput_target = batch['throughput']

        losses['throughput_mse'] = F.mse_loss(throughput_pred, throughput_target)

        # MAE in original space (for monitoring)
        with torch.no_grad():
            pred_original = torch.expm1(throughput_pred)
            target_original = torch.expm1(throughput_target)
            losses['throughput_mae'] = F.l1_loss(pred_original, target_original)

        # 2. Physical auxiliary losses
        if 'physical_aux' in outputs:
            physical_aux_losses = compute_auxiliary_losses(
                outputs['physical_aux'],
                batch,
                self.physical_aux_weights,
            )
            losses['physical_se'] = physical_aux_losses['se']
            losses['physical_ri_dist'] = physical_aux_losses['ri_dist']
            losses['physical_quality'] = physical_aux_losses['quality']
            losses['physical_total_aux'] = physical_aux_losses['total']

        # 3. LA auxiliary losses
        if 'la_aux' in outputs:
            la_aux_losses = compute_la_auxiliary_losses(
                outputs['la_aux'],
                batch,
                self.la_aux_weights,
            )
            losses['la_mcs_avg'] = la_aux_losses['mcs_avg']
            losses['la_su_ratio'] = la_aux_losses['su_ratio']
            losses['la_total_aux'] = la_aux_losses['total']

        # 4. Total loss
        total_loss = self.main_weight * losses['throughput_mse']

        if 'physical_total_aux' in losses:
            total_loss += losses['physical_total_aux']

        if 'la_total_aux' in losses:
            total_loss += losses['la_total_aux']

        losses['total'] = total_loss

        return losses
