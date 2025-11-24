"""
PyTorch Dataset for Hierarchical Model
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Optional


class HierarchicalDataset(Dataset):
    """
    PyTorch Dataset for hierarchical cellular network model

    Expects preprocessed data from CellularDataPreprocessor with:
    - Physical features: cqi_0-3, sinr_0-1, ri
    - MCS features: mcs_one_layer_su_mimo, etc.
    - Context features: ue_count, prb_util
    - Target: throughput (log-transformed)
    """

    def __init__(
        self,
        processed_data: Dict[str, np.ndarray],
        include_targets: bool = True,
    ):
        """
        Initialize dataset

        Args:
            processed_data: Dictionary from CellularDataPreprocessor.transform()
            include_targets: Whether to include target variables
        """
        self.data = processed_data
        self.include_targets = include_targets

        # Validate required keys
        self._validate_data()

        # Get dataset size
        first_key = list(self.data.keys())[0]
        self.length = len(self.data[first_key])

    def _validate_data(self):
        """Validate that required keys are present"""
        required_physical = [f'cqi_{i}' for i in range(4)] + \
                           [f'sinr_{i}' for i in range(2)] + ['ri']

        required_mcs = []
        for layer in ['one_layer', 'two_layer', 'three_layer', 'four_layer']:
            for mimo in ['su_mimo', 'mu_mimo']:
                required_mcs.append(f'mcs_{layer}_{mimo}')

        required_context = ['ue_count', 'prb_util']

        # Check physical features
        for key in required_physical:
            if key not in self.data:
                raise ValueError(f"Missing required physical feature: {key}")

        # Check MCS features
        for key in required_mcs:
            if key not in self.data:
                raise ValueError(f"Missing required MCS feature: {key}")

        # Check context features
        for key in required_context:
            if key not in self.data:
                raise ValueError(f"Missing required context feature: {key}")

        # Check target if needed
        if self.include_targets and 'throughput' not in self.data:
            raise ValueError("Missing target: throughput")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Dictionary with all features and target (if included)
        """
        sample = {}

        # Physical features
        for i in range(4):
            sample[f'cqi_{i}'] = torch.from_numpy(
                self.data[f'cqi_{i}'][idx]
            ).float()

        for i in range(2):
            sample[f'sinr_{i}'] = torch.from_numpy(
                self.data[f'sinr_{i}'][idx]
            ).float()

        sample['ri'] = torch.from_numpy(self.data['ri'][idx]).float()

        # MCS features
        for layer in ['one_layer', 'two_layer', 'three_layer', 'four_layer']:
            for mimo in ['su_mimo', 'mu_mimo']:
                key = f'mcs_{layer}_{mimo}'
                sample[key] = torch.from_numpy(self.data[key][idx]).float()

        # Context features
        sample['ue_count'] = torch.from_numpy(
            self.data['ue_count'][idx]
        ).float()
        sample['prb_util'] = torch.from_numpy(
            self.data['prb_util'][idx]
        ).float()

        # Target
        if self.include_targets:
            sample['throughput'] = torch.tensor(
                self.data['throughput'][idx],
                dtype=torch.float32,
            )

            # Auxiliary targets if present
            if 'spectral_efficiency' in self.data:
                sample['spectral_efficiency'] = torch.tensor(
                    self.data['spectral_efficiency'][idx],
                    dtype=torch.float32,
                )

        return sample


def collate_fn(batch):
    """
    Custom collate function for DataLoader

    Args:
        batch: List of samples from __getitem__

    Returns:
        Dictionary with batched tensors
    """
    # Get all keys from first sample
    keys = batch[0].keys()

    # Stack each key
    collated = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        collated[key] = torch.stack(tensors, dim=0)

    return collated
