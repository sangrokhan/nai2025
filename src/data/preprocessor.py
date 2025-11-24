"""
Data Preprocessor

Handles data loading, feature extraction, and preprocessing for cellular network data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class CellularDataPreprocessor:
    """
    Preprocessor for cellular network data

    Handles:
    - Physical layer features: CQI (×4), SINR (×2), RI (×1)
    - Link adaptation features: MCS (4 layers × 2 MIMO types × 32 indices)
    - log1p transformation
    - Within-group normalization
    """

    def __init__(self):
        # Feature group definitions
        self.physical_groups = self._define_physical_groups()
        self.mcs_groups = self._define_mcs_groups()

        # Scalers for each group
        self.scalers: Dict[str, StandardScaler] = {}

        # Normalization statistics
        self.is_fitted = False

    def _define_physical_groups(self) -> Dict[str, List[str]]:
        """Define physical layer feature groups"""
        groups = {}

        # CQI groups (4 groups of 16 features each)
        for i in range(4):
            groups[f'cqi_{i}'] = [f'CQI_{i}_{j}' for j in range(16)]

        # SINR groups (2 groups of 20 features each)
        for i in range(2):
            groups[f'sinr_{i}'] = [f'SINR_{i}_{j}' for j in range(20)]

        # RI group (4 features)
        groups['ri'] = [f'RI_{i}' for i in range(4)]

        return groups

    def _define_mcs_groups(self) -> Dict[str, List[str]]:
        """Define MCS feature groups"""
        groups = {}

        layers = ['ONE_LAYER', 'TWO_LAYER', 'THREE_LAYER', 'FOUR_LAYER']
        mimo_types = ['SU_MIMO', 'MU_MIMO']

        for layer in layers:
            for mimo_type in mimo_types:
                # Key for dataset access (lowercase with underscores)
                key = f'mcs_{layer.lower()}_{mimo_type.lower()}'

                # Column names in actual data (uppercase with underscores)
                columns = [
                    f'MCS_{layer}_{mimo_type}_MCS{i}'
                    for i in range(32)
                ]
                groups[key] = columns

        return groups

    def fit(self, df: pd.DataFrame) -> 'CellularDataPreprocessor':
        """
        Fit preprocessor on training data

        Args:
            df: Training dataframe

        Returns:
            self
        """
        # Fit scalers for physical groups
        for group_name, columns in self.physical_groups.items():
            if all(col in df.columns for col in columns):
                scaler = StandardScaler()
                # Apply log1p before fitting
                data = np.log1p(df[columns].values)
                scaler.fit(data)
                self.scalers[group_name] = scaler

        # Fit scalers for MCS groups
        for group_name, columns in self.mcs_groups.items():
            if all(col in df.columns for col in columns):
                scaler = StandardScaler()
                # Apply log1p before fitting
                data = np.log1p(df[columns].values)
                scaler.fit(data)
                self.scalers[group_name] = scaler

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transform data using fitted scalers

        Args:
            df: Dataframe to transform

        Returns:
            Dictionary with preprocessed feature groups
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        processed = {}

        # Transform physical groups
        for group_name, columns in self.physical_groups.items():
            if group_name in self.scalers and all(col in df.columns for col in columns):
                data = np.log1p(df[columns].values)
                processed[group_name] = self.scalers[group_name].transform(data)

        # Transform MCS groups
        for group_name, columns in self.mcs_groups.items():
            if group_name in self.scalers and all(col in df.columns for col in columns):
                data = np.log1p(df[columns].values)
                processed[group_name] = self.scalers[group_name].transform(data)

        # Add context features (ue_count, prb_util) if present
        if 'UE_COUNT' in df.columns:
            processed['ue_count'] = df['UE_COUNT'].values.reshape(-1, 1)

        if 'PRB_UTILIZATION' in df.columns:
            processed['prb_util'] = df['PRB_UTILIZATION'].values.reshape(-1, 1)

        # Add target if present
        if 'THROUGHPUT' in df.columns:
            # Apply log transformation to target
            processed['throughput'] = np.log1p(df['THROUGHPUT'].values)

        # Add auxiliary targets if present
        if 'SPECTRAL_EFFICIENCY' in df.columns:
            processed['spectral_efficiency'] = df['SPECTRAL_EFFICIENCY'].values

        return processed

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fit and transform in one step

        Args:
            df: Training dataframe

        Returns:
            Dictionary with preprocessed feature groups
        """
        return self.fit(df).transform(df)

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get all feature groups"""
        return {**self.physical_groups, **self.mcs_groups}
