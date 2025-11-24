#!/usr/bin/env python3
"""
Generate Sample Test Data

Creates realistic sample data for testing without requiring actual cellular network data.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_data(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate sample cellular network data

    Features:
    - Physical Layer: CQI (16×4), SINR (20×2), RI (4)
    - Link Adaptation: MCS (32×8)
    - Context: UE count, PRB utilization
    - Target: Throughput

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all features
    """
    np.random.seed(seed)

    data = {}

    print(f"Generating {n_samples} samples...")

    # ========================================
    # Physical Layer Features
    # ========================================

    print("  - Physical layer features (CQI, SINR, RI)...")

    # CQI: Channel Quality Indicator (0-15, typically)
    # 4 groups of 16 features each
    for group in range(4):
        for idx in range(16):
            # CQI follows roughly exponential distribution (lower values more common)
            cqi_values = np.random.exponential(scale=5, size=n_samples)
            cqi_values = np.clip(cqi_values, 0, 15)  # CQI range: 0-15
            data[f'CQI_{group}_{idx}'] = cqi_values

    # SINR: Signal-to-Interference-plus-Noise Ratio (dB)
    # 2 groups of 20 features each
    for group in range(2):
        for idx in range(20):
            # SINR typically ranges from -5 to 30 dB
            sinr_values = np.random.normal(loc=15, scale=8, size=n_samples)
            sinr_values = np.clip(sinr_values, -5, 30)
            data[f'SINR_{group}_{idx}'] = sinr_values

    # RI: Rank Indicator (indicates number of spatial layers)
    # 4 features representing rank distribution
    for idx in range(4):
        # RI typically favors lower ranks
        ri_values = np.random.exponential(scale=2, size=n_samples)
        ri_values = np.clip(ri_values, 0, 10)
        data[f'RI_{idx}'] = ri_values

    # ========================================
    # Link Adaptation Features (MCS)
    # ========================================

    print("  - Link adaptation features (MCS statistics)...")

    # MCS: Modulation and Coding Scheme
    # 4 layers × 2 MIMO types × 32 indices
    layers = ['ONE_LAYER', 'TWO_LAYER', 'THREE_LAYER', 'FOUR_LAYER']
    mimo_types = ['SU_MIMO', 'MU_MIMO']

    for layer in layers:
        for mimo_type in mimo_types:
            # Each MCS index represents count of transmissions
            for mcs_idx in range(32):
                # Higher MCS indices (better conditions) are less common
                # Create distribution that favors mid-range MCS
                mean_mcs = 15
                std_mcs = 8

                # Adjust mean based on layer (more layers = better conditions)
                layer_num = layers.index(layer) + 1
                adjusted_mean = mean_mcs + layer_num * 2

                mcs_values = np.random.gamma(
                    shape=2,
                    scale=adjusted_mean / 2,
                    size=n_samples
                )
                mcs_values = np.clip(mcs_values, 0, 1000)

                data[f'MCS_{layer}_{mimo_type}_MCS{mcs_idx}'] = mcs_values

    # ========================================
    # Context Features
    # ========================================

    print("  - Context features (UE count, PRB utilization)...")

    # UE Count: Number of active users (1-50 typically)
    data['UE_COUNT'] = np.random.poisson(lam=15, size=n_samples)
    data['UE_COUNT'] = np.clip(data['UE_COUNT'], 1, 50)

    # PRB Utilization: Physical Resource Block utilization (0-1)
    data['PRB_UTILIZATION'] = np.random.beta(a=2, b=2, size=n_samples)

    # ========================================
    # Target: Throughput
    # ========================================

    print("  - Target (throughput)...")

    # Generate throughput based on features (with noise)
    # Simplified relationship: throughput depends on SINR, CQI, MCS, and UE count

    # Average SINR (positive impact)
    avg_sinr = np.mean([data[f'SINR_{g}_{i}'] for g in range(2) for i in range(20)], axis=0)

    # Average CQI (positive impact)
    avg_cqi = np.mean([data[f'CQI_{g}_{i}'] for g in range(4) for i in range(16)], axis=0)

    # Total MCS (positive impact)
    total_mcs = sum(
        data[f'MCS_{layer}_{mimo}_MCS{idx}']
        for layer in layers
        for mimo in mimo_types
        for idx in range(32)
    )

    # UE count (negative impact - more users means less per-user throughput)
    ue_count = data['UE_COUNT']

    # PRB utilization (positive impact up to a point)
    prb_util = data['PRB_UTILIZATION']

    # Combine into throughput (in bps, e.g., 1,000 - 500,000)
    base_throughput = 10000
    throughput = (
        base_throughput +
        avg_sinr * 1000 +
        avg_cqi * 2000 +
        total_mcs * 0.1 +
        prb_util * 50000 -
        ue_count * 500
    )

    # Add noise
    noise = np.random.normal(0, throughput * 0.1, size=n_samples)
    throughput = throughput + noise

    # Clip to reasonable range
    throughput = np.clip(throughput, 1000, 1_000_000)

    data['THROUGHPUT'] = throughput

    # ========================================
    # Auxiliary Targets (Optional)
    # ========================================

    print("  - Auxiliary targets...")

    # Spectral Efficiency: bits/s/Hz (related to SINR)
    # SE ≈ log2(1 + SINR_linear)
    sinr_linear = 10 ** (avg_sinr / 10)
    data['SPECTRAL_EFFICIENCY'] = np.log2(1 + sinr_linear)

    print(f"✓ Generated {n_samples} samples with {len(data)} features")

    return pd.DataFrame(data)


def main():
    """Generate sample datasets"""
    print("="*70)
    print("GENERATING SAMPLE TEST DATA")
    print("="*70)
    print()

    # Create data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Generate datasets
    datasets = {
        'train': 1000,      # Training set
        'val': 300,         # Validation set
        'test': 200,        # Test set
        'tiny': 50,         # Tiny set for quick tests
    }

    for name, n_samples in datasets.items():
        print(f"\nGenerating {name} dataset ({n_samples} samples)...")

        # Use different seeds for different splits
        seed = {'train': 42, 'val': 43, 'test': 44, 'tiny': 45}[name]

        df = generate_sample_data(n_samples, seed=seed)

        # Save as parquet
        output_path = data_dir / f"{name}.parquet"
        df.to_parquet(output_path, index=False)

        print(f"✓ Saved to {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Show sample statistics
        if name == 'train':
            print(f"\n  Sample statistics (first dataset):")
            print(f"    Throughput: {df['THROUGHPUT'].mean():.0f} ± {df['THROUGHPUT'].std():.0f} bps")
            print(f"    UE Count: {df['UE_COUNT'].mean():.1f} ± {df['UE_COUNT'].std():.1f}")
            print(f"    PRB Util: {df['PRB_UTILIZATION'].mean():.3f} ± {df['PRB_UTILIZATION'].std():.3f}")

    # Generate data schema documentation
    print("\n" + "="*70)
    print("GENERATING DATA SCHEMA DOCUMENTATION")
    print("="*70)

    schema_doc = """# Data Schema

## Overview

Sample test data mimics cellular network measurements with:
- **Physical Layer**: CQI, SINR, RI measurements
- **Link Adaptation**: MCS (Modulation and Coding Scheme) statistics
- **Context**: Network load indicators
- **Target**: Throughput to predict

## Feature Groups

### 1. Physical Layer Features (116 features)

#### CQI (Channel Quality Indicator) - 64 features
- 4 groups × 16 features each
- Columns: `CQI_{group}_{index}` where group ∈ [0,3], index ∈ [0,15]
- Range: [0, 15]
- Distribution: Exponential (lower values more common)

#### SINR (Signal-to-Interference-plus-Noise Ratio) - 40 features
- 2 groups × 20 features each
- Columns: `SINR_{group}_{index}` where group ∈ [0,1], index ∈ [0,19]
- Range: [-5, 30] dB
- Distribution: Normal (μ=15, σ=8)

#### RI (Rank Indicator) - 4 features
- 1 group × 4 features
- Columns: `RI_{index}` where index ∈ [0,3]
- Range: [0, 10]
- Distribution: Exponential (favors lower ranks)

### 2. Link Adaptation Features (256 features)

#### MCS Statistics - 256 features
- 4 layers × 2 MIMO types × 32 MCS indices
- Layers: ONE_LAYER, TWO_LAYER, THREE_LAYER, FOUR_LAYER
- MIMO: SU_MIMO (Single User), MU_MIMO (Multi User)
- Columns: `MCS_{layer}_{mimo_type}_MCS{index}`
- Range: [0, 1000] (transmission counts)
- Distribution: Gamma (favors mid-range MCS values)

### 3. Context Features (2 features)

#### UE_COUNT
- Number of active User Equipment (users)
- Range: [1, 50]
- Distribution: Poisson (λ=15)

#### PRB_UTILIZATION
- Physical Resource Block utilization ratio
- Range: [0, 1]
- Distribution: Beta (α=2, β=2)

### 4. Target Variable

#### THROUGHPUT
- Predicted throughput in bits per second
- Range: [1,000, 1,000,000] bps
- Depends on: SINR (↑), CQI (↑), MCS (↑), PRB util (↑), UE count (↓)

### 5. Auxiliary Targets

#### SPECTRAL_EFFICIENCY
- Spectral efficiency in bits/s/Hz
- Calculated as: log₂(1 + SINR_linear)
- Used for auxiliary task validation

## Dataset Splits

| Split | Samples | Seed | Purpose |
|-------|---------|------|---------|
| train | 1,000   | 42   | Model training |
| val   | 300     | 43   | Validation & early stopping |
| test  | 200     | 44   | Final evaluation |
| tiny  | 50      | 45   | Quick tests & debugging |

## File Format

- **Format**: Apache Parquet
- **Compression**: Snappy (default)
- **Index**: Not included (index=False)

## Usage Examples

### Load data
```python
import pandas as pd

# Load training data
train_df = pd.read_parquet('data/train.parquet')

# Check shape
print(f"Shape: {train_df.shape}")  # (1000, 375)

# Check features
print(train_df.columns.tolist())
```

### Access feature groups
```python
# CQI features
cqi_cols = [col for col in train_df.columns if col.startswith('CQI_')]
cqi_data = train_df[cqi_cols]

# MCS features
mcs_cols = [col for col in train_df.columns if col.startswith('MCS_')]
mcs_data = train_df[mcs_cols]
```

### Statistics
```python
# Target distribution
print(train_df['THROUGHPUT'].describe())

# Context features
print(train_df[['UE_COUNT', 'PRB_UTILIZATION']].describe())
```

## Notes

- Data is synthetic but follows realistic distributions
- Throughput has correlation with physical layer metrics
- MCS distributions vary by layer (more layers = better conditions)
- Suitable for testing but NOT for production models
- Use real cellular network data for actual deployment
"""

    schema_path = data_dir / "DATA_SCHEMA.md"
    with open(schema_path, 'w') as f:
        f.write(schema_doc)

    print(f"✓ Saved schema documentation to {schema_path}")

    print("\n" + "="*70)
    print("✓ ALL SAMPLE DATA GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGenerated files in {data_dir}:")
    for file in sorted(data_dir.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
