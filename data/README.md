# Data Directory

## Quick Start

To generate sample test data:

```bash
# Install dependencies first
pip install numpy pandas pyarrow

# Generate sample data
python3 generate_sample_data.py
```

This will create:
- `train.parquet` (1,000 samples) - Training data
- `val.parquet` (300 samples) - Validation data
- `test.parquet` (200 samples) - Test data
- `tiny.parquet` (50 samples) - Quick testing
- `DATA_SCHEMA.md` - Data structure documentation

## Using Makefile

```bash
# Generate sample data (installs deps if needed)
make generate-data
```

## Data Structure

The sample data includes:

### Physical Layer (116 features)
- **CQI**: 4 groups × 16 features = 64 features
- **SINR**: 2 groups × 20 features = 40 features
- **RI**: 4 features

### Link Adaptation (256 features)
- **MCS**: 4 layers × 2 MIMO × 32 indices = 256 features

### Context (2 features)
- **UE_COUNT**: Number of active users
- **PRB_UTILIZATION**: Resource block utilization

### Target
- **THROUGHPUT**: Target variable to predict

### Total: 375 columns

## Using Your Own Data

Replace the sample `.parquet` files with your own data following the same structure:

```python
import pandas as pd

# Your data should have these columns
required_columns = [
    # CQI
    'CQI_0_0', 'CQI_0_1', ..., 'CQI_3_15',  # 64 columns
    # SINR
    'SINR_0_0', 'SINR_0_1', ..., 'SINR_1_19',  # 40 columns
    # RI
    'RI_0', 'RI_1', 'RI_2', 'RI_3',  # 4 columns
    # MCS
    'MCS_ONE_LAYER_SU_MIMO_MCS0', ...,  # 256 columns
    # Context
    'UE_COUNT', 'PRB_UTILIZATION',
    # Target
    'THROUGHPUT',
]

# Save as parquet
your_df.to_parquet('data/train.parquet', index=False)
```

## File Format

- **Format**: Apache Parquet (efficient columnar storage)
- **Compression**: Snappy (default)
- **No index**: `index=False` when saving

## Data Generation Details

The `generate_sample_data.py` script creates synthetic data with:

1. **Realistic distributions**:
   - CQI: Exponential (lower values more common)
   - SINR: Normal (μ=15 dB, σ=8 dB)
   - RI: Exponential (favors lower ranks)
   - MCS: Gamma (favors mid-range values)

2. **Correlated throughput**:
   - Higher SINR → Higher throughput
   - Higher CQI → Higher throughput
   - More users → Lower per-user throughput
   - Higher PRB utilization → Higher throughput

3. **Reproducible**:
   - Fixed random seeds per dataset split
   - train=42, val=43, test=44, tiny=45

## Troubleshooting

### "Module not found" errors

Install dependencies:
```bash
pip install numpy pandas pyarrow
```

### Large file sizes

Parquet files are compressed. For reference:
- train.parquet (~1000 samples): ~2-3 MB
- val.parquet (~300 samples): ~0.7-1 MB
- test.parquet (~200 samples): ~0.5 MB

### Loading data in code

```python
import pandas as pd
from src.data import CellularDataPreprocessor

# Load data
df = pd.read_parquet('data/train.parquet')

# Preprocess
preprocessor = CellularDataPreprocessor()
processed = preprocessor.fit_transform(df)

# Use in model...
```

## Notes

⚠️ **Important**: The sample data is synthetic and for testing only.

For production use:
- Use real cellular network measurements
- Validate data quality
- Check feature distributions
- Adjust model configuration as needed
