# Data Guide

## Overview

This project requires cellular network data with physical layer and link adaptation features. You can either:

1. **Use sample data** (for testing/learning)
2. **Use your own data** (for production)

## Option 1: Generate Sample Data (Recommended for Testing)

### Quick Start

```bash
# Using Makefile (recommended)
make generate-data

# Or directly
pip install numpy pandas pyarrow
python3 generate_sample_data.py
```

### What Gets Generated

```
data/
├── train.parquet          # 1,000 samples for training
├── val.parquet            # 300 samples for validation
├── test.parquet           # 200 samples for testing
├── tiny.parquet           # 50 samples for quick tests
├── DATA_SCHEMA.md         # Detailed schema documentation
└── README.md              # Data directory guide
```

### Sample Data Characteristics

- **375 features** per sample
- **Realistic distributions** (exponential, normal, gamma)
- **Correlated target** (throughput depends on features)
- **Reproducible** (fixed random seeds)
- **Compressed** (Parquet format, ~2-3 MB total)

### Testing Without Data Generation

Unit tests work without data files:

```bash
# These don't require data files
pytest tests/test_models.py -v
pytest tests/test_training.py -v

# These generate data internally
pytest tests/test_data.py -v
pytest tests/test_integration.py -v
```

## Option 2: Use Your Own Data

### Required Format

Your data must be a pandas DataFrame with these feature groups:

#### 1. Physical Layer Features (116 columns)

**CQI (64 columns):**
```
CQI_0_0, CQI_0_1, ..., CQI_0_15
CQI_1_0, CQI_1_1, ..., CQI_1_15
CQI_2_0, CQI_2_1, ..., CQI_2_15
CQI_3_0, CQI_3_1, ..., CQI_3_15
```

**SINR (40 columns):**
```
SINR_0_0, SINR_0_1, ..., SINR_0_19
SINR_1_0, SINR_1_1, ..., SINR_1_19
```

**RI (4 columns):**
```
RI_0, RI_1, RI_2, RI_3
```

#### 2. Link Adaptation Features (256 columns)

**MCS Statistics:**
```
MCS_ONE_LAYER_SU_MIMO_MCS0, ..., MCS_ONE_LAYER_SU_MIMO_MCS31
MCS_ONE_LAYER_MU_MIMO_MCS0, ..., MCS_ONE_LAYER_MU_MIMO_MCS31
MCS_TWO_LAYER_SU_MIMO_MCS0, ..., MCS_TWO_LAYER_SU_MIMO_MCS31
MCS_TWO_LAYER_MU_MIMO_MCS0, ..., MCS_TWO_LAYER_MU_MIMO_MCS31
MCS_THREE_LAYER_SU_MIMO_MCS0, ..., MCS_THREE_LAYER_SU_MIMO_MCS31
MCS_THREE_LAYER_MU_MIMO_MCS0, ..., MCS_THREE_LAYER_MU_MIMO_MCS31
MCS_FOUR_LAYER_SU_MIMO_MCS0, ..., MCS_FOUR_LAYER_SU_MIMO_MCS31
MCS_FOUR_LAYER_MU_MIMO_MCS0, ..., MCS_FOUR_LAYER_MU_MIMO_MCS31
```

#### 3. Context Features (2 columns)

```
UE_COUNT           # Number of active users
PRB_UTILIZATION    # PRB utilization ratio [0,1]
```

#### 4. Target (1 column)

```
THROUGHPUT         # Network throughput (target variable)
```

#### 5. Optional Auxiliary Targets

```
SPECTRAL_EFFICIENCY  # For auxiliary task training
```

### Preparing Your Data

```python
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')  # or from database, etc.

# Verify column names match expected format
required_cols = []

# Add CQI columns
for g in range(4):
    for i in range(16):
        required_cols.append(f'CQI_{g}_{i}')

# Add SINR columns
for g in range(2):
    for i in range(20):
        required_cols.append(f'SINR_{g}_{i}')

# Add RI columns
for i in range(4):
    required_cols.append(f'RI_{i}')

# Add MCS columns
layers = ['ONE_LAYER', 'TWO_LAYER', 'THREE_LAYER', 'FOUR_LAYER']
mimo_types = ['SU_MIMO', 'MU_MIMO']
for layer in layers:
    for mimo in mimo_types:
        for idx in range(32):
            required_cols.append(f'MCS_{layer}_{mimo}_MCS{idx}')

# Add context and target
required_cols.extend(['UE_COUNT', 'PRB_UTILIZATION', 'THROUGHPUT'])

# Check missing columns
missing = set(required_cols) - set(df.columns)
if missing:
    print(f"Missing columns: {missing}")

# Save as parquet
df.to_parquet('data/train.parquet', index=False)
```

### Data Splits

Create three datasets:

```python
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save
train_df.to_parquet('data/train.parquet', index=False)
val_df.to_parquet('data/val.parquet', index=False)
test_df.to_parquet('data/test.parquet', index=False)
```

## Data Validation

Check if your data is ready:

```bash
make check-data
```

Or manually:

```python
import pandas as pd
from src.data import CellularDataPreprocessor

# Load
df = pd.read_parquet('data/train.parquet')

# Validate
print(f"Shape: {df.shape}")
print(f"Expected: (n_samples, 375)")

# Try preprocessing
preprocessor = CellularDataPreprocessor()
processed = preprocessor.fit_transform(df)
print("✓ Data preprocessing successful!")
```

## Training with Your Data

Once data is ready:

```bash
# Make sure configs point to your data
# configs/medium_model.yaml:
#   paths:
#     train_data: ./data/train.parquet
#     val_data: ./data/val.parquet

# Train
make train-medium

# Or with custom config
python scripts/train.py --config configs/medium_model.yaml
```

## Data Statistics

Check data quality:

```python
import pandas as pd

df = pd.read_parquet('data/train.parquet')

# Target distribution
print("Throughput Statistics:")
print(df['THROUGHPUT'].describe())

# Feature ranges
print("\nFeature Ranges:")
print(f"CQI: [{df[[c for c in df.columns if 'CQI' in c]].min().min():.2f}, "
      f"{df[[c for c in df.columns if 'CQI' in c]].max().max():.2f}]")
print(f"SINR: [{df[[c for c in df.columns if 'SINR' in c]].min().min():.2f}, "
      f"{df[[c for c in df.columns if 'SINR' in c]].max().max():.2f}]")

# Missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")
```

## Common Issues

### Issue: Column Name Mismatch

**Error:** `KeyError: 'CQI_0_0'`

**Solution:** Check your column names match exactly:
```python
# Your columns
print(df.columns.tolist())

# Expected format
# CQI_0_0, not cqi_0_0 or CQI-0-0
```

### Issue: Wrong Data Types

**Error:** `TypeError: cannot convert float NaN to integer`

**Solution:** Handle missing values:
```python
df = df.fillna(0)  # or drop: df = df.dropna()
```

### Issue: Feature Range Problems

**Error:** Model training unstable

**Solution:** Check feature ranges are reasonable:
- CQI: [0, 15]
- SINR: [-10, 40] dB
- MCS counts: [0, few thousands]
- PRB_UTILIZATION: [0, 1]

### Issue: Data Too Large

**Error:** Out of memory

**Solution:** Use chunked loading or sample:
```python
# Sample data
df = pd.read_parquet('data/train.parquet')
df_sample = df.sample(n=10000, random_state=42)
df_sample.to_parquet('data/train_sample.parquet', index=False)
```

## Data Best Practices

1. **Always validate** before training
2. **Check distributions** match expected ranges
3. **Handle missing values** appropriately
4. **Split data properly** (train/val/test)
5. **Use Parquet format** for efficiency
6. **Document data source** and preprocessing
7. **Version your data** if possible
8. **Keep test set separate** until final evaluation

## Data Privacy

⚠️ **Important:**

- Do NOT commit real data to git
- Data files are in `.gitignore`
- Use sample data for public repos
- Secure real data appropriately
- Follow your organization's data policies

## Support

For data-related issues:

1. Check `DATA_SCHEMA.md` for detailed format
2. Run `make check-data` for validation
3. Try with sample data first
4. Check the test files for examples
5. Open an issue on GitHub

## References

- **Data Schema**: `data/DATA_SCHEMA.md`
- **Example Generation**: `generate_sample_data.py`
- **Preprocessor**: `src/data/preprocessor.py`
- **Dataset Class**: `src/data/dataset.py`
- **Tests**: `tests/test_data.py`
