# Cellular Network Optimization - Hierarchical Model

A hierarchical deep learning model for predicting cellular network throughput using physical layer and link adaptation features.

## Overview

This project implements a novel hierarchical architecture that processes cellular network data through multiple specialized encoders:

- **Physical Layer Encoder**: Processes CQI, SINR, and RI features
- **Link Adaptation Encoder**: Processes MCS statistics with channel-aware modulation
- **Auxiliary Tasks**: Validates representation quality at each layer
- **Throughput Predictor**: Combines hierarchical representations for final prediction

### Key Features

- Modular architecture with independent encoder modules
- Multi-task learning with auxiliary losses
- Attention-based feature integration
- Comprehensive testing and analysis tools
- Flexible configuration system

## Performance

Current performance on validation set (Medium Model):

- **R²**: 0.7713
- **MAPE**: 10.90%
- **Train/Val Gap**: ~11% (throughput only)

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/nai2025.git
cd nai2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Place your training and validation data in the `data/` directory:

```
data/
├── train.parquet
└── val.parquet
```

Data should include:
- Physical features: CQI, SINR, RI
- MCS statistics: all layer/MIMO combinations
- Context: UE count, PRB utilization
- Target: Throughput

### 2. Run Tests

Verify installation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_models.py -v
pytest tests/test_data.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### 3. Train Model

```bash
# Train with medium model (recommended)
python scripts/train.py --config configs/medium_model.yaml

# Train with debug mode
python scripts/train.py --config configs/medium_model.yaml --debug

# Train on CPU
python scripts/train.py --config configs/medium_model.yaml --device cpu
```

### 4. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=./runs

# Open browser to http://localhost:6006
```

### 5. Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/medium_model/best_model.pt \
    --data ./data/test.parquet \
    --output ./evaluation_results.json
```

### 6. Analyze Results

```bash
python scripts/analyze.py \
    --checkpoint ./checkpoints/medium_model/best_model.pt \
    --data ./data/val.parquet \
    --output-dir ./analysis_output
```

## Project Structure

```
nai2025/
├── src/                          # Source code
│   ├── models/                   # Model modules
│   │   ├── physical_encoder.py
│   │   ├── la_encoder.py
│   │   ├── auxiliary_tasks.py
│   │   └── hierarchical_model.py
│   ├── data/                     # Data processing
│   │   ├── preprocessor.py
│   │   └── dataset.py
│   ├── training/                 # Training components
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── logger.py
│   ├── analysis/                 # Analysis tools
│   │   └── analyzer.py
│   └── utils/                    # Utilities
│       ├── early_stopping.py
│       ├── config.py
│       └── metrics.py
├── tests/                        # Test suite
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_integration.py
├── configs/                      # Configuration files
│   ├── small_model.yaml
│   ├── medium_model.yaml
│   └── large_model.yaml
├── scripts/                      # Execution scripts
│   ├── train.py
│   ├── evaluate.py
│   └── analyze.py
├── data/                         # Data directory
├── checkpoints/                  # Model checkpoints
├── runs/                         # TensorBoard logs
├── analysis_output/              # Analysis results
├── requirements.txt
├── setup.py
├── claude.md                     # AI assistant guide
└── README.md
```

## Model Configurations

Three pre-configured model sizes are available:

### Small Model
- Fast training and debugging
- Channel dim: 64
- Parameters: ~500K
- GPU memory: ~2GB

```bash
python scripts/train.py --config configs/small_model.yaml
```

### Medium Model (Recommended)
- Best performance/cost balance
- Channel dim: 128
- Parameters: ~2M
- GPU memory: ~4GB

```bash
python scripts/train.py --config configs/medium_model.yaml
```

### Large Model
- Maximum performance
- Channel dim: 256
- Parameters: ~8M
- GPU memory: ~8GB

```bash
python scripts/train.py --config configs/large_model.yaml
```

## Architecture Details

### Physical Layer Encoder

Processes raw physical layer features:

```
Input: CQI (16×4), SINR (20×2), RI (4)
↓
Feature Encoders (7 groups)
↓
Transformer (multi-head attention)
↓
Output: h_channel [B, channel_dim]
```

### Link Adaptation Encoder

Processes MCS statistics with channel awareness:

```
Input: MCS (32×8 groups) + h_channel
↓
MCS Encoders with FiLM modulation
↓
Transformer integration
↓
Fusion with h_channel
↓
Output: h_LA [B, channel_dim]
```

### Throughput Predictor

Combines representations for final prediction:

```
Input: [h_channel, h_LA, ue_count, prb_util]
↓
Deep MLP (3-4 layers)
↓
Output: Throughput prediction
```

## Loss Function

Multi-task loss with auxiliary supervision:

```
L_total = L_throughput + α·L_physical_aux + β·L_la_aux
```

Where:
- **L_throughput**: MSE in log space + MAE in original space
- **L_physical_aux**: SE + RI distribution + Channel quality
- **L_la_aux**: MCS average + SU/MU ratio

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v

# Specific test function
pytest tests/test_training.py::test_auxiliary_loss_computation -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Style

```bash
# Format code
black src/ tests/ scripts/

# Check style
flake8 src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/
```

### Type Checking

```bash
mypy src/
```

## Troubleshooting

### Issue: Auxiliary Loss is Zero

Check debug output:
```bash
python scripts/train.py --config configs/medium_model.yaml --debug
```

Look for:
- Model outputs contain auxiliary predictions
- Loss computation shows non-zero values

### Issue: Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 64  # Reduce from 128
```

### Issue: Poor Performance

Try:
1. Increase regularization (dropout, weight decay)
2. Use data augmentation
3. Collect more training data
4. Ensemble multiple models

## Citation

If you use this code, please cite:

```bibtex
@software{cellular_network_optimization,
  title={Hierarchical Model for Cellular Network Throughput Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/nai2025}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact:
- Email: your.email@example.com
- GitHub: @yourusername

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Research community for inspiring architecture ideas
