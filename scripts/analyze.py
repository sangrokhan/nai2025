#!/usr/bin/env python
"""
Analysis Script

Analyzes model representations and generates visualizations.

Usage:
    python scripts/analyze.py \
        --checkpoint ./checkpoints/best_model.pt \
        --data ./data/val.parquet \
        --output-dir ./analysis_output
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data import CellularDataPreprocessor, HierarchicalDataset
from src.data.dataset import collate_fn
from src.models import HierarchicalModel
from src.analysis.analyzer import HierarchicalModelAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze model representations')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data for analysis (parquet)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./analysis_output',
        help='Directory to save analysis outputs',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to analyze',
    )

    args = parser.parse_args()

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)

    if args.max_samples is not None:
        df = df.iloc[:args.max_samples]

    print(f"Analyzing {len(df):,} samples")

    # Preprocess
    print("Preprocessing data...")
    preprocessor = CellularDataPreprocessor()
    processed = preprocessor.fit_transform(df)

    # Create dataset
    dataset = HierarchicalDataset(processed, include_targets=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create model
    print("\nCreating model...")
    model = HierarchicalModel(
        channel_dim=128,
        num_transformer_layers=2,
        num_attention_heads=8,
        predictor_hidden_dim=256,
        predictor_num_layers=3,
        dropout=0.15,
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded from epoch {checkpoint['epoch']}")

    # Create analyzer
    analyzer = HierarchicalModelAnalyzer(model, device=device)

    # Run analysis
    print("\n" + "="*80)
    print("Running Analysis")
    print("="*80 + "\n")

    analyzer.analyze(dataloader, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
