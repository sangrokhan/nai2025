#!/usr/bin/env python
"""
Evaluation Script

Evaluates a trained model on test data.

Usage:
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/best_model.pt \
        --data ./data/test.parquet \
        --output ./evaluation_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import CellularDataPreprocessor, HierarchicalDataset
from src.data.dataset import collate_fn
from src.models import HierarchicalModel
from src.utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
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
        help='Path to test data (parquet)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./evaluation_results.json',
        help='Path to save results',
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
        help='Batch size for evaluation',
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
    test_df = pd.read_parquet(args.data)
    print(f"Test samples: {len(test_df):,}")

    # Preprocess
    print("Preprocessing data...")
    preprocessor = CellularDataPreprocessor()

    # We need to fit on some data - use test data itself
    # In production, use the same preprocessor that was used for training
    test_processed = preprocessor.fit_transform(test_df)

    # Create dataset
    test_dataset = HierarchicalDataset(test_processed, include_targets=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create model (need to know architecture)
    # For now, assume medium model defaults
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
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")

    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch)

            all_preds.append(outputs['throughput_pred'].cpu())
            all_targets.append(batch['throughput'].cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0).squeeze(-1).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Compute metrics
    metrics = compute_metrics(all_targets, all_preds)

    # Convert to original space for additional metrics
    preds_original = np.expm1(all_preds)
    targets_original = np.expm1(all_targets)

    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"MAE (log space): {metrics['mae']:.4f}")
    print(f"RMSE (log space): {metrics['rmse']:.4f}")
    print(f"MAPE (original space): {metrics['mape']:.2f}%")
    print(f"\nMean Target (original): {targets_original.mean():.2f}")
    print(f"Mean Prediction (original): {preds_original.mean():.2f}")
    print("="*80)

    # Save results
    results = {
        'checkpoint': str(args.checkpoint),
        'data': str(args.data),
        'n_samples': len(test_df),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'statistics': {
            'mean_target': float(targets_original.mean()),
            'mean_prediction': float(preds_original.mean()),
            'std_target': float(targets_original.std()),
            'std_prediction': float(preds_original.std()),
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
