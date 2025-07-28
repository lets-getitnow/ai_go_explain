"""
Contextual Channel Detector
===========================
Purpose
-------
Analyze variance between baseline and variant activations to identify which
channels are sensitive to global inputs (komi, history, ko state, etc.) vs
pure board shape.

Loads paired activation files (e.g. baseline vs zero_global) and computes
per-channel variance statistics. Channels with high variance are flagged as
"contextual" and excluded from spatial NMF analysis.

High-Level Requirements
-----------------------
• Load baseline and variant activation matrices (same shape N×C)
• Compute per-channel variance metrics (coefficient of variation, etc.)
• Apply statistical thresholds to classify channels as spatial/contextual
• Output JSON mask mapping channel_id → {"spatial"|"contextual"}
• Zero-fallback mandate: fail fast on any unexpected data shapes

Usage
------
python3 contextual_channel_detector.py \
  --baseline activations_variants/pooled_rconv14.out__baseline.npy \
  --variant activations_variants/pooled_rconv14.out__zero_global.npy \
  --output contextual_mask.json \
  --threshold 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def load_activation_data(baseline_path: Path, variant_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load baseline and variant activation matrices."""
    print(f"[INFO] Loading baseline: {baseline_path}")
    baseline = np.load(baseline_path)
    print(f"[INFO] Loading variant: {variant_path}")
    variant = np.load(variant_path)
    
    if baseline.shape != variant.shape:
        raise ValueError(
            f"Shape mismatch: baseline {baseline.shape} vs variant {variant.shape}")
    
    print(f"[INFO] Loaded {baseline.shape[0]} positions × {baseline.shape[1]} channels")
    return baseline, variant


def compute_channel_variance(baseline: np.ndarray, variant: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-channel variance metrics between baseline and variant."""
    print("[INFO] Computing channel variance metrics...")
    
    # Basic statistics
    baseline_mean = np.mean(baseline, axis=0)
    variant_mean = np.mean(variant, axis=0)
    
    # Coefficient of variation (CV = std/mean) for each channel
    baseline_cv = np.std(baseline, axis=0) / (np.abs(baseline_mean) + 1e-8)
    variant_cv = np.std(variant, axis=0) / (np.abs(variant_mean) + 1e-8)
    
    # Mean absolute difference between baseline and variant
    mean_diff = np.mean(np.abs(baseline - variant), axis=0)
    
    # Relative change in mean activation
    relative_change = np.abs(variant_mean - baseline_mean) / (np.abs(baseline_mean) + 1e-8)
    
    # Kolmogorov-Smirnov test p-values (lower = more different distributions)
    ks_pvalues = np.array([
        stats.ks_2samp(baseline[:, i], variant[:, i])[1]
        for i in range(baseline.shape[1])
    ])
    
    return {
        "baseline_mean": baseline_mean,
        "variant_mean": variant_mean,
        "baseline_cv": baseline_cv,
        "variant_cv": variant_cv,
        "mean_diff": mean_diff,
        "relative_change": relative_change,
        "ks_pvalues": ks_pvalues,
    }


def classify_channels(variance_metrics: Dict[str, np.ndarray], threshold: float = 0.1) -> Dict[int, str]:
    """Classify channels as spatial or contextual based on variance metrics."""
    print(f"[INFO] Classifying channels with threshold {threshold}...")
    
    n_channels = len(variance_metrics["baseline_mean"])
    classifications = {}
    
    # Use relative change as primary metric
    relative_change = variance_metrics["relative_change"]
    
    for i in range(n_channels):
        if relative_change[i] > threshold:
            classifications[i] = "contextual"
        else:
            classifications[i] = "spatial"
    
    n_contextual = sum(1 for v in classifications.values() if v == "contextual")
    n_spatial = sum(1 for v in classifications.values() if v == "spatial")
    
    print(f"[INFO] Classified {n_contextual} contextual, {n_spatial} spatial channels")
    print(f"[INFO] Contextual percentage: {n_contextual/n_channels*100:.1f}%")
    
    return classifications


def save_contextual_mask(classifications: Dict[int, str], output_path: Path, 
                        variance_metrics: Dict[str, np.ndarray]) -> None:
    """Save contextual mask and variance statistics to JSON."""
    print(f"[INFO] Saving contextual mask to {output_path}")
    
    # Prepare detailed output
    output_data = {
        "channel_classifications": classifications,
        "summary": {
            "total_channels": len(classifications),
            "spatial_channels": sum(1 for v in classifications.values() if v == "spatial"),
            "contextual_channels": sum(1 for v in classifications.values() if v == "contextual"),
            "contextual_percentage": sum(1 for v in classifications.values() if v == "contextual") / len(classifications) * 100
        },
        "variance_statistics": {
            "relative_change_mean": float(np.mean(variance_metrics["relative_change"])),
            "relative_change_std": float(np.std(variance_metrics["relative_change"])),
            "relative_change_max": float(np.max(variance_metrics["relative_change"])),
            "mean_diff_mean": float(np.mean(variance_metrics["mean_diff"])),
            "ks_pvalues_mean": float(np.mean(variance_metrics["ks_pvalues"])),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved contextual mask with {output_data['summary']['contextual_channels']} contextual channels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect contextual channels from variant analysis")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline activation matrix (.npy)")
    parser.add_argument("--variant", required=True, type=Path, help="Variant activation matrix (.npy)")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON mask file")
    parser.add_argument("--threshold", type=float, default=0.1, help="Relative change threshold for classification")
    
    args = parser.parse_args()
    
    # Load data
    baseline, variant = load_activation_data(args.baseline, args.variant)
    
    # Compute variance metrics
    variance_metrics = compute_channel_variance(baseline, variant)
    
    # Classify channels
    classifications = classify_channels(variance_metrics, args.threshold)
    
    # Save results
    save_contextual_mask(classifications, args.output, variance_metrics)
    
    print("✅ Contextual channel detection completed")


if __name__ == "__main__":
    main() 