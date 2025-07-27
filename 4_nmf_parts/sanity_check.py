#!/usr/bin/env python3
"""
NMF Parts Sanity Check

This script provides diagnostic metrics to assess the quality of Non-negative Matrix 
Factorization (NMF) results for Go position analysis. It helps determine whether the 
extracted parts are meaningful and identify potential preprocessing issues.

Requirements:
- NMF parts must be stored in nmf_components.npy
- Original pooled activations should be available for comparison
- Parts should be interpretable as Go board patterns

The script evaluates:
1. Part uniqueness - whether parts capture distinct patterns
2. Input data quality - checks for negative values that violate NMF assumptions  
3. Part sparsity - whether parts are focused or overly dense

Usage:
    python3 sanity_check.py

Output interpretation:
- Mean cosine distance > 0.3: Parts are sufficiently different
- Non-negative input: Data is suitable for NMF
- High sparsity: Parts are focused and interpretable
"""

import numpy as np
import os
import json
from sklearn.metrics import pairwise_distances

def load_nmf_data():
    """Load NMF parts and metadata."""
    parts_file = 'nmf_components.npy'
    activations_file = 'nmf_activations.npy'
    meta_file = 'nmf_meta.json'
    
    if not os.path.exists(parts_file):
        raise FileNotFoundError(f"Parts file not found: {parts_file}")
    
    W = np.load(parts_file)
    H = np.load(activations_file) if os.path.exists(activations_file) else None
    
    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
    
    return W, H, meta

def compute_part_uniqueness(W):
    """Compute mean pairwise cosine distance between parts."""
    # Ensure parts are rows
    if W.shape[0] < W.shape[1]:
        W_rank = W
    else:
        W_rank = W.T
    
    # L2-normalize each part vector
    norms = np.linalg.norm(W_rank, axis=1, keepdims=True)
    Wn = W_rank / np.where(norms == 0, 1, norms)
    
    # Compute cosine distances
    sim = Wn @ Wn.T
    n = sim.shape[0]
    dist = 1 - sim
    
    # Only upper-triangular without diagonal
    mean_dist = dist[np.triu_indices(n, k=1)].mean()
    min_dist = dist[np.triu_indices(n, k=1)].min()
    max_dist = dist[np.triu_indices(n, k=1)].max()
    
    return {
        'mean_distance': mean_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'n_parts': W_rank.shape[0],
        'feature_dim': W_rank.shape[1]
    }

def check_negativity(data_file):
    """Check if activations contain negative values."""
    if not os.path.exists(data_file):
        return None
    
    data = np.load(data_file)
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    
    return {
        'min_value': min_val,
        'max_value': max_val,
        'mean_value': mean_val,
        'has_negatives': min_val < 0,
        'shape': data.shape
    }

def main():
    print("NMF Parts Sanity Check")
    print("=" * 40)
    
    # Load data
    try:
        W, H, meta = load_nmf_data()
        print(f"✓ Loaded NMF data")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return
    
    # Part uniqueness
    print("\n1. Part Uniqueness:")
    uniqueness = compute_part_uniqueness(W)
    print(f"   Parts: {uniqueness['n_parts']} × {uniqueness['feature_dim']}")
    print(f"   Mean cosine distance: {uniqueness['mean_distance']:.4f}")
    print(f"   Distance range: [{uniqueness['min_distance']:.4f}, {uniqueness['max_distance']:.4f}]")
    
    if uniqueness['mean_distance'] > 0.3:
        print("   ✓ Parts differ enough to be interesting")
    else:
        print("   ⚠ Parts may be too similar - check preprocessing")
    
    # Check original activations for negativity
    print("\n2. Input Data Check:")
    pooled_file = '../3_extract_activations/activations/pooled_rconv14.out.npy'
    neg_check = check_negativity(pooled_file)
    
    if neg_check:
        print(f"   Original data shape: {neg_check['shape']}")
        print(f"   Value range: [{neg_check['min_value']:.4f}, {neg_check['max_value']:.4f}]")
        print(f"   Mean value: {neg_check['mean_value']:.4f}")
        
        if neg_check['has_negatives']:
            print("   ⚠ Contains negative values - consider ReLU preprocessing")
        else:
            print("   ✓ Non-negative input (good for NMF)")
    else:
        print("   ⚠ Could not check original data")
    
    # Part sparsity
    print("\n3. Part Sparsity:")
    if W.shape[0] < W.shape[1]:
        W_rank = W
    else:
        W_rank = W.T
    
    # Count non-zero elements per part
    sparsity = np.mean(W_rank == 0, axis=1)
    mean_sparsity = np.mean(sparsity)
    print(f"   Mean sparsity per part: {mean_sparsity:.3f}")
    print(f"   Sparsity range: [{np.min(sparsity):.3f}, {np.max(sparsity):.3f}]")
    
    if mean_sparsity > 0.5:
        print("   ✓ Parts are reasonably sparse")
    else:
        print("   ⚠ Parts are dense - consider sparsity constraints")

if __name__ == "__main__":
    main() 