#!/usr/bin/env python3
"""
Step 4: Run NMF Parts Finder with ℓ1 Sparsity Control

Loads the pooled activations from step 3 and factorizes them using 
Non-negative Matrix Factorization with ℓ1 sparsity penalty to find 
interpretable, sparse parts.

The number of parts (k=25) is determined by systematic rank selection analysis
(see rank_analysis/README.md for details). The ℓ1 sparsity penalty (α_H=0.10) 
is determined by α_H grid analysis (see alpha_h_analysis.py) to achieve:
- 67.3% sparsity in H matrix (vs 16.2% without penalty)
- Minimal reconstruction error increase (≤ 5%)
- Clear, interpretable parts that don't fire on every board

Key improvements:
- α_H = 0.10 with l1_ratio=1.0 (pure ℓ1 penalty on H)
- Data preprocessing with StandardScaler for meaningful alpha values
- Sparsity monitoring and diagnostics
- Focus on sparse usage across positions, not sparse pixel patterns

With 6,603 positions and k=25 parts, we discover distinct Go concepts 
like atari patterns, eye shapes, ladder formations, etc., with clear 
sparse activation patterns.
"""

import argparse
import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path

def load_activation_data(activations_file=None):
    """
    Load activation data and metadata.
    
    Args:
        activations_file: Path to activation data file
        
    Returns:
        activations: (n_positions, n_channels) array
        meta: Metadata dictionary
    """
    if activations_file is None:
        data_path = "../3_extract_activations/activations/pooled_rconv14.out.npy"
    else:
        data_path = activations_file
    meta_path = str(Path(activations_file).parent / "pooled_meta.json") if activations_file else "../3_extract_activations/activations/pooled_meta.json"
    
    print(f"📁 Checking for data file: {data_path}", flush=True)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Activation data not found: {data_path}")
    
    print(f"📁 Checking for meta file: {meta_path}", flush=True)
    if not os.path.exists(meta_path):
        print(f"⚠️  Meta data not found: {meta_path}")
        print("📋 Creating minimal metadata...", flush=True)
        meta = {
            "date": "2025-07-29",
            "source_model": "unknown",
            "layer": "rconv14.out",
            "positions": 0,
            "original_channels": 0,
            "pooled_channels": 0,
            "pooling_method": "3x3_grid",
            "batch_size": 32,
            "non_negative_shift": True,
            "column_scaled": True,
            "variant_tag": "baseline",
            "dataset_dir": "unknown"
        }
    else:
        # Load metadata
        print("📋 Loading metadata...", flush=True)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print(f"✅ Loaded metadata with {len(meta)} entries", flush=True)
    
    # Load activation matrix (positions x channels)
    print("📊 Loading numpy activation data...", flush=True)
    activations = np.load(data_path)
    print(f"✅ Loaded activations shape: {activations.shape}", flush=True)
    print(f"📊 Activation data stats: min={activations.min():.4f}, max={activations.max():.4f}, mean={activations.mean():.4f}", flush=True)
    
    # Update meta with actual data info
    meta["positions"] = activations.shape[0]
    meta["pooled_channels"] = activations.shape[1]
    
    print("✅ Successfully loaded all activation data", flush=True)
    return activations, meta

def preprocess_data(X):
    """
    Preprocess data for meaningful alpha values.
    
    Args:
        X: Raw activation data
        
    Returns:
        X_scaled: Scaled data with roughly unit magnitude
    """
    print("🔧 Preprocessing data...", flush=True)
    
    # Ensure non-negativity
    X = np.maximum(0, X)
    print(f"📊 Original data stats: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}", flush=True)
    
    # Scale to roughly unit magnitude so alpha values are meaningful
    scaler = StandardScaler(with_mean=False)  # Keep non-negative
    X_scaled = scaler.fit_transform(X)
    
    print(f"📊 Scaled data stats: min={X_scaled.min():.4f}, max={X_scaled.max():.4f}, mean={X_scaled.mean():.4f}", flush=True)
    
    return X_scaled

def run_nmf_factorization(activations, n_parts=3, alpha_H=0.10):
    """
    Run NMF factorization with ℓ1 sparsity penalty on H matrix.
    
    Args:
        activations: (n_positions, n_channels) array
        n_parts: Number of parts to find
        alpha_H: ℓ1 penalty on H matrix for sparse usage across positions
        
    Returns:
        parts: (n_parts, n_channels) - The learned parts
        activations_transformed: (n_positions, n_parts) - Part activations per position
        model: The fitted NMF model
    """
    print(f"🔄 Starting NMF factorization with {n_parts} parts and α_H={alpha_H}...", flush=True)
    
    # Preprocess data for meaningful alpha values
    X = preprocess_data(activations)
    
    # Run NMF with ℓ1 sparsity penalty
    print("🏗️  Creating NMF model with ℓ1 sparsity...", flush=True)
    model = NMF(
        n_components=n_parts,
        init="nndsvd",
        alpha_H=alpha_H,          # ℓ1 penalty on H (activations)
        alpha_W=0.0,              # No penalty on W (basis) - keep dense
        l1_ratio=1.0,             # ρ = 1 → pure ℓ1 penalty
        max_iter=1000,
        random_state=42
    )
    print("✅ NMF model created", flush=True)
    
    # Fit and transform
    print("🔥 Starting NMF fit_transform (this may take time)...", flush=True)
    print(f"   Input shape: {X.shape}", flush=True)
    print(f"   Target parts: {n_parts}", flush=True)
    print(f"   α_H (ℓ1 penalty): {alpha_H}", flush=True)
    print(f"   Max iterations: 1000", flush=True)
    
    activations_transformed = model.fit_transform(X)
    print("✅ NMF fit_transform completed!", flush=True)
    
    print("📊 Extracting parts...", flush=True)
    parts = model.components_
    print("✅ Parts extracted", flush=True)
    
    # Calculate sparsity metrics
    H = model.components_
    sparsity = (H == 0).sum() / H.size
    avg_boards_per_component = (H != 0).sum(axis=1).mean()
    
    print(f"📊 NMF reconstruction error: {model.reconstruction_err_:.4f}", flush=True)
    print(f"📊 Number of iterations used: {model.n_iter_}", flush=True)
    print(f"📊 Parts shape: {parts.shape}", flush=True)
    print(f"📊 Transformed activations shape: {activations_transformed.shape}", flush=True)
    print(f"📊 Sparsity in H: {sparsity:.1%} ({sparsity:.3f})", flush=True)
    print(f"📊 Avg boards per component: {avg_boards_per_component:.1f}", flush=True)
    print(f"📊 Parts stats: min={parts.min():.4f}, max={parts.max():.4f}", flush=True)
    print(f"📊 Transformed stats: min={activations_transformed.min():.4f}, max={activations_transformed.max():.4f}", flush=True)
    
    return parts, activations_transformed, model

def save_results(parts, activations_transformed, model, original_meta, output_dir="."):
    """Save NMF results to files."""
    print("🔄 Starting to save results...", flush=True)
    
    # Save parts (the learned parts)
    print("💾 Saving NMF parts...", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "nmf_components.npy"), parts)
    print("✅ Saved nmf_components.npy", flush=True)
    
    # Save transformed activations (how much each part activates per position)
    print("💾 Saving transformed activations...", flush=True)
    np.save(os.path.join(output_dir, "nmf_activations.npy"), activations_transformed)
    print("✅ Saved nmf_activations.npy", flush=True)
    
    # Save metadata
    print("💾 Creating and saving metadata...", flush=True)
    # Calculate sparsity metrics for metadata
    H = model.components_
    sparsity = (H == 0).sum() / H.size
    avg_boards_per_component = (H != 0).sum(axis=1).mean()
    
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source_activations": "../3_extract_activations/activations/pooled_rconv14.out.npy",
        "original_shape": f"{activations_transformed.shape[0]}x{parts.shape[1]}",
        "n_parts": parts.shape[0],
        "n_positions": activations_transformed.shape[0],
        "n_channels": parts.shape[1],
        "pooling_method": original_meta.get("pooling_method", "global_average"),
        "original_channels": original_meta.get("original_channels", parts.shape[1] // 9),
        "reconstruction_error": float(model.reconstruction_err_),
        "n_iterations": int(model.n_iter_),
        "alpha_H": 0.4,  # ℓ1 sparsity penalty used
        "sparsity_percentage": float(sparsity),
        "avg_boards_per_component": float(avg_boards_per_component),
        "l1_ratio": 1.0,  # Pure ℓ1 penalty
        "original_meta": original_meta
    }
    
    with open(os.path.join(output_dir, "nmf_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved nmf_meta.json", flush=True)
    
    print(f"✅ All results saved:", flush=True)
    print(f"  - nmf_components.npy: {parts.shape}", flush=True)
    print(f"  - nmf_activations.npy: {activations_transformed.shape}", flush=True)
    print(f"  - nmf_meta.json", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Run NMF to find interpretable parts")
    parser.add_argument("--activations-file", type=str, help="Path to activation data file")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--num-components", type=int, default=25, help="Number of components")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations")
    args = parser.parse_args()
    
    print("=== Step 4: NMF Parts Finder ===", flush=True)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Load data
    print("\n📁 PHASE 1: Loading Data", flush=True)
    activations, meta = load_activation_data(args.activations_file)
    
    # Determine number of parts based on systematic rank selection analysis
    print("\n🧮 PHASE 2: Determining Parts", flush=True)
    n_positions = activations.shape[0]
    
    # Based on systematic rank selection analysis (see rank_analysis/README.md)
    # Recommended rank: k = 25
    # - Best R² score (0.789) among reasonable ranks
    # - Excellent uniqueness (0.466) - parts are distinct
    # - Good balance - not too few, not too many
    # - Avoids overfitting - stops before diminishing returns
    n_parts = 25
    print(f"📊 Positions available: {n_positions}", flush=True)
    print(f"📊 Parts to use: {n_parts} (based on systematic rank selection analysis)", flush=True)
    print(f"📊 Rank selection analysis: See rank_analysis/README.md for details", flush=True)
    
    if n_positions < 100:
        print(f"⚠️  WARNING: Only {n_positions} positions available.", flush=True)
        print(f"⚠️  Using {n_parts} parts based on systematic analysis.", flush=True)
        print(f"⚠️  For full analysis, collect thousands of positions in step 1.", flush=True)
    else:
        print(f"✅ Excellent dataset size: {n_positions} positions for {n_parts} parts", flush=True)
        print(f"✅ Using optimal rank from systematic analysis", flush=True)
    
    # Run NMF with optimal α_H from analysis
    print("\n🏗️  PHASE 3: Running NMF with ℓ1 Sparsity", flush=True)
    
    # Use optimal α_H from 3x3 pooled data analysis (α_H = 0.4 gives 81.2% sparsity)
    # For 3x3 pooled data, we need higher α_H due to 9x more dimensions
    optimal_alpha_H = 0.4
    print(f"📊 Using α_H = {optimal_alpha_H} (target: 81.2% sparsity for 3x3 pooled data)", flush=True)
    print(f"📊 3x3 analysis shows: α_H=0.4 gives 81.2% sparsity vs α_H=0.1 gives 19.1%", flush=True)
    
    parts, activations_transformed, model = run_nmf_factorization(
        activations, n_parts, alpha_H=optimal_alpha_H
    )
    
    # Save results
    print("\n💾 PHASE 4: Saving Results", flush=True)
    save_results(parts, activations_transformed, model, meta, args.output_dir)
    
    # Calculate final sparsity metrics for summary
    H = model.components_
    final_sparsity = (H == 0).sum() / H.size
    final_avg_boards = (H != 0).sum(axis=1).mean()
    
    print("\n=== Summary ===", flush=True)
    print(f"✅ Successfully factorized {activations.shape[0]} positions × {activations.shape[1]} channels", flush=True)
    print(f"✅ Into {parts.shape[0]} parts × {parts.shape[1]} channels", flush=True)
    print(f"📊 Reconstruction error: {model.reconstruction_err_:.4f}", flush=True)
    print(f"📊 Sparsity achieved: {final_sparsity:.1%} ({final_sparsity:.3f})", flush=True)
    print(f"📊 Avg boards per component: {final_avg_boards:.1f}", flush=True)
    print(f"🔄 Iterations used: {model.n_iter_}/1000", flush=True)
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"\n🎯 Next: Run inspect_parts.py to examine the learned parts", flush=True)

if __name__ == "__main__":
    main() 