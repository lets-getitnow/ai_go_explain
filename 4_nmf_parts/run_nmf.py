#!/usr/bin/env python3
"""
Step 4: Run simple parts finder (NMF)

Loads the pooled activations from step 3 and factorizes them using 
Non-negative Matrix Factorization to find interpretable parts.

With 6,603 positions, we can extract 50 meaningful parts. This allows
us to discover distinct Go concepts like atari patterns, eye shapes, 
ladder formations, etc.
"""

import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import os

def load_activation_data():
    """Load the pooled activation data from step 3."""
    print("🔄 Starting to load activation data...", flush=True)
    
    data_path = "../3_extract_activations/activations/pooled_rconv14.out.npy"
    meta_path = "../3_extract_activations/activations/pooled_meta.json"
    
    print(f"📁 Checking for data file: {data_path}", flush=True)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Activation data not found: {data_path}")
    
    print(f"📁 Checking for meta file: {meta_path}", flush=True)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta data not found: {meta_path}")
    
    # Load activation matrix (positions x channels)
    print("📊 Loading numpy activation data...", flush=True)
    activations = np.load(data_path)
    print(f"✅ Loaded activations shape: {activations.shape}", flush=True)
    print(f"📊 Activation data stats: min={activations.min():.4f}, max={activations.max():.4f}, mean={activations.mean():.4f}", flush=True)
    
    # Load metadata
    print("📋 Loading metadata...", flush=True)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"✅ Loaded metadata with {len(meta)} entries", flush=True)
    
    print("✅ Successfully loaded all activation data", flush=True)
    return activations, meta

def run_nmf_factorization(activations, n_parts=3):
    """
    Run NMF factorization on the activation data.
    
    Args:
        activations: (n_positions, n_channels) array
        n_parts: Number of parts to find
        
    Returns:
        parts: (n_parts, n_channels) - The learned parts
        activations_transformed: (n_positions, n_parts) - Part activations per position
        model: The fitted NMF model
    """
    print(f"🔄 Starting NMF factorization with {n_parts} parts...", flush=True)
    
    # Ensure non-negative data (should already be from step 3)
    print("🔧 Ensuring non-negative data...", flush=True)
    original_min = activations.min()
    activations = np.maximum(activations, 0)
    if original_min < 0:
        print(f"⚠️  Clipped {(activations == 0).sum()} negative values (original min: {original_min:.4f})", flush=True)
    else:
        print("✅ Data already non-negative", flush=True)
    
    # Run NMF
    print("🏗️  Creating NMF model...", flush=True)
    model = NMF(
        n_components=n_parts,
        random_state=42,
        max_iter=1000,
        alpha_W=0.01,  # Small L1 regularization for sparsity
        alpha_H=0.01
    )
    print("✅ NMF model created", flush=True)
    
    # Fit and transform
    print("🔥 Starting NMF fit_transform (this may take time)...", flush=True)
    print(f"   Input shape: {activations.shape}", flush=True)
    print(f"   Target parts: {n_parts}", flush=True)
    print(f"   Max iterations: 1000", flush=True)
    
    activations_transformed = model.fit_transform(activations)
    print("✅ NMF fit_transform completed!", flush=True)
    
    print("📊 Extracting parts...", flush=True)
    parts = model.components_
    print("✅ Parts extracted", flush=True)
    
    print(f"📊 NMF reconstruction error: {model.reconstruction_err_:.4f}", flush=True)
    print(f"📊 Number of iterations used: {model.n_iter_}", flush=True)
    print(f"📊 Parts shape: {parts.shape}", flush=True)
    print(f"📊 Transformed activations shape: {activations_transformed.shape}", flush=True)
    print(f"📊 Parts stats: min={parts.min():.4f}, max={parts.max():.4f}", flush=True)
    print(f"📊 Transformed stats: min={activations_transformed.min():.4f}, max={activations_transformed.max():.4f}", flush=True)
    
    return parts, activations_transformed, model

def save_results(parts, activations_transformed, model, original_meta):
    """Save NMF results to files."""
    print("🔄 Starting to save results...", flush=True)
    
    # Save parts (the learned parts)
    print("💾 Saving NMF parts...", flush=True)
    np.save("nmf_components.npy", parts)
    print("✅ Saved nmf_components.npy", flush=True)
    
    # Save transformed activations (how much each part activates per position)
    print("💾 Saving transformed activations...", flush=True)
    np.save("nmf_activations.npy", activations_transformed)
    print("✅ Saved nmf_activations.npy", flush=True)
    
    # Save metadata
    print("💾 Creating and saving metadata...", flush=True)
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source_activations": "../3_extract_activations/activations/pooled_rconv14.out.npy",
        "original_shape": f"{activations_transformed.shape[0]}x{parts.shape[1]}",
        "n_parts": parts.shape[0],
        "n_positions": activations_transformed.shape[0],
        "n_channels": parts.shape[1],
        "reconstruction_error": float(model.reconstruction_err_),
        "n_iterations": int(model.n_iter_),
        "original_meta": original_meta
    }
    
    with open("nmf_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved nmf_meta.json", flush=True)
    
    print(f"✅ All results saved:", flush=True)
    print(f"  - nmf_components.npy: {parts.shape}", flush=True)
    print(f"  - nmf_activations.npy: {activations_transformed.shape}", flush=True)
    print(f"  - nmf_meta.json", flush=True)

def main():
    print("=== Step 4: NMF Parts Finder ===", flush=True)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Load data
    print("\n📁 PHASE 1: Loading Data", flush=True)
    activations, meta = load_activation_data()
    
    # Determine number of parts
    print("\n🧮 PHASE 2: Determining Parts", flush=True)
    n_positions = activations.shape[0]
    
    # With 6,603 positions, we can extract many more meaningful parts
    # Rule of thumb: aim for 50-100 parts for this dataset size
    n_parts = min(50, n_positions // 10)  # Conservative but much more reasonable
    print(f"📊 Positions available: {n_positions}", flush=True)
    print(f"📊 Parts to use: {n_parts}", flush=True)
    
    if n_positions < 100:
        print(f"⚠️  WARNING: Only {n_positions} positions available.", flush=True)
        print(f"⚠️  Using {n_parts} parts instead of 50-70 recommended.", flush=True)
        print(f"⚠️  For full analysis, collect thousands of positions in step 1.", flush=True)
    else:
        print(f"✅ Good dataset size: {n_positions} positions for {n_parts} parts", flush=True)
    
    # Run NMF
    print("\n🏗️  PHASE 3: Running NMF", flush=True)
    parts, activations_transformed, model = run_nmf_factorization(
        activations, n_parts
    )
    
    # Save results
    print("\n💾 PHASE 4: Saving Results", flush=True)
    save_results(parts, activations_transformed, model, meta)
    
    print("\n=== Summary ===", flush=True)
    print(f"✅ Successfully factorized {activations.shape[0]} positions × {activations.shape[1]} channels", flush=True)
    print(f"✅ Into {parts.shape[0]} parts × {parts.shape[1]} channels", flush=True)
    print(f"📊 Reconstruction error: {model.reconstruction_err_:.4f}", flush=True)
    print(f"🔄 Iterations used: {model.n_iter_}/1000", flush=True)
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"\n🎯 Next: Run inspect_parts.py to examine the learned parts", flush=True)

if __name__ == "__main__":
    main() 