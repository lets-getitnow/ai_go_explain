#!/usr/bin/env python3
"""
Step 4: Run simple parts finder (NMF)

Loads the pooled activations from step 3 and factorizes them using 
Non-negative Matrix Factorization to find interpretable parts/components.

Since we only have 4 positions, we use 3 components (NMF typically can't
learn more components than samples). In a full run with thousands of positions,
this would be 50-70 components as specified in the README.
"""

import numpy as np
import json
from datetime import datetime
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import os

def load_activation_data():
    """Load the pooled activation data from step 3."""
    print("🔄 Starting to load activation data...")
    
    data_path = "../3_extract_activations/activations/pooled_rconv14.out.npy"
    meta_path = "../3_extract_activations/activations/pooled_meta.json"
    
    print(f"📁 Checking for data file: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Activation data not found: {data_path}")
    
    print(f"📁 Checking for meta file: {meta_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta data not found: {meta_path}")
    
    # Load activation matrix (positions x channels)
    print("📊 Loading numpy activation data...")
    activations = np.load(data_path)
    print(f"✅ Loaded activations shape: {activations.shape}")
    print(f"📊 Activation data stats: min={activations.min():.4f}, max={activations.max():.4f}, mean={activations.mean():.4f}")
    
    # Load metadata
    print("📋 Loading metadata...")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"✅ Loaded metadata with {len(meta)} entries")
    
    print("✅ Successfully loaded all activation data")
    return activations, meta

def run_nmf_factorization(activations, n_components=3):
    """
    Run NMF factorization on the activation data.
    
    Args:
        activations: (n_positions, n_channels) array
        n_components: Number of parts to find
        
    Returns:
        components: (n_components, n_channels) - The learned parts
        activations_transformed: (n_positions, n_components) - Part activations per position
        model: The fitted NMF model
    """
    print(f"🔄 Starting NMF factorization with {n_components} components...")
    
    # Ensure non-negative data (should already be from step 3)
    print("🔧 Ensuring non-negative data...")
    original_min = activations.min()
    activations = np.maximum(activations, 0)
    if original_min < 0:
        print(f"⚠️  Clipped {(activations == 0).sum()} negative values (original min: {original_min:.4f})")
    else:
        print("✅ Data already non-negative")
    
    # Run NMF
    print("🏗️  Creating NMF model...")
    model = NMF(
        n_components=n_components,
        random_state=42,
        max_iter=1000,
        alpha_W=0.01,  # Small L1 regularization for sparsity
        alpha_H=0.01
    )
    print("✅ NMF model created")
    
    # Fit and transform
    print("🔥 Starting NMF fit_transform (this may take time)...")
    print(f"   Input shape: {activations.shape}")
    print(f"   Target components: {n_components}")
    print(f"   Max iterations: 1000")
    
    activations_transformed = model.fit_transform(activations)
    print("✅ NMF fit_transform completed!")
    
    print("📊 Extracting components...")
    components = model.components_
    print("✅ Components extracted")
    
    print(f"📊 NMF reconstruction error: {model.reconstruction_err_:.4f}")
    print(f"📊 Number of iterations used: {model.n_iter_}")
    print(f"📊 Components shape: {components.shape}")
    print(f"📊 Transformed activations shape: {activations_transformed.shape}")
    print(f"📊 Components stats: min={components.min():.4f}, max={components.max():.4f}")
    print(f"📊 Transformed stats: min={activations_transformed.min():.4f}, max={activations_transformed.max():.4f}")
    
    return components, activations_transformed, model

def save_results(components, activations_transformed, model, original_meta):
    """Save NMF results to files."""
    print("🔄 Starting to save results...")
    
    # Save components (the learned parts)
    print("💾 Saving NMF components...")
    np.save("nmf_components.npy", components)
    print("✅ Saved nmf_components.npy")
    
    # Save transformed activations (how much each part activates per position)
    print("💾 Saving transformed activations...")
    np.save("nmf_activations.npy", activations_transformed)
    print("✅ Saved nmf_activations.npy")
    
    # Save metadata
    print("💾 Creating and saving metadata...")
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source_activations": "../3_extract_activations/activations/pooled_rconv14.out.npy",
        "original_shape": f"{activations_transformed.shape[0]}x{components.shape[1]}",
        "n_components": components.shape[0],
        "n_positions": activations_transformed.shape[0],
        "n_channels": components.shape[1],
        "reconstruction_error": float(model.reconstruction_err_),
        "n_iterations": int(model.n_iter_),
        "original_meta": original_meta
    }
    
    with open("nmf_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved nmf_meta.json")
    
    print(f"✅ All results saved:")
    print(f"  - nmf_components.npy: {components.shape}")
    print(f"  - nmf_activations.npy: {activations_transformed.shape}")
    print(f"  - nmf_meta.json")

def main():
    print("=== Step 4: NMF Parts Finder ===")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n📁 PHASE 1: Loading Data")
    activations, meta = load_activation_data()
    
    # Determine number of components
    print("\n🧮 PHASE 2: Determining Components")
    n_positions = activations.shape[0]
    n_components = min(3, n_positions - 1)  # Conservative: fewer than positions
    print(f"📊 Positions available: {n_positions}")
    print(f"📊 Components to use: {n_components}")
    
    if n_positions < 10:
        print(f"⚠️  WARNING: Only {n_positions} positions available.")
        print(f"⚠️  Using {n_components} components instead of 50-70 recommended.")
        print(f"⚠️  For full analysis, collect thousands of positions in step 1.")
    
    # Run NMF
    print("\n🏗️  PHASE 3: Running NMF")
    components, activations_transformed, model = run_nmf_factorization(
        activations, n_components
    )
    
    # Save results
    print("\n💾 PHASE 4: Saving Results")
    save_results(components, activations_transformed, model, meta)
    
    print("\n=== Summary ===")
    print(f"✅ Successfully factorized {activations.shape[0]} positions × {activations.shape[1]} channels")
    print(f"✅ Into {components.shape[0]} parts × {components.shape[1]} channels")
    print(f"📊 Reconstruction error: {model.reconstruction_err_:.4f}")
    print(f"🔄 Iterations used: {model.n_iter_}/1000")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🎯 Next: Run inspect_parts.py to examine the learned parts")

if __name__ == "__main__":
    main() 