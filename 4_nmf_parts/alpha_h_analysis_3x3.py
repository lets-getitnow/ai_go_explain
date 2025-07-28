#!/usr/bin/env python3
"""
Alpha_H Analysis for 3x3 Pooled NMF Sparsity Control

This script tests different α_H values for 3x3 pooled data to find the optimal
balance between sparsity and reconstruction quality.

With 3x3 pooling, we have 9x more dimensions (4608 vs 512), so we need to
adjust α_H accordingly to achieve the same sparsity levels.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

def load_activation_data():
    """Load the 3x3 pooled activation data."""
    print("🔄 Loading 3x3 pooled activation data...", flush=True)
    
    data_path = "../3_extract_activations/activations/pooled_rconv14.out.npy"
    meta_path = "../3_extract_activations/activations/pooled_meta.json"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"3x3 activation data not found: {data_path}")
    
    # Load activation matrix (positions x channels)
    activations = np.load(data_path)
    print(f"✅ Loaded 3x3 activations shape: {activations.shape}", flush=True)
    
    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"✅ Loaded metadata with {len(meta)} entries", flush=True)
    print(f"📊 Pooling method: {meta.get('pooling_method', 'unknown')}", flush=True)
    print(f"📊 Original channels: {meta.get('original_channels', 'unknown')}", flush=True)
    print(f"📊 Pooled channels: {meta.get('pooled_channels', 'unknown')}", flush=True)
    
    return activations, meta

def preprocess_data(X):
    """Preprocess data for meaningful alpha values."""
    print("🔧 Preprocessing 3x3 pooled data...", flush=True)
    
    # Ensure non-negativity
    X = np.maximum(0, X)
    print(f"📊 Original data stats: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}", flush=True)
    
    # Scale to roughly unit magnitude so alpha values are meaningful
    scaler = StandardScaler(with_mean=False)  # Keep non-negative
    X_scaled = scaler.fit_transform(X)
    
    print(f"📊 Scaled data stats: min={X_scaled.min():.4f}, max={X_scaled.max():.4f}, mean={X_scaled.mean():.4f}", flush=True)
    
    return X_scaled

def test_alpha_values(X, n_parts=25):
    """Test different α_H values and record metrics."""
    print(f"🧪 Testing α_H values for 3x3 pooled data with {n_parts} parts...", flush=True)
    
    # Test a wider range of α_H values for 3x3 data
    alpha_values = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    
    results = []
    
    for alpha in alpha_values:
        print(f"🔬 Testing α_H = {alpha}...", flush=True)
        
        # Run NMF with current α_H
        model = NMF(
            n_components=n_parts,
            init="nndsvd",
            alpha_H=alpha,
            alpha_W=0.0,
            l1_ratio=1.0,
            max_iter=200,  # Fewer iterations for faster testing
            random_state=42
        )
        
        # Fit the model
        model.fit(X)
        
        # Calculate metrics
        H = model.components_
        sparsity = (H == 0).sum() / H.size
        avg_boards_per_component = (H != 0).sum(axis=1).mean()
        reconstruction_error = model.reconstruction_err_
        
        results.append({
            'alpha_H': float(alpha),
            'sparsity': float(sparsity),
            'avg_boards_per_component': float(avg_boards_per_component),
            'reconstruction_error': float(reconstruction_error),
            'n_iterations': int(model.n_iter_)
        })
        
        print(f"   📊 Sparsity: {sparsity:.1%}", flush=True)
        print(f"   📊 Avg boards/component: {avg_boards_per_component:.1f}", flush=True)
        print(f"   📊 Reconstruction error: {reconstruction_error:.2f}", flush=True)
    
    return results

def create_analysis_plots(results):
    """Create diagnostic plots for α_H analysis."""
    print("📊 Creating analysis plots...", flush=True)
    
    alphas = [r['alpha_H'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    errors = [r['reconstruction_error'] for r in results]
    boards_per_comp = [r['avg_boards_per_component'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Sparsity vs α_H
    axes[0, 0].plot(alphas, sparsities, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('α_H (ℓ1 penalty)')
    axes[0, 0].set_ylabel('Sparsity (% zeros)')
    axes[0, 0].set_title('Sparsity vs α_H for 3x3 Pooled Data')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Reconstruction Error vs α_H
    axes[0, 1].plot(alphas, errors, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('α_H (ℓ1 penalty)')
    axes[0, 1].set_ylabel('Reconstruction Error')
    axes[0, 1].set_title('Reconstruction Error vs α_H')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Plot 3: Avg Boards per Component vs α_H
    axes[1, 0].plot(alphas, boards_per_comp, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('α_H (ℓ1 penalty)')
    axes[1, 0].set_ylabel('Avg Boards per Component')
    axes[1, 0].set_title('Component Specificity vs α_H')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Sparsity vs Reconstruction Error (trade-off)
    axes[1, 1].scatter(sparsities, errors, c=alphas, cmap='viridis', s=100)
    axes[1, 1].set_xlabel('Sparsity (% zeros)')
    axes[1, 1].set_ylabel('Reconstruction Error')
    axes[1, 1].set_title('Sparsity vs Error Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar for α_H values
    scatter = axes[1, 1].scatter(sparsities, errors, c=alphas, cmap='viridis', s=100)
    plt.colorbar(scatter, ax=axes[1, 1], label='α_H')
    
    plt.tight_layout()
    plt.savefig('alpha_h_analysis_3x3.png', dpi=150, bbox_inches='tight')
    print("✅ Saved analysis plots: alpha_h_analysis_3x3.png", flush=True)

def generate_recommendations(results):
    """Generate recommendations based on analysis results."""
    print("📋 Generating recommendations...", flush=True)
    
    # Find optimal α_H based on target sparsity (70-90%)
    target_sparsity_min = 0.70
    target_sparsity_max = 0.90
    
    candidates = []
    for r in results:
        if target_sparsity_min <= r['sparsity'] <= target_sparsity_max:
            candidates.append(r)
    
    if candidates:
        # Choose the one with lowest reconstruction error
        best = min(candidates, key=lambda x: x['reconstruction_error'])
        recommendation = best['alpha_H']
        print(f"🎯 RECOMMENDED α_H = {recommendation}", flush=True)
        print(f"   📊 Sparsity: {best['sparsity']:.1%}", flush=True)
        print(f"   📊 Reconstruction error: {best['reconstruction_error']:.2f}", flush=True)
        print(f"   📊 Avg boards/component: {best['avg_boards_per_component']:.1f}", flush=True)
    else:
        # Find closest to target
        closest = min(results, key=lambda x: abs(x['sparsity'] - 0.80))
        recommendation = closest['alpha_H']
        print(f"⚠️  No α_H achieves target sparsity (70-90%)", flush=True)
        print(f"🎯 CLOSEST α_H = {recommendation} (sparsity: {closest['sparsity']:.1%})", flush=True)
    
    # Save recommendations
    recommendations = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "pooling_method": "3x3_grid",
        "recommended_alpha_H": recommendation,
        "target_sparsity_range": [target_sparsity_min, target_sparsity_max],
        "analysis_results": results
    }
    
    with open("alpha_h_recommendations_3x3.json", 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("✅ Saved recommendations: alpha_h_recommendations_3x3.json", flush=True)
    
    return recommendation

def main():
    """Run α_H analysis for 3x3 pooled data."""
    print("=== α_H Analysis for 3x3 Pooled Data ===")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Load 3x3 pooled data
    activations, meta = load_activation_data()
    
    # Preprocess data
    X = preprocess_data(activations)
    
    # Test different α_H values
    results = test_alpha_values(X, n_parts=25)
    
    # Create analysis plots
    create_analysis_plots(results)
    
    # Generate recommendations
    recommended_alpha = generate_recommendations(results)
    
    print("\n=== Summary ===", flush=True)
    print(f"📊 3x3 pooled data shape: {activations.shape}", flush=True)
    print(f"📊 Tested {len(results)} α_H values", flush=True)
    print(f"🎯 Recommended α_H: {recommended_alpha}", flush=True)
    print(f"📊 Expected improvement: Higher sparsity for 3x3 pooled data", flush=True)

if __name__ == "__main__":
    main() 