#!/usr/bin/env python3
"""
Alpha_H Analysis for NMF Sparsity Control

This script implements the ℓ1 sparsity penalty analysis described in the methodology.
It tests different α_H values to find the optimal balance between sparsity and 
reconstruction quality.

The goal is to find α_H that gives:
- Sparsity: 70-90% zeros in H matrix
- Reconstruction error increase ≤ 5% over α_H = 0
- Clear, interpretable parts that don't fire on every board

Based on the methodology:
- We penalize H (activations) rather than W (basis) for sparse usage across positions
- Use l1_ratio=1.0 for pure ℓ1 penalty
- Target sparsity 70-90% with minimal reconstruction error increase
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

def load_activation_data():
    """Load the pooled activation data from step 3."""
    print("🔄 Loading activation data...", flush=True)
    
    data_path = "../3_extract_activations/activations/pooled_rconv14.out.npy"
    meta_path = "../3_extract_activations/activations/pooled_meta.json"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Activation data not found: {data_path}")
    
    # Load activation matrix (positions x channels)
    activations = np.load(data_path)
    print(f"✅ Loaded activations shape: {activations.shape}", flush=True)
    
    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"✅ Loaded metadata with {len(meta)} entries", flush=True)
    
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

def run_alpha_grid_analysis(X, k=25):
    """
    Run α_H grid analysis to find optimal sparsity penalty.
    
    Args:
        X: Preprocessed activation data
        k: Number of components (using recommended k=25)
        
    Returns:
        results: List of (alpha, reconstruction_error, sparsity) tuples
    """
    print(f"🔬 Running α_H grid analysis with k={k}...", flush=True)
    
    # Grid of α_H values to test
    l1_grid = [0.00, 0.01, 0.05, 0.1, 0.2, 0.4]
    
    results = []
    
    for i, alpha in enumerate(l1_grid):
        print(f"  Testing α_H = {alpha:.2f} ({i+1}/{len(l1_grid)})...", flush=True)
        
        nmf = NMF(
            n_components=k,
            init="nndsvd",
            alpha_H=alpha,
            alpha_W=0.0,          # Basis can stay dense
            l1_ratio=1.0,         # ρ = 1 → pure ℓ1 on H
            max_iter=400,
            random_state=42
        )
        
        # Fit and transform
        W = nmf.fit_transform(X)
        H = nmf.components_
        
        # Calculate sparsity: fraction of zero activations in H
        sparsity = (H == 0).sum() / H.size
        
        # Reconstruction error reported by scikit (Frobenius)
        rec_err = nmf.reconstruction_err_
        
        # Average boards per component (non-zero activations)
        avg_boards_per_component = (H != 0).sum(axis=1).mean()
        
        results.append({
            'alpha': alpha,
            'reconstruction_error': rec_err,
            'sparsity': sparsity,
            'avg_boards_per_component': avg_boards_per_component,
            'n_iterations': nmf.n_iter_
        })
        
        print(f"    Sparsity: {sparsity:.3f}, Error: {rec_err:.4f}, Avg boards/comp: {avg_boards_per_component:.1f}", flush=True)
    
    return results

def analyze_results(results):
    """
    Analyze results to find optimal α_H.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        optimal_alpha: Recommended α_H value
        analysis: Analysis summary
    """
    print("\n📊 Analyzing results...", flush=True)
    
    # Find baseline (α_H = 0)
    baseline = next(r for r in results if r['alpha'] == 0.0)
    baseline_error = baseline['reconstruction_error']
    
    print(f"📊 Baseline (α_H=0): Error={baseline_error:.4f}, Sparsity={baseline['sparsity']:.3f}", flush=True)
    
    # Find optimal α_H based on criteria
    optimal_alpha = None
    optimal_sparsity = None
    optimal_error = None
    
    for result in results:
        if result['alpha'] == 0.0:
            continue
            
        sparsity = result['sparsity']
        error = result['reconstruction_error']
        error_increase = (error - baseline_error) / baseline_error
        
        # Criteria: 70-90% sparsity, error increase ≤ 5%
        if 0.7 <= sparsity <= 0.9 and error_increase <= 0.05:
            if optimal_alpha is None or sparsity > optimal_sparsity:
                optimal_alpha = result['alpha']
                optimal_sparsity = sparsity
                optimal_error = error
    
    # If no perfect match, find best compromise
    if optimal_alpha is None:
        print("⚠️  No α_H found meeting strict criteria, finding best compromise...", flush=True)
        
        # Find α_H with highest sparsity while keeping error increase reasonable
        best_score = -1
        for result in results:
            if result['alpha'] == 0.0:
                continue
                
            sparsity = result['sparsity']
            error_increase = (result['reconstruction_error'] - baseline_error) / baseline_error
            
            # Score: sparsity - 2*error_increase (prioritize sparsity)
            score = sparsity - 2 * error_increase
            
            if score > best_score:
                best_score = score
                optimal_alpha = result['alpha']
                optimal_sparsity = sparsity
                optimal_error = result['reconstruction_error']
    
    analysis = {
        'baseline_error': baseline_error,
        'optimal_alpha': optimal_alpha,
        'optimal_sparsity': optimal_sparsity,
        'optimal_error': optimal_error,
        'error_increase': (optimal_error - baseline_error) / baseline_error if optimal_error else None,
        'recommendation': f"Use α_H = {optimal_alpha:.2f} for {optimal_sparsity:.1%} sparsity with {((optimal_error - baseline_error) / baseline_error * 100):.1f}% error increase"
    }
    
    print(f"✅ {analysis['recommendation']}", flush=True)
    
    return optimal_alpha, analysis

def create_diagnostic_plots(results, save_path="alpha_h_analysis.png"):
    """
    Create diagnostic plots for α_H analysis.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the plot
    """
    print(f"📈 Creating diagnostic plots...", flush=True)
    
    alphas = [r['alpha'] for r in results]
    errors = [r['reconstruction_error'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    avg_boards = [r['avg_boards_per_component'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Reconstruction Error vs α_H
    ax1.plot(alphas, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('α_H (ℓ1 penalty)')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('Reconstruction Error vs α_H')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sparsity vs α_H
    ax2.plot(alphas, sparsities, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Target: 70%')
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target: 90%')
    ax2.set_xlabel('α_H (ℓ1 penalty)')
    ax2.set_ylabel('Sparsity (% zeros in H)')
    ax2.set_title('Sparsity vs α_H')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Boards per Component
    ax3.plot(alphas, avg_boards, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('α_H (ℓ1 penalty)')
    ax3.set_ylabel('Avg Boards per Component')
    ax3.set_title('Component Usage Sparsity')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined Analysis
    baseline_error = errors[0]  # α_H = 0
    error_increases = [(e - baseline_error) / baseline_error * 100 for e in errors]
    
    ax4.plot(alphas, error_increases, 'mo-', linewidth=2, markersize=8)
    ax4.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    ax4.set_xlabel('α_H (ℓ1 penalty)')
    ax4.set_ylabel('Error Increase (%)')
    ax4.set_title('Reconstruction Error Increase')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved diagnostic plots to {save_path}", flush=True)

def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "="*60)
    print("α_H ANALYSIS RESULTS")
    print("="*60)
    print("alpha  |  recon_err   sparsity   avg_boards/comp")
    print("-" * 60)
    
    for result in results:
        alpha = result['alpha']
        err = result['reconstruction_error']
        sparsity = result['sparsity']
        avg_boards = result['avg_boards_per_component']
        
        print(f"{alpha:5.2f}  |  {err:10.4f}   {sparsity:7.3f}      {avg_boards:8.1f}")
    
    print("="*60)

def main():
    print("=== α_H Sparsity Analysis ===", flush=True)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Load and preprocess data
    print("\n📁 PHASE 1: Loading and Preprocessing", flush=True)
    activations, meta = load_activation_data()
    X = preprocess_data(activations)
    
    # Run α_H grid analysis
    print("\n🔬 PHASE 2: α_H Grid Analysis", flush=True)
    results = run_alpha_grid_analysis(X, k=25)
    
    # Analyze results
    print("\n📊 PHASE 3: Analysis", flush=True)
    optimal_alpha, analysis = analyze_results(results)
    
    # Create diagnostic plots
    print("\n📈 PHASE 4: Visualizations", flush=True)
    create_diagnostic_plots(results)
    
    # Print results table
    print_results_table(results)
    
    # Save analysis results
    print("\n💾 PHASE 5: Saving Results", flush=True)
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = []
    for result in results:
        results_serializable.append({
            'alpha': float(result['alpha']),
            'reconstruction_error': float(result['reconstruction_error']),
            'sparsity': float(result['sparsity']),
            'avg_boards_per_component': float(result['avg_boards_per_component']),
            'n_iterations': int(result['n_iterations'])
        })
    
    analysis_serializable = {
        'baseline_error': float(analysis['baseline_error']),
        'optimal_alpha': float(analysis['optimal_alpha']) if analysis['optimal_alpha'] is not None else None,
        'optimal_sparsity': float(analysis['optimal_sparsity']) if analysis['optimal_sparsity'] is not None else None,
        'optimal_error': float(analysis['optimal_error']) if analysis['optimal_error'] is not None else None,
        'error_increase': float(analysis['error_increase']) if analysis['error_increase'] is not None else None,
        'recommendation': analysis['recommendation']
    }
    
    analysis_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results_serializable,
        'optimal_alpha': float(optimal_alpha) if optimal_alpha is not None else None,
        'analysis': analysis_serializable,
        'recommendation': analysis['recommendation']
    }
    
    with open('alpha_h_analysis_results.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print("✅ Saved analysis results to alpha_h_analysis_results.json", flush=True)
    
    print("\n=== Summary ===", flush=True)
    print(f"✅ Analyzed {len(results)} α_H values", flush=True)
    print(f"✅ Optimal α_H: {optimal_alpha:.2f}", flush=True)
    print(f"📊 {analysis['recommendation']}", flush=True)
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"\n🎯 Next: Update run_nmf.py with optimal α_H = {optimal_alpha:.2f}", flush=True)

if __name__ == "__main__":
    main() 