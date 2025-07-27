#!/usr/bin/env python3
"""
Systematic NMF Rank Selection

This script implements a systematic approach to find the optimal number of NMF parts
by analyzing reconstruction quality, component uniqueness, and interpretability.

## What is Rank Selection?

Rank selection is the process of choosing the optimal number of "parts" or "components" 
for NMF decomposition. Too few parts and you miss important patterns; too many and 
you fit noise.

## What Does "Reconstruction" Mean?

Reconstruction refers to how well the NMF model can "rebuild" or "recreate" the original 
data from the learned parts:

1. Original Data: Your activation matrix (positions × channels)
2. NMF Decomposition: Breaks this into two matrices:
   - Parts Matrix: (k parts × channels) - the learned "concepts"
   - Activations Matrix: (positions × k parts) - how much each part activates per position
3. Reconstruction: Multiply these back together: Activations × Parts = Reconstructed Data

Mathematical Formula: Original Data ≈ Activations × Parts

The R² score measures how well the reconstruction matches the original:
- R² = 1.0: Perfect reconstruction (100% of original data explained)
- R² = 0.8: 80% of original data explained
- R² = 0.5: 50% of original data explained

## Quick & Dirty Rank-Selection Recipe (≤ 30 min):

1. Make a reconstruction curve - For ranks k = 3, 5, 10, 15, 25, 40, 60 run NMF for ≤ 20 iterations each
2. Plot component uniqueness - Compute cosine distance between weight vectors (want > 0.3)
3. Visual spot-check only the elbow ranks
4. Pick the smallest rank that gives ≥ 15 interpretable parts

## Output:

- rank_analysis/rank_selection_analysis.png: Comprehensive visualizations
- rank_analysis/rank_analysis_report.txt: Detailed numerical analysis
- rank_analysis/README.md: Complete documentation

## Requirements:
- matplotlib for visualizations
- scikit-learn for NMF and metrics
- numpy for numerical operations
- seaborn for enhanced plotting
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_activation_data():
    """Load the pooled activation data from step 3."""
    print("🔄 Loading activation data...", flush=True)
    
    data_path = "../3_extract_activations/activations/pooled_rconv14.out.npy"
    meta_path = "../3_extract_activations/activations/pooled_meta.json"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Activation data not found: {data_path}")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta data not found: {meta_path}")
    
    # Load activation matrix (positions x channels)
    activations = np.load(data_path)
    print(f"✅ Loaded activations shape: {activations.shape}", flush=True)
    
    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"✅ Loaded metadata with {len(meta)} entries", flush=True)
    
    return activations, meta

def prepare_data_for_analysis(activations):
    """Prepare data for rank selection analysis."""
    print("🔧 Preparing data for analysis...", flush=True)
    
    # Ensure non-negative data
    original_min = activations.min()
    activations = np.maximum(activations, 0)
    if original_min < 0:
        print(f"⚠️  Clipped {(activations == 0).sum()} negative values", flush=True)
    
    # Split into train/test for reconstruction evaluation
    train_data, test_data = train_test_split(activations, test_size=0.1, random_state=42)
    print(f"📊 Train shape: {train_data.shape}, Test shape: {test_data.shape}", flush=True)
    
    return train_data, test_data

def compute_reconstruction_quality(train_data, test_data, ranks):
    """Compute R² reconstruction quality for different ranks."""
    print("📊 Computing reconstruction quality...", flush=True)
    
    r2_scores = []
    reconstruction_errors = []
    
    for rank in ranks:
        print(f"  Testing rank {rank}...", flush=True)
        
        # Run NMF with limited iterations for speed
        model = NMF(
            n_components=rank,
            random_state=42,
            max_iter=20,  # Fast evaluation
            alpha_W=0.01,
            alpha_H=0.01
        )
        
        # Fit on training data
        train_reconstructed = model.fit_transform(train_data)
        train_reconstructed = np.dot(train_reconstructed, model.components_)
        
        # Evaluate on test data
        test_reconstructed = model.transform(test_data)
        test_reconstructed = np.dot(test_reconstructed, model.components_)
        
        # Compute R² score
        r2 = r2_score(test_data.flatten(), test_reconstructed.flatten())
        r2_scores.append(r2)
        
        # Store reconstruction error
        reconstruction_errors.append(model.reconstruction_err_)
        
        print(f"    Rank {rank}: R² = {r2:.4f}, Error = {model.reconstruction_err_:.4f}", flush=True)
    
    return r2_scores, reconstruction_errors

def compute_component_uniqueness(train_data, ranks):
    """Compute component uniqueness using cosine distance."""
    print("🔍 Computing component uniqueness...", flush=True)
    
    uniqueness_scores = []
    
    for rank in ranks:
        print(f"  Testing rank {rank}...", flush=True)
        
        # Run NMF
        model = NMF(
            n_components=rank,
            random_state=42,
            max_iter=20,
            alpha_W=0.01,
            alpha_H=0.01
        )
        
        model.fit(train_data)
        components = model.components_
        
        # Compute cosine distances between all pairs of components
        distances = cosine_distances(components)
        
        # Get mean distance (excluding diagonal)
        np.fill_diagonal(distances, np.nan)  # Exclude self-comparisons
        mean_distance = np.nanmean(distances)
        uniqueness_scores.append(mean_distance)
        
        print(f"    Rank {rank}: Mean uniqueness = {mean_distance:.4f}", flush=True)
    
    return uniqueness_scores

def find_elbow_point(x_values, y_values):
    """Find the elbow point in a curve using the maximum curvature method."""
    if len(x_values) < 3:
        return x_values[0] if len(x_values) > 0 else None
    
    # Compute first and second derivatives
    dx = np.gradient(x_values)
    dy = np.gradient(y_values)
    d2y = np.gradient(dy)
    
    # Compute curvature
    curvature = np.abs(d2y) / (1 + dy**2)**1.5
    
    # Find point of maximum curvature
    elbow_idx = np.argmax(curvature)
    return x_values[elbow_idx]

def create_visualizations(ranks, r2_scores, reconstruction_errors, uniqueness_scores, output_dir="rank_analysis"):
    """Create comprehensive visualizations for rank selection."""
    print("🎨 Creating visualizations...", flush=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NMF Rank Selection Analysis', fontsize=16, fontweight='bold')
    
    # 1. Reconstruction Quality (R²)
    ax1.plot(ranks, r2_scores, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Number of Components (k)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Reconstruction Quality', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(ranks) + 5)
    
    # Find and mark elbow point
    elbow_rank = find_elbow_point(ranks, r2_scores)
    if elbow_rank:
        elbow_idx = ranks.index(elbow_rank)
        ax1.axvline(x=elbow_rank, color='red', linestyle='--', alpha=0.7, label=f'Elbow: k={elbow_rank}')
        ax1.legend()
    
    # 2. Reconstruction Error
    ax2.plot(ranks, reconstruction_errors, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Number of Components (k)', fontsize=12)
    ax2.set_ylabel('Reconstruction Error', fontsize=12)
    ax2.set_title('NMF Reconstruction Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(ranks) + 5)
    
    # 3. Component Uniqueness
    ax3.plot(ranks, uniqueness_scores, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Uniqueness threshold (0.3)')
    ax3.set_xlabel('Number of Components (k)', fontsize=12)
    ax3.set_ylabel('Mean Cosine Distance', fontsize=12)
    ax3.set_title('Component Uniqueness', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(ranks) + 5)
    ax3.legend()
    
    # 4. Combined Analysis
    # Normalize scores for comparison
    r2_norm = np.array(r2_scores) / max(r2_scores)
    uniqueness_norm = np.array(uniqueness_scores) / max(uniqueness_scores)
    
    ax4.plot(ranks, r2_norm, 'o-', linewidth=2, markersize=8, label='R² Score (normalized)', color='#2E86AB')
    ax4.plot(ranks, uniqueness_norm, 's-', linewidth=2, markersize=8, label='Uniqueness (normalized)', color='#F18F01')
    ax4.set_xlabel('Number of Components (k)', fontsize=12)
    ax4.set_ylabel('Normalized Score', fontsize=12)
    ax4.set_title('Combined Analysis', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max(ranks) + 5)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_selection_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed analysis report
    create_analysis_report(ranks, r2_scores, reconstruction_errors, uniqueness_scores, output_dir)
    
    print(f"✅ Visualizations saved to {output_dir}/", flush=True)

def create_analysis_report(ranks, r2_scores, reconstruction_errors, uniqueness_scores, output_dir):
    """Create a detailed text report of the analysis."""
    print("📝 Creating analysis report...", flush=True)
    
    report_path = os.path.join(output_dir, 'rank_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("NMF Rank Selection Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("RECONSTRUCTION QUALITY ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for i, rank in enumerate(ranks):
            f.write(f"Rank {rank:2d}: R² = {r2_scores[i]:.4f}, Error = {reconstruction_errors[i]:.4f}\n")
        
        f.write(f"\nBest R² Score: {max(r2_scores):.4f} at rank {ranks[np.argmax(r2_scores)]}\n")
        
        # Find elbow point
        elbow_rank = find_elbow_point(ranks, r2_scores)
        if elbow_rank:
            f.write(f"Elbow Point: k = {elbow_rank}\n")
        
        f.write("\nCOMPONENT UNIQUENESS ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for i, rank in enumerate(ranks):
            status = "✓" if uniqueness_scores[i] > 0.3 else "✗"
            f.write(f"Rank {rank:2d}: Uniqueness = {uniqueness_scores[i]:.4f} {status}\n")
        
        f.write(f"\nUniqueness threshold: 0.3\n")
        f.write(f"Ranks above threshold: {sum(1 for s in uniqueness_scores if s > 0.3)}\n")
        
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        
        # Find optimal rank based on criteria
        optimal_ranks = []
        for i, rank in enumerate(ranks):
            if uniqueness_scores[i] > 0.3 and r2_scores[i] > 0.8:  # Good uniqueness and reconstruction
                optimal_ranks.append((rank, r2_scores[i], uniqueness_scores[i]))
        
        if optimal_ranks:
            f.write("Recommended ranks (good uniqueness + reconstruction):\n")
            for rank, r2, uniqueness in sorted(optimal_ranks, key=lambda x: x[0]):
                f.write(f"  k = {rank}: R² = {r2:.4f}, Uniqueness = {uniqueness:.4f}\n")
        else:
            f.write("No ranks meet both criteria. Consider:\n")
            f.write("  - Lowering uniqueness threshold\n")
            f.write("  - Accepting lower reconstruction quality\n")
            f.write("  - Collecting more data\n")
        
        # Rule of thumb analysis
        f.write("\nRULE OF THUMB ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("For tiny datasets (N ≈ 3–5k):\n")
        f.write("Choose rank where k × feature-dimensionality < 0.2 × N\n")
        f.write("With 512 channels and 8 positions:\n")
        f.write("  k × 512 < 0.2 × 8\n")
        f.write("  k < 1.6 / 512 ≈ 0.003\n")
        f.write("This suggests very low ranks due to small dataset size.\n")
        f.write("Consider collecting more positions for meaningful analysis.\n")
    
    print(f"✅ Analysis report saved to {report_path}", flush=True)

def suggest_optimal_rank(ranks, r2_scores, uniqueness_scores):
    """Suggest optimal rank based on analysis."""
    print("\n🎯 RANK RECOMMENDATIONS", flush=True)
    print("=" * 30, flush=True)
    
    # Find ranks with good uniqueness
    good_uniqueness = [(i, r) for i, r in enumerate(ranks) if uniqueness_scores[i] > 0.3]
    
    if not good_uniqueness:
        print("⚠️  No ranks meet uniqueness threshold (0.3)", flush=True)
        print("   Consider lowering threshold or collecting more data", flush=True)
        return None
    
    # Find ranks with good reconstruction (R² > 0.8)
    good_reconstruction = [(i, r) for i, r in enumerate(ranks) if r2_scores[i] > 0.8]
    
    if not good_reconstruction:
        print("⚠️  No ranks meet reconstruction threshold (R² > 0.8)", flush=True)
        print("   Consider lowering threshold or collecting more data", flush=True)
        return None
    
    # Find intersection
    good_ranks = set(good_uniqueness) & set(good_reconstruction)
    
    if not good_ranks:
        print("⚠️  No ranks meet both criteria", flush=True)
        print("   Balancing uniqueness and reconstruction quality...", flush=True)
        # Find rank with best balance
        scores = []
        for i, rank in enumerate(ranks):
            balance_score = (r2_scores[i] * 0.6) + (uniqueness_scores[i] * 0.4)
            scores.append((rank, balance_score))
        
        best_rank = max(scores, key=lambda x: x[1])[0]
        print(f"   Best balanced rank: k = {best_rank}", flush=True)
        return best_rank
    
    # Choose smallest rank that meets both criteria
    optimal_rank = min(good_ranks, key=lambda x: x[1])[1]
    print(f"✅ Recommended rank: k = {optimal_rank}", flush=True)
    print(f"   R² = {r2_scores[ranks.index(optimal_rank)]:.4f}", flush=True)
    print(f"   Uniqueness = {uniqueness_scores[ranks.index(optimal_rank)]:.4f}", flush=True)
    
    return optimal_rank

def main():
    """Main function for rank selection analysis."""
    print("=== NMF Rank Selection Analysis ===", flush=True)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Load data
    print("\n📁 PHASE 1: Loading Data", flush=True)
    activations, meta = load_activation_data()
    
    # Prepare data
    print("\n🔧 PHASE 2: Preparing Data", flush=True)
    train_data, test_data = prepare_data_for_analysis(activations)
    
    # Define ranks to test
    ranks = [3, 5, 10, 15, 25, 40, 60]
    print(f"\n🧮 PHASE 3: Testing Ranks {ranks}", flush=True)
    
    # Compute reconstruction quality
    print("\n📊 Computing reconstruction quality...", flush=True)
    r2_scores, reconstruction_errors = compute_reconstruction_quality(train_data, test_data, ranks)
    
    # Compute component uniqueness
    print("\n🔍 Computing component uniqueness...", flush=True)
    uniqueness_scores = compute_component_uniqueness(train_data, ranks)
    
    # Create visualizations
    print("\n🎨 PHASE 4: Creating Visualizations", flush=True)
    create_visualizations(ranks, r2_scores, reconstruction_errors, uniqueness_scores)
    
    # Suggest optimal rank
    print("\n🎯 PHASE 5: Making Recommendations", flush=True)
    optimal_rank = suggest_optimal_rank(ranks, r2_scores, uniqueness_scores)
    
    # Summary
    print("\n=== SUMMARY ===", flush=True)
    print(f"✅ Analyzed {len(ranks)} different ranks", flush=True)
    print(f"✅ Created visualizations in rank_analysis/", flush=True)
    print(f"✅ Generated detailed report", flush=True)
    if optimal_rank:
        print(f"🎯 Recommended rank: k = {optimal_rank}", flush=True)
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    print(f"\n📋 Next steps:", flush=True)
    print(f"   1. Review visualizations in rank_analysis/", flush=True)
    print(f"   2. Read detailed report", flush=True)
    print(f"   3. Run NMF with recommended rank", flush=True)
    print(f"   4. Inspect parts for interpretability", flush=True)

if __name__ == "__main__":
    main() 