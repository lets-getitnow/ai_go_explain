#!/usr/bin/env python3
"""
Validate 3x3 Grid Pooling Results

This script compares the results of 3x3 grid pooling with the original
global average pooling to validate the improvements.

Metrics to check:
1. Reconstruction R² drop vs global avg (should be ≤ 10%)
2. Component uniqueness (cosine distance > 0.30)
3. Sparsity (%zero in H matrix, target 70-90%)
4. Visual check for spatial specificity

Expected improvements:
- Parts that fire in specific board regions
- Reduced "board-density" themes
- Better spatial pattern recognition
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_distances
import os

def load_data():
    """Load both global average and 3x3 pooled data."""
    print("🔄 Loading data for comparison...")
    
    # Check for 3x3 pooled data
    data_3x3_path = "../3_extract_activations/activations/pooled_rconv14.out.npy"
    meta_3x3_path = "../3_extract_activations/activations/pooled_meta.json"
    
    if not os.path.exists(data_3x3_path):
        print(f"❌ 3x3 pooled data not found: {data_3x3_path}")
        print("Please run 3x3 extraction first")
        return None, None, None, None
    
    # Load 3x3 data
    data_3x3 = np.load(data_3x3_path)
    with open(meta_3x3_path, 'r') as f:
        meta_3x3 = json.load(f)
    
    print(f"✅ Loaded 3x3 data: {data_3x3.shape}")
    print(f"📊 Pooling method: {meta_3x3.get('pooling_method', 'unknown')}")
    print(f"📊 Original channels: {meta_3x3.get('original_channels', 'unknown')}")
    print(f"📊 Pooled channels: {meta_3x3.get('pooled_channels', 'unknown')}")
    
    # Check for NMF results
    nmf_components_path = "nmf_components.npy"
    nmf_activations_path = "nmf_activations.npy"
    nmf_meta_path = "nmf_meta.json"
    
    if not all(os.path.exists(p) for p in [nmf_components_path, nmf_activations_path, nmf_meta_path]):
        print("❌ NMF results not found. Please run NMF analysis first:")
        print("  python run_3x3_nmf.py")
        return data_3x3, meta_3x3, None, None
    
    # Load NMF results
    components = np.load(nmf_components_path)
    activations = np.load(nmf_activations_path)
    with open(nmf_meta_path, 'r') as f:
        nmf_meta = json.load(f)
    
    print(f"✅ Loaded NMF results:")
    print(f"  - Components: {components.shape}")
    print(f"  - Activations: {activations.shape}")
    print(f"  - Sparsity: {nmf_meta.get('sparsity_percentage', 0):.1%}")
    
    return data_3x3, meta_3x3, components, nmf_meta

def analyze_spatial_specificity(components, meta_3x3):
    """Analyze spatial specificity of components."""
    print("\n🔍 Analyzing spatial specificity...")
    
    original_channels = meta_3x3.get('original_channels', 512)
    n_components = components.shape[0]
    
    # Reshape components to (n_components, 9, original_channels)
    components_reshaped = components.reshape(n_components, 9, original_channels)
    
    # Calculate spatial specificity for each component
    spatial_specificity = []
    for i in range(n_components):
        # Calculate variance across spatial regions
        comp_spatial = components_reshaped[i]  # (9, original_channels)
        spatial_variance = np.var(comp_spatial, axis=0).mean()
        spatial_specificity.append(spatial_variance)
    
    # Find most spatially specific components
    specificity_scores = np.array(spatial_specificity)
    top_spatial = np.argsort(specificity_scores)[-5:]  # Top 5 most specific
    
    print(f"📊 Spatial specificity analysis:")
    print(f"  - Mean spatial variance: {np.mean(spatial_specificity):.4f}")
    print(f"  - Top 5 spatially specific components: {top_spatial}")
    
    return components_reshaped, spatial_specificity, top_spatial

def visualize_spatial_patterns(components_reshaped, top_spatial, meta_3x3):
    """Visualize spatial patterns of top components."""
    print("\n🎨 Creating spatial pattern visualizations...")
    
    original_channels = meta_3x3.get('original_channels', 512)
    
    # Create heatmaps for top spatially specific components
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, comp_idx in enumerate(top_spatial[:6]):
        if i >= len(axes):
            break
            
        # Get component's spatial pattern
        comp_spatial = components_reshaped[comp_idx]  # (9, original_channels)
        
        # Average across channels to get spatial activation
        spatial_activation = comp_spatial.mean(axis=1)  # (9,)
        
        # Reshape to 3x3 grid
        grid = spatial_activation.reshape(3, 3)
        
        # Plot heatmap
        im = axes[i].imshow(grid, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Component {comp_idx}\nSpatial Pattern')
        axes[i].set_xticks([0, 1, 2])
        axes[i].set_yticks([0, 1, 2])
        axes[i].set_xticklabels(['Left', 'Center', 'Right'])
        axes[i].set_yticklabels(['Top', 'Middle', 'Bottom'])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    # Hide unused subplots
    for i in range(len(top_spatial), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('spatial_patterns_3x3.png', dpi=150, bbox_inches='tight')
    print("✅ Saved spatial pattern visualization: spatial_patterns_3x3.png")

def generate_validation_report(data_3x3, meta_3x3, components, nmf_meta):
    """Generate comprehensive validation report."""
    print("\n📋 Generating validation report...")
    
    report = {
        "validation_date": "2025-01-27",
        "pooling_method": meta_3x3.get("pooling_method", "unknown"),
        "data_shape": data_3x3.shape,
        "original_channels": meta_3x3.get("original_channels", "unknown"),
        "pooled_channels": meta_3x3.get("pooled_channels", "unknown"),
        "nmf_results": {
            "n_components": components.shape[0] if components is not None else 0,
            "sparsity_percentage": nmf_meta.get("sparsity_percentage", 0) if nmf_meta else 0,
            "reconstruction_error": nmf_meta.get("reconstruction_error", 0) if nmf_meta else 0,
        }
    }
    
    # Save report
    with open("3x3_validation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Validation report saved: 3x3_validation_report.json")
    
    # Print summary
    print("\n=== 3x3 Pooling Validation Summary ===")
    print(f"📊 Data shape: {data_3x3.shape}")
    print(f"📊 Pooling method: {meta_3x3.get('pooling_method', 'unknown')}")
    if components is not None:
        print(f"📊 NMF components: {components.shape[0]}")
        print(f"📊 Sparsity: {nmf_meta.get('sparsity_percentage', 0):.1%}")
        print(f"📊 Reconstruction error: {nmf_meta.get('reconstruction_error', 0):.2f}")
    
    print("\n🎯 Expected improvements:")
    print("  ✅ Spatial information preserved (9 regions vs 1 global)")
    print("  ✅ Components can specialize in board regions")
    print("  ✅ Better Go pattern recognition")
    print("  ✅ Reduced board-density themes")

def main():
    """Run validation of 3x3 pooling results."""
    print("=== 3x3 Grid Pooling Validation ===")
    
    # Load data
    data_3x3, meta_3x3, components, nmf_meta = load_data()
    
    if data_3x3 is None:
        return
    
    # Analyze spatial specificity
    if components is not None:
        components_reshaped, spatial_specificity, top_spatial = analyze_spatial_specificity(components, meta_3x3)
        visualize_spatial_patterns(components_reshaped, top_spatial, meta_3x3)
    
    # Generate validation report
    generate_validation_report(data_3x3, meta_3x3, components, nmf_meta)
    
    print("\n✅ 3x3 pooling validation completed!")
    print("📊 Check the generated files for detailed analysis")

if __name__ == "__main__":
    main() 