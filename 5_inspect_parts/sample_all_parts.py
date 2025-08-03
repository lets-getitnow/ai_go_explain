#!/usr/bin/env python3
"""
Sample positions from all parts to ensure coverage.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any

def load_nmf_data(nmf_dir: Path):
    """Load NMF data from directory."""
    activations_file = nmf_dir / "nmf_activations.npy"
    components_file = nmf_dir / "nmf_components.npy"
    
    activations = np.load(activations_file)
    components = np.load(components_file) if components_file.exists() else None
    
    print(f"Loaded activations: {activations.shape}")
    if components is not None:
        print(f"Loaded components: {components.shape}")
    
    return activations, components

def sample_positions_per_part(activations: np.ndarray, positions_per_part: int = 20) -> List[Dict[str, Any]]:
    """Sample the strongest positions for each part."""
    n_parts = activations.shape[1]
    n_positions = activations.shape[0]
    
    sampled_positions = []
    
    for part_idx in range(n_parts):
        print(f"Sampling Part {part_idx}...")
        
        # Get strongest positions for this part
        part_activations = activations[:, part_idx]
        top_indices = np.argsort(part_activations)[-positions_per_part:][::-1]
        
        for rank, position_idx in enumerate(top_indices, 1):
            activation_strength = float(part_activations[position_idx])
            
            # Calculate percentile
            less_than_count = np.sum(part_activations < activation_strength)
            total_count = len(part_activations)
            activation_percentile = (less_than_count / total_count) * 100.0
            
            sampled_positions.append({
                'part_idx': part_idx,
                'position_idx': int(position_idx),
                'rank': rank,
                'activation_strength': activation_strength,
                'activation_percentile': activation_percentile
            })
    
    return sampled_positions

def main():
    parser = argparse.ArgumentParser(description="Sample positions from all parts")
    parser.add_argument("--nmf-dir", type=Path, required=True, help="Directory containing NMF data")
    parser.add_argument("--output-file", type=Path, required=True, help="Output CSV file")
    parser.add_argument("--positions-per-part", type=int, default=20, help="Number of positions per part")
    
    args = parser.parse_args()
    
    print("=== Sampling Positions from All Parts ===")
    
    # Load NMF data
    activations, components = load_nmf_data(args.nmf_dir)
    
    # Sample positions from each part
    sampled_positions = sample_positions_per_part(activations, args.positions_per_part)
    
    # Create DataFrame
    df = pd.DataFrame(sampled_positions)
    
    # Save to CSV
    df.to_csv(args.output_file, index=False)
    print(f"Saved {len(sampled_positions)} positions to {args.output_file}")
    
    # Print summary
    print("\n=== Summary ===")
    for part_idx in range(activations.shape[1]):
        part_positions = df[df['part_idx'] == part_idx]
        if len(part_positions) > 0:
            max_strength = part_positions['activation_strength'].max()
            min_strength = part_positions['activation_strength'].min()
            print(f"Part {part_idx}: {len(part_positions)} positions, strength range: {min_strength:.3f} - {max_strength:.3f}")

if __name__ == "__main__":
    main() 