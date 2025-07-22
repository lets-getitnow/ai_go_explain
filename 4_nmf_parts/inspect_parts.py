#!/usr/bin/env python3
"""
Step 4: Inspect NMF Parts

Analyzes the learned NMF components by showing which positions 
activate each part most strongly and loading the actual board 
positions for visual inspection.
"""

import numpy as np
import json
import os

def load_nmf_results():
    """Load the NMF results from run_nmf.py."""
    components = np.load("nmf_components.npy")
    activations = np.load("nmf_activations.npy") 
    
    with open("nmf_meta.json", 'r') as f:
        meta = json.load(f)
    
    return components, activations, meta

def load_position_mapping():
    """Load the mapping from position indices to .npz files."""
    pos_file = "../3_extract_activations/activations/pos_index_to_npz.txt"
    
    with open(pos_file, 'r') as f:
        npz_files = [line.strip() for line in f if line.strip()]
    
    return npz_files

def load_board_position(npz_path):
    """
    Load and decode a board position from an .npz file.
    Returns a simple representation for inspection.
    """
    try:
        data = np.load(npz_path)
        
        # Look for common keys in KataGo training data
        possible_keys = ['pos_tensor', 'board', 'input', 'binaryInputNCHWPacked']
        board_data = None
        
        for key in possible_keys:
            if key in data:
                board_data = data[key]
                break
        
        if board_data is None:
            available_keys = list(data.keys())
            return f"Board data not found. Available keys: {available_keys}"
        
        # Basic info about the position
        info = {
            'shape': board_data.shape,
            'dtype': board_data.dtype,
            'file': os.path.basename(npz_path)
        }
        
        return info
        
    except Exception as e:
        return f"Error loading {npz_path}: {e}"

def analyze_parts(components, activations, npz_files):
    """
    Analyze each NMF component/part.
    
    For each part:
    1. Find which positions activate it most strongly
    2. Show the channel pattern (which channels are important)
    3. Load the corresponding board positions
    """
    
    n_parts = components.shape[0]
    
    print("=== NMF Parts Analysis ===")
    print(f"Found {n_parts} parts from {len(npz_files)} positions")
    print()
    
    for part_idx in range(n_parts):
        print(f"--- Part {part_idx} ---")
        
        # Get this part's channel weights
        part_weights = components[part_idx]
        
        # Find top channels for this part
        top_channel_indices = np.argsort(part_weights)[-10:][::-1]  # Top 10 channels
        top_channel_weights = part_weights[top_channel_indices]
        
        print(f"Top channels: {list(zip(top_channel_indices, top_channel_weights))}")
        
        # Find positions where this part activates strongly
        part_activations = activations[:, part_idx]
        position_rankings = np.argsort(part_activations)[::-1]  # Highest first
        
        print(f"Position activations: {part_activations}")
        
        # Show top positions for this part
        print("Strongest positions:")
        for rank, pos_idx in enumerate(position_rankings):
            activation_strength = part_activations[pos_idx]
            npz_file = npz_files[pos_idx]
            
            # Load board position info
            board_info = load_board_position(npz_file)
            
            print(f"  #{rank+1}: Position {pos_idx} (strength: {activation_strength:.4f})")
            print(f"       File: {os.path.basename(npz_file)}")
            print(f"       Board: {board_info}")
        
        print()

def generate_summary_report(components, activations, npz_files):
    """Generate a summary report of the parts."""
    
    report_lines = []
    report_lines.append("# NMF Parts Summary Report")
    report_lines.append(f"Generated: {json.dumps(datetime.now().isoformat())}")
    report_lines.append("")
    
    n_parts = components.shape[0]
    
    for part_idx in range(n_parts):
        report_lines.append(f"## Part {part_idx}")
        
        # Part statistics
        part_weights = components[part_idx]
        sparsity = np.sum(part_weights == 0) / len(part_weights)
        max_weight = np.max(part_weights)
        
        report_lines.append(f"- Sparsity: {sparsity:.2%}")
        report_lines.append(f"- Max weight: {max_weight:.4f}")
        
        # Position rankings
        part_activations = activations[:, part_idx]
        best_pos_idx = np.argmax(part_activations)
        best_activation = part_activations[best_pos_idx]
        
        report_lines.append(f"- Strongest activation: Position {best_pos_idx} ({best_activation:.4f})")
        report_lines.append(f"- File: {os.path.basename(npz_files[best_pos_idx])}")
        report_lines.append("")
    
    # Save report
    with open("parts_summary.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to parts_summary.md")

def main():
    print("=== Inspecting NMF Parts ===")
    
    # Load results
    components, activations, meta = load_nmf_results()
    npz_files = load_position_mapping()
    
    print(f"Loaded {components.shape[0]} parts from {len(npz_files)} positions")
    print(f"Each part has {components.shape[1]} channel weights")
    print()
    
    # Analyze each part
    analyze_parts(components, activations, npz_files)
    
    # Generate summary
    from datetime import datetime
    generate_summary_report(components, activations, npz_files)
    
    print("=== Next Steps ===")
    print("1. Examine the position files manually to understand patterns")
    print("2. Look for Go-specific features (atari, ladders, eyes, etc.)")
    print("3. Consider collecting more positions if patterns aren't clear")

if __name__ == "__main__":
    main() 