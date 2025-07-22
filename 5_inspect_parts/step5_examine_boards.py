#!/usr/bin/env python3
"""
Step 5: Examine Board Positions

Load and decode actual board positions where each NMF part activates strongly
to look for interpretable Go patterns.
"""

import numpy as np
import json
import os

def load_nmf_results():
    """Load NMF results and position mapping."""
    components = np.load("nmf_components.npy")
    activations = np.load("nmf_activations.npy")
    
    with open("nmf_meta.json", 'r') as f:
        meta = json.load(f)
    
    # Load position to file mapping
    pos_file = "../3_extract_activations/activations/pos_index_to_npz.txt"
    with open(pos_file, 'r') as f:
        npz_files = [line.strip() for line in f if line.strip()]
    
    return components, activations, meta, npz_files

def decode_board_from_npz(npz_path, position_index_in_file):
    """
    Load and decode a specific board position from an .npz file.
    Returns the board state in a readable format.
    """
    full_path = f"../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz/tdata/{npz_path}"
    
    try:
        data = np.load(full_path)
        
        # KataGo training data typically uses 'binaryInputNCHWPacked'
        if 'binaryInputNCHWPacked' in data:
            board_data = data['binaryInputNCHWPacked']
            
            # Shape is typically (batch_size, channels, height, width)
            print(f"Raw shape: {board_data.shape}, dtype: {board_data.dtype}")
            
            # Get the specific position
            if position_index_in_file < board_data.shape[0]:
                position = board_data[position_index_in_file]
                
                # Decode the board state
                # Channel 0 and 1 are typically black and white stones
                # Height/width are typically 19x19 but stored as 22x7 (packed format)
                return {
                    'shape': position.shape,
                    'channels': position.shape[0],
                    'raw_data': position,
                    'file': npz_path,
                    'position_in_file': position_index_in_file
                }
            else:
                return f"Position {position_index_in_file} not found in file (max: {board_data.shape[0]})"
        
        else:
            available_keys = list(data.keys())
            return f"binaryInputNCHWPacked not found. Available keys: {available_keys}"
            
    except Exception as e:
        return f"Error loading {npz_path}: {e}"

def find_top_positions_for_part(part_idx, activations, npz_files, n_top=5):
    """Find the top N positions where a part activates strongly."""
    part_activations = activations[:, part_idx]
    top_indices = np.argsort(part_activations)[-n_top:][::-1]  # Top N, highest first
    
    results = []
    for rank, pos_idx in enumerate(top_indices):
        activation_strength = part_activations[pos_idx]
        npz_file = npz_files[pos_idx]
        
        # Calculate which position within the file this corresponds to
        # We need to figure out the cumulative position mapping
        position_in_file = calculate_position_in_file(pos_idx, npz_files)
        
        results.append({
            'rank': rank + 1,
            'global_position': pos_idx,
            'activation': activation_strength,
            'file': npz_file,
            'position_in_file': position_in_file
        })
    
    return results

def calculate_position_in_file(global_pos, npz_files):
    """
    Calculate which position within a specific .npz file a global position index refers to.
    This requires understanding the cumulative structure.
    """
    # Count positions in each file to determine the mapping
    file_sizes = {}
    cumulative = 0
    
    for i, npz_file in enumerate(npz_files):
        if npz_file not in file_sizes:
            # Try to determine file size
            full_path = f"../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz/tdata/{npz_file}"
            try:
                data = np.load(full_path)
                if 'binaryInputNCHWPacked' in data:
                    file_sizes[npz_file] = data['binaryInputNCHWPacked'].shape[0]
                else:
                    file_sizes[npz_file] = 1  # fallback
            except:
                file_sizes[npz_file] = 1  # fallback
        
        if i == global_pos:
            # Find which file this global position maps to
            current_cumulative = 0
            for j, file_name in enumerate(npz_files[:i+1]):
                if file_name == npz_file:
                    return global_pos - current_cumulative
                current_cumulative += 1
    
    # Simplified: assume each entry in npz_files represents one position
    return 0

def main():
    print("=== Step 5: Examining Board Positions for NMF Parts ===")
    
    # Load results
    components, activations, meta, npz_files = load_nmf_results()
    n_parts = components.shape[0]
    
    print(f"Loaded {n_parts} parts from {len(npz_files)} positions")
    print()
    
    # Examine top positions for each part
    for part_idx in range(n_parts):
        print(f"=== PART {part_idx} ===")
        
        top_positions = find_top_positions_for_part(part_idx, activations, npz_files, n_top=3)
        
        for pos_info in top_positions:
            print(f"Rank #{pos_info['rank']}: Activation {pos_info['activation']:.4f}")
            print(f"  Global position: {pos_info['global_position']}")
            print(f"  File: {pos_info['file']}")
            print(f"  Position in file: {pos_info['position_in_file']}")
            
            # Load and decode the board
            board_info = decode_board_from_npz(pos_info['file'], pos_info['position_in_file'])
            
            if isinstance(board_info, dict):
                print(f"  Board shape: {board_info['shape']}")
                print(f"  Channels: {board_info['channels']}")
                
                # For Go expert analysis, you can examine board_info['raw_data']
                # The first few channels typically represent:
                # 0: Black stones, 1: White stones, 2-3: Recent moves, etc.
                print(f"  Raw data available for analysis")
                
                # Optionally save for detailed inspection
                output_file = f"part_{part_idx}_rank_{pos_info['rank']}_pos_{pos_info['global_position']}.npy"
                np.save(output_file, board_info['raw_data'])
                print(f"  Saved to: {output_file}")
                
            else:
                print(f"  Error: {board_info}")
            
            print()
        
        print("-" * 50)
        print()

if __name__ == "__main__":
    main() 