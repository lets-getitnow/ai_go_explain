#!/usr/bin/env python3
"""
Step 5: Simple Board Position Examiner

Load the actual board positions where NMF parts activate strongly.
"""

import numpy as np
import json

def load_data():
    """Load all necessary data."""
    # Load NMF results
    activations = np.load("../4_nmf_parts/nmf_activations.npy")
    
    # Load position mapping
    with open("../3_extract_activations/activations/pos_index_to_npz.txt", 'r') as f:
        pos_to_file = [line.strip() for line in f]
    
    return activations, pos_to_file

def get_strongest_positions(part_idx, activations, n_top=3):
    """Get the strongest activating positions for a part."""
    part_activations = activations[:, part_idx]
    top_indices = np.argsort(part_activations)[-n_top:][::-1]
    
    results = []
    for rank, pos_idx in enumerate(top_indices):
        results.append({
            'rank': rank + 1,
            'global_pos': pos_idx,
            'activation': part_activations[pos_idx]
        })
    return results

def calculate_file_position(global_pos, pos_to_file):
    """
    Calculate which position within a file the global position refers to.
    """
    target_file = pos_to_file[global_pos]
    
    # Count how many times this file appears before this position
    position_in_file = 0
    for i in range(global_pos + 1):
        if pos_to_file[i] == target_file:
            if i == global_pos:
                break
            position_in_file += 1
    
    return target_file, position_in_file

def load_board_position(file_path, position_in_file):
    """Load a specific board position from an .npz file."""
    try:
        data = np.load(file_path)
        
        if 'binaryInputNCHWPacked' in data:
            board_data = data['binaryInputNCHWPacked']
            
            if position_in_file < board_data.shape[0]:
                position = board_data[position_in_file]
                return {
                    'success': True,
                    'shape': position.shape,
                    'data': position,
                    'total_positions_in_file': board_data.shape[0]
                }
            else:
                return {'success': False, 'error': f'Position {position_in_file} >= file size {board_data.shape[0]}'}
        else:
            return {'success': False, 'error': f'No binaryInputNCHWPacked in {list(data.keys())}'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    print("=== Step 5: Board Position Analysis ===")
    
    activations, pos_to_file = load_data()
    n_parts = activations.shape[1]
    
    print(f"Analyzing {n_parts} parts across {len(pos_to_file)} positions")
    print()
    
    for part_idx in range(n_parts):
        print(f"=== PART {part_idx} ===")
        
        strongest = get_strongest_positions(part_idx, activations, n_top=3)
        
        for pos_info in strongest:
            global_pos = pos_info['global_pos']
            activation = pos_info['activation']
            rank = pos_info['rank']
            
            # Find which file and position within file
            file_path, pos_in_file = calculate_file_position(global_pos, pos_to_file)
            filename = file_path.split('/')[-1]
            
            print(f"  Rank #{rank}: Activation {activation:.4f}")
            print(f"    Global position: {global_pos}")
            print(f"    File: {filename}")
            print(f"    Position in file: {pos_in_file}")
            
            # Load the actual board data
            board_result = load_board_position(file_path, pos_in_file)
            
            if board_result['success']:
                board_data = board_result['data']
                print(f"    Board shape: {board_result['shape']}")
                print(f"    Channels: {board_data.shape[0]}")
                
                # Save for your analysis
                output_file = f"part{part_idx}_rank{rank}_pos{global_pos}.npy"
                np.save(output_file, board_data)
                print(f"    Saved board data to: {output_file}")
                
                # Basic info about the board state
                print(f"    File contains {board_result['total_positions_in_file']} total positions")
                
            else:
                print(f"    ERROR: {board_result['error']}")
            
            print()
        
        print("-" * 60)
        print()

if __name__ == "__main__":
    main() 