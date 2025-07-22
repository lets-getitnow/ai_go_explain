#!/usr/bin/env python3
"""
Correlate board positions with SGF games and move numbers.

Examine the structure of .npz files to understand how to map
board positions back to specific moves in SGF games.
"""

import numpy as np
import os

def examine_npz_structure():
    """Examine what data is available in .npz files."""
    print("=== Examining .npz file structure ===")
    
    tdata_dir = "../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz/tdata"
    npz_files = [f for f in os.listdir(tdata_dir) if f.endswith('.npz')]
    
    for npz_file in npz_files[:1]:  # Just examine one file first
        print(f"\nFile: {npz_file}")
        
        data = np.load(os.path.join(tdata_dir, npz_file))
        
        print(f"Available keys: {list(data.keys())}")
        
        for key in data.keys():
            array = data[key]
            print(f"  {key}: shape={array.shape}, dtype={array.dtype}")
            
            # Show a few sample values for metadata fields
            if array.ndim == 1 and len(array) < 20:
                print(f"    Sample values: {array[:5]}")
            elif array.ndim == 1:
                print(f"    Sample values: {array[:3]} ... {array[-2:]}")

def decode_metadata():
    """Decode the metadata arrays to understand game/move correlation."""
    print("\n=== Decoding Metadata Arrays ===")
    
    tdata_dir = "../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz/tdata"
    npz_files = [f for f in os.listdir(tdata_dir) if f.endswith('.npz')]
    
    # Examine the first file in detail
    npz_file = npz_files[0]
    print(f"\nDetailed analysis of: {npz_file}")
    
    data = np.load(os.path.join(tdata_dir, npz_file))
    
    # Look at globalInputNC - likely contains game metadata
    global_input = data['globalInputNC']
    print(f"\nglobalInputNC analysis:")
    print(f"  Shape: {global_input.shape}")
    print(f"  First 5 positions, all 19 channels:")
    for i in range(5):
        print(f"    Position {i}: {global_input[i]}")
    
    # Look at policyTargetsNCMove - contains move information
    policy_targets = data['policyTargetsNCMove']
    print(f"\npolicyTargetsNCMove analysis:")
    print(f"  Shape: {policy_targets.shape}")
    print(f"  First 5 positions:")
    for i in range(5):
        print(f"    Position {i}: {policy_targets[i]}")
    
    # Look at qValueTargetsNCMove
    qvalue_targets = data['qValueTargetsNCMove']
    print(f"\nqValueTargetsNCMove analysis:")
    print(f"  Shape: {qvalue_targets.shape}")
    print(f"  First 5 positions:")
    for i in range(5):
        print(f"    Position {i}: {qvalue_targets[i]}")
    
    return data

def analyze_nmf_positions():
    """Analyze the specific positions that activate strongly in NMF parts."""
    print("\n=== Analyzing NMF Strong Activation Positions ===")
    
    # Load the positions we care about from your previous analysis
    strong_positions = [
        # Part 0 top positions
        {'part': 0, 'rank': 1, 'global_pos': 1388, 'file': '5DC45ABC0B69CCEF.npz', 'pos_in_file': 100},
        {'part': 0, 'rank': 2, 'global_pos': 2442, 'file': 'E8EA0728D892E382.npz', 'pos_in_file': 717},
        # Part 1 top positions  
        {'part': 1, 'rank': 1, 'global_pos': 1256, 'file': '4A3919085F2F7840.npz', 'pos_in_file': 968},
        {'part': 1, 'rank': 2, 'global_pos': 1254, 'file': '4A3919085F2F7840.npz', 'pos_in_file': 966},
        # Part 2 top positions
        {'part': 2, 'rank': 1, 'global_pos': 834, 'file': '4A3919085F2F7840.npz', 'pos_in_file': 546},
    ]
    
    tdata_dir = "../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz/tdata"
    
    for pos_info in strong_positions:
        print(f"\nPart {pos_info['part']}, Rank {pos_info['rank']}:")
        print(f"  Global position: {pos_info['global_pos']}")
        print(f"  File: {pos_info['file']}, Position in file: {pos_info['pos_in_file']}")
        
        # Load the metadata for this specific position
        file_path = os.path.join(tdata_dir, pos_info['file'])
        data = np.load(file_path)
        
        pos_idx = pos_info['pos_in_file']
        
        # Extract metadata for this position
        global_input = data['globalInputNC'][pos_idx]
        policy_target = data['policyTargetsNCMove'][pos_idx] 
        qvalue_target = data['qValueTargetsNCMove'][pos_idx]
        
        print(f"  globalInputNC: {global_input}")
        print(f"  policyTargetsNCMove: {policy_target}")
        print(f"  qValueTargetsNCMove: {qvalue_target}")
        
        # Try to decode any obvious patterns
        print(f"  Global input non-zero channels: {np.where(global_input != 0)[0]}")
        print(f"  Global input non-zero values: {global_input[global_input != 0]}")
    
    return strong_positions

def examine_sgf_structure():
    """Look at the SGF file structure."""
    print("\n=== Examining SGF file structure ===")
    
    sgf_path = "../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz/sgfs/EE2D166B1D652580.sgfs"
    
    with open(sgf_path, 'r') as f:
        content = f.read()
    
    # Count games (each starts with "(;")
    games = content.split('(;')[1:]  # Skip first empty part
    print(f"Number of games in SGF file: {len(games)}")
    
    # Show structure of first game
    if games:
        first_game = "(;" + games[0]
        lines = first_game.split('\n')
        print(f"\nFirst game preview (first 10 lines):")
        for i, line in enumerate(lines[:10]):
            print(f"  {i+1}: {line[:80]}...")

def find_correlations():
    """Try to find correlations between .npz and .sgf data."""
    print("\n=== Looking for correlations ===")
    
    # The key insight: we need to understand if .npz files contain:
    # 1. Game IDs that match SGF games
    # 2. Move numbers within each game
    # 3. Any other metadata that links them
    
    tdata_dir = "../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz/tdata"
    
    # Look for patterns in filenames
    npz_files = [f for f in os.listdir(tdata_dir) if f.endswith('.npz')]
    sgf_file = "EE2D166B1D652580.sgfs"
    
    print(f"NPZ files: {npz_files}")
    print(f"SGF file: {sgf_file}")
    
    # Check if any NPZ filenames relate to SGF filename
    sgf_base = sgf_file.replace('.sgfs', '')
    print(f"SGF base name: {sgf_base}")
    
    for npz_file in npz_files:
        npz_base = npz_file.replace('.npz', '')
        print(f"NPZ base: {npz_base}")

if __name__ == "__main__":
    examine_npz_structure()
    data = decode_metadata()
    strong_positions = analyze_nmf_positions()
    examine_sgf_structure()
    find_correlations() 