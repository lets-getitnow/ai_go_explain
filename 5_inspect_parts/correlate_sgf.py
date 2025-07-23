#!/usr/bin/env python3
"""
Correlate board positions with SGF games and move numbers.

Examine the structure of .npz files to understand how to map
board positions back to specific moves in SGF games.
"""

# -*- coding: utf-8 -*-
import numpy as np
import os
import json
# SGF generation helper

def coord_to_sgf(coord):
    # coord as (row,col) zero-index -> sgf two letters a-g
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return letters[coord[1]] + letters[coord[0]]

def board_to_sgf(black_set, white_set, next_move_coord, board_size=7):
    props = [f"(;FF[4]GM[1]SZ[{board_size}]KM[7.5]"]
    if black_set:
        ab = ''.join([f"[{coord_to_sgf(c)}]" for c in black_set])
        props.append(f"AB{ab}")
    if white_set:
        aw = ''.join([f"[{coord_to_sgf(c)}]" for c in white_set])
        props.append(f"AW{aw}")
    # add next move as mainline
    if next_move_coord is not None:
        color = 'B' if len(black_set) == len(white_set) else 'W'
        props.append(f"{')'.join([])}")
    header = ''.join(props) + ")"
    if next_move_coord is None:
        return header
    move_prop = f";{color}[{coord_to_sgf(next_move_coord)}])"
    return header[:-1] + move_prop

# --- helper functions for move decoding ---
BOARD_SIZE = 7
PASS_INDEX = BOARD_SIZE * BOARD_SIZE  # 49 for 7x7


def decode_move(policy_targets: np.ndarray) -> int:
    """Return move index with highest count from channel 1 (chosen move)."""
    return int(policy_targets[1].argmax())


def idx_to_coord(idx: int) -> str:
    if idx == PASS_INDEX:
        return "PASS"
    row, col = divmod(idx, BOARD_SIZE)
    return f"({row},{col})"


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
    
    base_dir = "../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz"
    tdata_dir = os.path.join(base_dir, "tdata")
    sgf_dir = os.path.join(base_dir, "sgfs")
    sgf_files = [f for f in os.listdir(sgf_dir) if f.endswith(".sgfs")]
    sgf_file = sgf_files[0] if sgf_files else "UNKNOWN.sgfs"
    
    for pos_info in strong_positions:
        print(f"\nPart {pos_info['part']}, Rank {pos_info['rank']}:")
        print(f"  Global position: {pos_info['global_pos']}")
        print(f"  File: {pos_info['file']}, Position in file: {pos_info['pos_in_file']}")
        
        # Load the metadata for this specific position
        file_path = os.path.join(tdata_dir, pos_info['file'])
        data = np.load(file_path)
        
        pos_idx = pos_info['pos_in_file']
        
        # Decode chosen move
        policy_target = data['policyTargetsNCMove'][pos_idx]
        move_idx = decode_move(policy_target)
        coord = idx_to_coord(move_idx)

        pos_info['move_idx'] = move_idx
        pos_info['coord'] = coord
        pos_info['sgf'] = sgf_file

        # Print concise info
        print(f"  Chosen move index: {move_idx}  -> coord {coord} (sgf {sgf_file})")
    
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

    # --- Determine SGF directory ---
    try:
        sgf_dir  # type: ignore  # already defined in analyze_nmf_positions
    except NameError:
        base_dir = "../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz"
        sgf_dir = os.path.join(base_dir, "sgfs") if os.path.isdir(base_dir) else "."

    # determine sgf_file (single bundle with many games)
    candidates = [f for f in os.listdir(sgf_dir) if f.endswith('.sgfs')]
    if not candidates:
        raise RuntimeError("No .sgfs file found in sgf_dir")
    sgf_file = candidates[0]

    sgf_path = os.path.join(sgf_dir, sgf_file)
    with open(sgf_path, 'r') as f:
        sgf_raw = f.read()

    # --- Split bundle into individual games ---
    def split_games(text: str):
        games = []
        buf = []
        depth = 0
        for ch in text:
            if ch == '(': depth += 1
            if depth>0: buf.append(ch)
            if ch == ')':
                depth -= 1
                if depth==0 and buf:
                    games.append(''.join(buf))
                    buf=[]
        return games

    games = split_games(sgf_raw.strip())
    print(f"Loaded {len(games)} SGF games from {sgf_file}")

    import re
    move_counts=[len(re.findall(r';[BW]\[',g)) for g in games]
    cum=0
    boundaries=[]
    for c in move_counts:
        boundaries.append(cum)
        cum+=c
    total_positions=cum
    print(f"Total positions represented by games: {total_positions}")

    # --- Save clipped SGFs ---
    for pos in strong_positions:
        gpos=pos['global_pos']
        if gpos>=total_positions:
            raise RuntimeError(f"global_pos {gpos} exceeds total positions {total_positions}")
        # binary search for game index
        import bisect
        game_idx=bisect.bisect_right(boundaries,gpos)-1
        turn=gpos-boundaries[game_idx]
        game_text=games[game_idx]
        parts=game_text.split(';')
        header=parts[0]
        moves=parts[1:turn+1]
        clipped=';'.join([header]+moves)+')'
        out_name=f"sgf_pos{gpos}.sgf"
        with open(out_name,'w') as f:
            f.write(clipped)
        print(f"  Wrote SGF to {out_name} (game {game_idx}, turn {turn})")
    examine_sgf_structure()
    find_correlations() 