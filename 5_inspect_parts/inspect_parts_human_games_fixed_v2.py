#!/usr/bin/env python3
"""
Step 5 – Inspect Parts (Human Games Version) - FIXED V2
=======================================================

Fixed version that works with board data from NPZ files instead of SGF files.

Expected Working Directory
-------------------------
This script expects to be run from the project root directory:
    /Users/hunterp/dev/ai_go_explain

Path Assumptions
----------------
- NPZ files: <output_dir>/npz_files/ (from pipeline)
- NMF data: <output_dir>/nmf_parts/ (from pipeline)

Usage
-----
python3 5_inspect_parts/inspect_parts_human_games_fixed_v2.py \
    --nmf-dir test/nmf_parts \
    --npz-dir test/npz_files \
    --output-dir test/inspect_parts \
    --positions-per-part 20 \
    --board-size 13
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re

# Constants
BOARD_SIZE = 13

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_nmf_data(nmf_dir: Path):
    """Load NMF data from directory."""
    activations_file = nmf_dir / "nmf_activations.npy"
    components_file = nmf_dir / "nmf_components.npy"
    
    if not activations_file.exists():
        raise FileNotFoundError(f"Activations file not found: {activations_file}")
    
    activations = np.load(activations_file)
    components = np.load(components_file) if components_file.exists() else None
    
    print(f"Loaded activations: {activations.shape}")
    if components is not None:
        print(f"Loaded components: {components.shape}")
    
    return activations, components

def load_board_data(npz_dir: Path) -> Dict[str, Any]:
    """Load board data from NPZ files."""
    board_data = {}
    
    print(f"Looking for NPZ files in: {npz_dir}")
    npz_files = list(npz_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files")
    
    for npz_file in npz_files:
        print(f"Processing NPZ file: {npz_file.name}")
        data = np.load(npz_file, allow_pickle=True)
        
        game_id = data['game_id'][0].decode('utf-8') if len(data['game_id']) > 0 else npz_file.stem
        
        # Extract board data
        if 'binaryInputNCHWPacked' in data:
            board_data[game_id] = {
                'board_data': data['binaryInputNCHWPacked'],
                'policy_data': data.get('policyTargetsNCMove', None),
                'value_data': data.get('valueTargetsNC', None),
                'game_id': game_id
            }
            print(f"  Loaded {len(data['binaryInputNCHWPacked'])} positions")
    
    print(f"Loaded {len(board_data)} games")
    return board_data

def unpack_board_data(packed_data: np.ndarray) -> np.ndarray:
    """Unpack binary board data from packed format."""
    # Unpack the bits
    unpacked = np.unpackbits(packed_data, axis=2)
    
    # For 13x13 board, we expect 176 bits (13*13 + 7 padding)
    # Reshape to (batch, channels, height, width)
    batch_size, channels, packed_size = unpacked.shape
    expected_size = (BOARD_SIZE * BOARD_SIZE + 7) // 8 * 8  # 176 bits
    
    if packed_size != expected_size:
        raise ValueError(f"Expected {expected_size} bits, got {packed_size}")
    
    # Reshape to (batch, channels, 13, 13)
    unpacked = unpacked[:, :, :BOARD_SIZE*BOARD_SIZE]  # Remove padding
    unpacked = unpacked.reshape(batch_size, channels, BOARD_SIZE, BOARD_SIZE)
    
    return unpacked

def board_to_sgf(board_data: np.ndarray, position_idx: int) -> str:
    """Convert board data to SGF format."""
    # Extract the specific position
    if len(board_data.shape) == 4:  # (batch, channels, height, width)
        position = board_data[position_idx]
    else:
        position = board_data
    
    # Create SGF header
    sgf_parts = [f"(;FF[4]GM[1]SZ[{BOARD_SIZE}]"]
    
    # Add board state (stones)
    # This is a simplified version - in practice you'd need to decode the board channels
    # For now, create a basic SGF with a comment
    sgf_parts.append("C[Board position from NPZ data])")
    
    return "".join(sgf_parts)

def analyze_position(position_idx: int, part_idx: int, activations: np.ndarray, 
                   board_data: Dict[str, Any], board_size: int = 13, 
                   components: np.ndarray = None) -> Dict[str, Any]:
    """Analyze a specific position."""
    # Get activation strength for this position and part
    activation_strength = float(activations[position_idx, part_idx])
    
    # Get game data (assuming single game for now)
    game_id = list(board_data.keys())[0]
    game_info = board_data[game_id]
    
    # Create SGF content for this position
    board_tensor = game_info['board_data']
    sgf_content = board_to_sgf(board_tensor, position_idx)
    
    # Calculate activation percentile
    all_activations = activations[:, part_idx]
    if np.max(all_activations) == np.min(all_activations):
        activation_percentile = 50.0
    else:
        less_than_count = np.sum(all_activations < activation_strength)
        total_count = len(all_activations)
        activation_percentile = (less_than_count / total_count) * 100.0
    
    # Calculate uniqueness score and part comparison
    uniqueness_score = calculate_uniqueness_score(activations, position_idx, part_idx)
    part_comparison = calculate_part_comparison(activations, position_idx, part_idx)
    
    # Calculate channel activity if components are available
    channel_activity = []
    if components is not None:
        channel_activity = calculate_channel_activity(components, part_idx, board_size)
    
    return {
        'position_idx': position_idx,
        'part_idx': part_idx,
        'activation_strength': activation_strength,
        'activation_percentile': activation_percentile,
        'sgf_content': sgf_content,
        'move_coord': "Unknown",
        'turn_number': position_idx,
        'game_id': game_id,
        'total_moves': len(game_info['board_data']),
        'policy_analysis': {},
        'value_analysis': {},
        'uniqueness_score': uniqueness_score,
        'part_comparison': part_comparison,
        'channel_activity': channel_activity
    }

def calculate_uniqueness_score(activations: np.ndarray, position_idx: int, part_idx: int) -> float:
    """Calculate how unique this activation is compared to other parts."""
    current_activation = activations[position_idx, part_idx]
    other_activations = [activations[position_idx, i] for i in range(activations.shape[1]) if i != part_idx]
    max_other = max(other_activations) if other_activations else 0.0
    
    if current_activation + max_other > 0:
        return float(current_activation / (current_activation + max_other))
    return 0.0

def calculate_channel_activity(components: np.ndarray, part_idx: int, board_size: int = 13) -> List[Dict[str, Any]]:
    """Calculate which channels are most active for this part."""
    if components is None:
        return []
    
    # Get the component weights for this part
    component_weights = components[part_idx]  # Shape: (channels,)
    
    # Find channels with highest weights
    top_indices = np.argsort(component_weights)[-10:][::-1]  # Top 10
    
    channel_activity = []
    for idx in top_indices:
        weight = float(component_weights[idx])
        if weight > 0.01:  # Only include significant channels
            channel_activity.append({
                'channel': int(idx),
                'weight': weight
            })
    
    return channel_activity

def calculate_part_comparison(activations: np.ndarray, position_idx: int, part_idx: int) -> Dict[str, Any]:
    """Calculate comparison with other parts."""
    current_activation = activations[position_idx, part_idx]
    part_activations = activations[:, part_idx]
    
    # Find similar positions (other high-activating positions for this part)
    similar_indices = np.argsort(part_activations)[-10:][::-1]
    similar_positions = [int(idx) for idx in similar_indices if idx != position_idx][:5]
    
    # Ranking across all positions for this part
    position_rank = np.sum(part_activations > current_activation) + 1
    
    return {
        'similar_positions': similar_positions,
        'part_rank': int(position_rank),
        'activation_percentile': float(100 * (1 - position_rank / len(part_activations)))
    }

def generate_csv_summary(analyses: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate CSV summary of all analyses."""
    summary_data = []
    
    for analysis in analyses:
        summary_data.append({
            'part_idx': analysis['part_idx'],
            'position_idx': analysis['position_idx'],
            'activation_strength': analysis['activation_strength'],
            'activation_percentile': analysis['activation_percentile'],
            'uniqueness_score': analysis['uniqueness_score'],
            'part_rank': analysis['part_comparison']['part_rank'],
            'game_id': analysis['game_id'],
            'turn_number': analysis['turn_number'],
            'sgf_content': analysis['sgf_content']
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = output_dir / "strong_positions_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Inspect NMF parts for human games")
    parser.add_argument("--nmf-dir", type=Path, required=True, help="Directory containing NMF data")
    parser.add_argument("--npz-dir", type=Path, required=True, help="Directory containing NPZ files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--positions-per-part", type=int, default=20, help="Number of positions to analyze per part")
    parser.add_argument("--board-size", type=int, default=13, help="Board size")
    
    args = parser.parse_args()
    
    print("=== Step 5 – Inspect Parts (Human Games) - FIXED V2 ===")
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load NMF data
    activations, components = load_nmf_data(args.nmf_dir)
    n_parts = activations.shape[1]
    print(f"Found {n_parts} parts")
    
    # Load board data
    board_data = load_board_data(args.npz_dir)
    
    # Analyze positions for each part
    analyses = []
    
    for part_idx in range(n_parts):
        print(f"\nAnalyzing Part {part_idx}...")
        
        # Get strongest positions for this part
        part_activations = activations[:, part_idx]
        top_indices = np.argsort(part_activations)[-args.positions_per_part:][::-1]
        
        for rank, position_idx in enumerate(top_indices, 1):
            print(f"  Position {position_idx} (rank {rank})")
            
            analysis = analyze_position(
                position_idx=position_idx,
                part_idx=part_idx,
                activations=activations,
                board_data=board_data,
                board_size=args.board_size,
                components=components
            )
            
            analysis['rank'] = rank
            analyses.append(analysis)
    
    # Generate summary
    generate_csv_summary(analyses, args.output_dir)
    
    print(f"\nAnalysis complete! Analyzed {len(analyses)} positions across {n_parts} parts.")

if __name__ == "__main__":
    main() 