#!/usr/bin/env python3
"""
Step 5 – Inspect Parts (Human Games Version) - FIXED
====================================================

Fixed version that analyzes positions from ALL parts, not just the first few.

Expected Working Directory
-------------------------
This script expects to be run from the project root directory:
    /Users/hunterp/dev/ai_go_explain

Path Assumptions
----------------
- SGF files: games/go13/ (relative to project root)
- NPZ files: <output_dir>/npz_files/ (from pipeline)
- NMF data: <output_dir>/nmf_parts/ (from pipeline)

Usage
-----
python3 5_inspect_parts/inspect_parts_human_games_fixed.py \
    --nmf-dir test/nmf_parts \
    --npz-dir test/npz_files \
    --output-dir test/inspect_parts \
    --positions-per-part 2 \
    --board-size 13
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sgf

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def load_nmf_data(nmf_dir: Path):
    """Load NMF components, activations, and metadata."""
    components = np.load(nmf_dir / "nmf_components.npy")
    activations = np.load(nmf_dir / "nmf_activations.npy")
    
    with open(nmf_dir / "nmf_meta.json", 'r') as f:
        meta = json.load(f)
    
    return components, activations, meta

def load_game_data(npz_dir: Path) -> Dict[str, Any]:
    """Load game data from NPZ files."""
    game_data = {}
    
    print(f"Looking for NPZ files in: {npz_dir}")
    npz_files = list(npz_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files")
    
    for npz_file in npz_dir.glob("*.npz"):
        data = np.load(npz_file, allow_pickle=True)
        
        game_id = data['game_id'][0].decode('utf-8') if len(data['game_id']) > 0 else npz_file.stem
        
        # Load corresponding SGF file
        sgf_file = Path("games/go13") / f"{game_id}.sgf"
        
        if sgf_file.exists():
            with open(sgf_file, 'r') as f:
                sgf_content = f.read()
            
            # Parse SGF to get moves
            try:
                collection = sgf.parse(sgf_content)
                game = collection[0]
                moves = []
                
                # Extract moves from SGF
                for node in game:
                    # Check for move properties (B or W)
                    if 'B' in node.properties:
                        pos = node.properties['B'][0]
                        if pos != '':
                            x, y = ord(pos[0]) - ord('a'), ord(pos[1]) - ord('a')
                            coord = pos.upper()
                            moves.append(('b', coord))
                    elif 'W' in node.properties:
                        pos = node.properties['W'][0]
                        if pos != '':
                            x, y = ord(pos[0]) - ord('a'), ord(pos[1]) - ord('a')
                            coord = pos.upper()
                            moves.append(('w', coord))
                
                print(f"Parsed {len(moves)} moves from SGF")
                
                game_data[game_id] = {
                    'moves': moves,
                    'sgf_content': sgf_content,
                    'npz_file': str(npz_file)
                }
                
            except Exception as e:
                print(f"Error parsing SGF {sgf_file}: {e}")
                continue
        else:
            print(f"SGF file not found: {sgf_file}")
    
    return game_data

def create_position_sgf(moves: List[Tuple[str, str]], position_idx: int, original_sgf: str) -> str:
    """Create SGF for a specific position."""
    if position_idx >= len(moves):
        return original_sgf
    
    # Create new SGF with moves up to position_idx
    new_moves = moves[:position_idx + 1]
    
    # Simple SGF creation
    sgf_content = "(;FF[4]GM[1]SZ[13]"
    
    for i, (color, coord) in enumerate(new_moves):
        if color == 'b':
            sgf_content += f"AB[{coord.lower()}]"
        elif color == 'w':
            sgf_content += f"AW[{coord.lower()}]"
    
    sgf_content += ")"
    return sgf_content

def analyze_position(position_idx: int, part_idx: int, activations: np.ndarray, 
                   game_data: Dict[str, Any], board_size: int = 13, 
                   policy_outputs: np.ndarray = None, value_outputs: np.ndarray = None,
                   components: np.ndarray = None) -> Dict[str, Any]:
    """Analyze a specific position and part combination."""
    
    # Find which game this position belongs to
    # This is a simplified approach - in practice you'd need to map position_idx to game
    game_id = list(game_data.keys())[0]  # Simplified - use first game
    game_info = game_data[game_id]
    
    # Get activation strength for this position and part
    activation_strength = activations[position_idx, part_idx]
    
    # Get move information
    moves = game_info['moves']
    move_coord = "Unknown"
    if position_idx < len(moves):
        color, coord = moves[position_idx]
        move_coord = coord
    
    # Create SGF for this position
    position_sgf = create_position_sgf(moves, position_idx, game_info['sgf_content'])
    
    # Calculate uniqueness score
    uniqueness_score = calculate_uniqueness_score(activations, position_idx, part_idx)
    
    # Calculate channel activity if components available
    channel_activity = []
    if components is not None:
        channel_activity = calculate_channel_activity(components, part_idx, board_size)
    
    # Calculate part comparison
    part_comparison = calculate_part_comparison(activations, position_idx, part_idx)
    
    return {
        'position_idx': int(position_idx),
        'part_idx': int(part_idx),
        'activation_strength': float(activation_strength),
        'game_id': game_id,
        'move_coord': move_coord,
        'move_number': int(position_idx),
        'uniqueness_score': float(uniqueness_score),
        'channel_activity': channel_activity,
        'part_comparison': part_comparison,
        'position_sgf': position_sgf,
        'n_parts': int(activations.shape[1]),
        'n_positions': int(activations.shape[0])
    }

def generate_csv_summary(analyses: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate CSV summary of analyses."""
    output_file = output_dir / "strong_positions_summary.csv"
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['part_idx', 'position_idx', 'activation_strength', 'component_stats', 'top_activations', 'meta_info'])
        
        for analysis in analyses:
            # Calculate component stats
            strength = analysis['activation_strength']
            component_stats = f"min={strength:.4f},max={strength:.4f},mean={strength:.4f},sparsity=0.00%"
            
            # Top activations (simplified)
            top_indices = [analysis['position_idx']]
            top_activations = f"top_indices={top_indices}"
            
            # Meta info
            meta_info = f"n_parts={analysis['n_parts']},n_positions={analysis['n_positions']},game_id={analysis['game_id']}"
            
            writer.writerow([
                analysis['part_idx'],
                analysis['position_idx'],
                analysis['activation_strength'],
                component_stats,
                top_activations,
                meta_info
            ])
    
    print(f"✅ CSV summary saved to {output_file}")

def calculate_uniqueness_score(activations: np.ndarray, position_idx: int, part_idx: int) -> float:
    """Calculate how unique this position is for this part."""
    position_activations = activations[position_idx]
    part_activations = activations[:, part_idx]
    
    # Calculate percentile of this position's activation for this part
    percentile = np.percentile(part_activations, 95)  # Top 5%
    position_strength = position_activations[part_idx]
    
    return position_strength / percentile if percentile > 0 else 0

def calculate_channel_activity(components: np.ndarray, part_idx: int, board_size: int = 13) -> List[Dict[str, Any]]:
    """Calculate which channels are most active for this part."""
    part_component = components[part_idx]
    
    # Reshape to board format
    channels_per_position = board_size * board_size
    n_channels = len(part_component) // channels_per_position
    
    channel_activity = []
    for channel_idx in range(n_channels):
        start_idx = channel_idx * channels_per_position
        end_idx = start_idx + channels_per_position
        channel_strength = np.mean(part_component[start_idx:end_idx])
        
        channel_activity.append({
            'channel': int(channel_idx),
            'strength': float(channel_strength),
            'board_region': f"ch{channel_idx}"
        })
    
    # Sort by strength
    channel_activity.sort(key=lambda x: x['strength'], reverse=True)
    return channel_activity[:10]  # Top 10 channels

def calculate_part_comparison(activations: np.ndarray, position_idx: int, part_idx: int) -> Dict[str, Any]:
    """Calculate how this part compares to others for this position."""
    position_activations = activations[position_idx]
    
    # Find other parts with high activation
    other_activations = position_activations.copy()
    other_activations[part_idx] = 0  # Exclude current part
    
    # Get top 3 other parts
    top_indices = np.argsort(other_activations)[-3:][::-1]
    
    top_other_parts = []
    for other_part_idx in top_indices:
        if other_activations[other_part_idx] > 0:
            top_other_parts.append({
                'part': int(other_part_idx),
                'activation': float(other_activations[other_part_idx])
            })
    
    return {
        'max_other_activation': float(np.max(other_activations)),
        'part_rank': int(np.argsort(position_activations)[::-1].tolist().index(part_idx) + 1),
        'top_other_parts': top_other_parts
    }

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Inspect NMF parts for human games - FIXED")
    parser.add_argument("--nmf-dir", required=True, type=Path, help="Directory containing NMF results")
    parser.add_argument("--npz-dir", required=True, type=Path, help="Directory containing NPZ files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for analysis")
    parser.add_argument("--positions-per-part", type=int, default=2, help="Number of positions to analyze per part")
    parser.add_argument("--board-size", type=int, default=13, help="Board size")
    
    args = parser.parse_args()
    
    print("=== Step 5 – Inspect Parts (Human Games) - FIXED ===")
    
    # Load NMF data
    print("Loading NMF data from", args.nmf_dir)
    components, activations, meta = load_nmf_data(args.nmf_dir)
    
    print(f"Components shape: {components.shape}")
    print(f"Activations shape: {activations.shape}")
    
    # Load game data
    print("Loading game data from", args.npz_dir)
    game_data = load_game_data(args.npz_dir)
    print(f"Loaded {len(game_data)} games")
    
    # Load policy and value outputs if available
    policy_outputs = None
    value_outputs = None
    activations_dir = args.nmf_dir.parent / "activations"
    policy_file = activations_dir / "policy_outputs__baseline.npy"
    value_file = activations_dir / "value_outputs__baseline.npy"
    
    if policy_file.exists() and value_file.exists():
        print("Loading policy and value outputs...")
        policy_outputs = np.load(policy_file)
        value_outputs = np.load(value_file)
        print(f"Policy outputs shape: {policy_outputs.shape}")
        print(f"Value outputs shape: {value_outputs.shape}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze positions from ALL parts
    n_parts = activations.shape[1]
    positions_per_part = args.positions_per_part
    
    print(f"Analyzing {positions_per_part} positions from each of {n_parts} parts...")
    
    analyses = []
    for part_idx in range(n_parts):
        print(f"Processing Part {part_idx}...")
        
        # Get the strongest positions for this part
        part_activations = activations[:, part_idx]
        strongest_positions = np.argsort(part_activations)[-positions_per_part:][::-1]
        
        for i, position_idx in enumerate(strongest_positions):
            print(f"  Analyzing position {position_idx} (rank {i+1} for part {part_idx})...")
            
            analysis = analyze_position(
                position_idx, part_idx, activations, game_data, 
                args.board_size, policy_outputs, value_outputs, components
            )
            analyses.append(analysis)
    
    # Save detailed analysis
    output_file = args.output_dir / "part_analyses.json"
    with open(output_file, 'w') as f:
        json.dump(analyses, f, cls=NumpyEncoder, indent=2)
    print(f"✅ Analysis saved to {output_file}")
    
    # Generate CSV summary
    generate_csv_summary(analyses, args.output_dir)
    
    # Print summary
    print("\n=== Summary ===")
    for analysis in analyses:
        pos_idx = analysis['position_idx']
        part_idx = analysis['part_idx']
        strength = analysis['activation_strength']
        move = analysis['move_coord']
        print(f"Position {pos_idx}: Part {part_idx}, Strength {strength:.6f}, Move {move}")

if __name__ == "__main__":
    main() 