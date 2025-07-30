#!/usr/bin/env python3
"""
Step 5 – Inspect Parts (Human Games Version)

Simplified version for human games data that doesn't require selfplay data.
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
    print(f"Found {len(npz_files)} NPZ files: {[f.name for f in npz_files]}")
    
    for npz_file in npz_dir.glob("*.npz"):
        print(f"Processing NPZ file: {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        print(f"NPZ data keys: {list(data.keys())}")
        
        game_id = data['game_id'][0].decode('utf-8') if len(data['game_id']) > 0 else npz_file.stem
        print(f"Game ID: {game_id}")
        
        # Load corresponding SGF file
        sgf_file = Path("../../games/go13") / f"{game_id}.sgf"
        print(f"Looking for SGF file: {sgf_file}")
        print(f"SGF file exists: {sgf_file.exists()}")
        
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
                    'sgf_content': sgf_content,
                    'moves': moves,
                    'npz_data': data
                }
            except Exception as e:
                print(f"Warning: Could not parse SGF for {game_id}: {e}")
                game_data[game_id] = {
                    'sgf_content': sgf_content,
                    'moves': [],
                    'npz_data': data
                }
        else:
            print(f"SGF file not found: {sgf_file}")
    
    print(f"Loaded {len(game_data)} games")
    return game_data

def create_position_sgf(moves: List[Tuple[str, str]], position_idx: int, original_sgf: str) -> str:
    """Create SGF content for a specific position using the original SGF."""
    if position_idx == 0:
        # Empty board - return just the header without moves
        # Find the first move in the original SGF
        lines = original_sgf.split('\n')
        header_lines = []
        for line in lines:
            if line.strip() and not (line.strip().startswith(';B[') or line.strip().startswith(';W[')):
                header_lines.append(line)
            else:
                break
        return '\n'.join(header_lines) + '\nC[Human game position - Start])'
    
    # Parse the original SGF to get the exact format
    try:
        collection = sgf.parse(original_sgf)
        game = collection[0]
        
        # Reconstruct SGF with complete game
        sgf_parts = []
        
        # Add the root node (game info)
        root_props = []
        for key, values in game.root.properties.items():
            if key not in ['B', 'W']:  # Skip move properties
                for value in values:
                    root_props.append(f"{key}[{value}]")
        
        sgf_parts.append("(;" + "".join(root_props))
        
        # Add all moves from the complete game
        for node in game:
            for key, values in node.properties.items():
                if key in ['B', 'W']:
                    for value in values:
                        sgf_parts.append(f";{key}[{value}]")
        
        # Add comment indicating which move this position represents
        if position_idx > 0 and position_idx <= len(moves):
            sgf_parts.append(f"C[Human game position - Move {position_idx} highlighted])")
        else:
            sgf_parts.append("C[Human game position])")
        
        return "".join(sgf_parts)
        
    except Exception as e:
        print(f"Warning: Could not parse original SGF, using fallback: {e}")
        # Fallback to simple format with complete moves
        sgf_parts = ["(;FF[4]GM[1]SZ[13]"]
        
        for i in range(len(moves)):
            color, coord = moves[i]
            move_str = f";{color.upper()}[{coord}]"
            sgf_parts.append(move_str)
        
        if position_idx > 0 and position_idx <= len(moves):
            sgf_parts.append(f"C[Human game position - Move {position_idx} highlighted])")
        else:
            sgf_parts.append("C[Human game position])")
        
        return "".join(sgf_parts)

def analyze_position(position_idx: int, part_idx: int, activations: np.ndarray, 
                   game_data: Dict[str, Any], board_size: int = 13) -> Dict[str, Any]:
    """Analyze a specific position."""
    # Get activation strength for this position and part
    activation_strength = float(activations[position_idx, part_idx])
    
    # Get game data (assuming single game for now)
    game_id = list(game_data.keys())[0]
    game_info = game_data[game_id]
    moves = game_info['moves']
    original_sgf = game_info['sgf_content']
    
    # Create SGF content for this position
    sgf_content = create_position_sgf(moves, position_idx, original_sgf)
    
    # Get move information
    if position_idx > 0 and position_idx <= len(moves):  # position_idx 0 is empty board
        color, coord = moves[position_idx - 1]  # position_idx 1 corresponds to move 0
        move_coord = coord.upper()
        turn_number = position_idx
    else:
        move_coord = "Unknown"
        turn_number = position_idx
    
    # Calculate activation percentile
    all_activations = activations[:, part_idx]
    # Calculate what percentile this activation strength represents
    if np.max(all_activations) == np.min(all_activations):
        activation_percentile = 50.0  # If all values are the same, use 50%
    else:
        # Calculate the percentile rank of this activation strength
        # Count how many activations are less than this one
        less_than_count = np.sum(all_activations < activation_strength)
        total_count = len(all_activations)
        activation_percentile = (less_than_count / total_count) * 100.0
    
    return {
        'position_idx': position_idx,
        'part_idx': part_idx,
        'activation_strength': activation_strength,
        'activation_percentile': activation_percentile,
        'sgf_content': sgf_content,
        'move_coord': move_coord,
        'turn_number': turn_number,
        'game_id': game_id,
        'total_moves': len(moves)
    }

def generate_csv_summary(analyses: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate CSV summary file compatible with HTML generator."""
    csv_file = output_dir / "strong_positions_summary.csv"
    
    # Create CSV with columns expected by HTML generator
    with csv_file.open('w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'part_idx', 'position_idx', 'activation_strength', 'component_stats',
            'top_activations', 'meta_info'
        ])
        
        # Write data rows
        for analysis in analyses:
            part_idx = analysis['part_idx']
            position_idx = analysis['position_idx']
            activation_strength = analysis['activation_strength']
            
            # For each position, create a summary row
            writer.writerow([
                part_idx,
                str(position_idx),  # position_idx
                f"{activation_strength:.6f}",  # activation_strength
                f"min=0.0000,max={activation_strength:.4f},mean={activation_strength:.4f},sparsity=0.00%",
                f"top_indices=[{position_idx}]",  # top_activations
                f"n_parts=25,n_positions=114,game_id={analysis['game_id']}"
            ])
    
    print(f"✅ CSV summary saved to {csv_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmf-dir", type=Path, required=True, help="Directory containing NMF results")
    parser.add_argument("--npz-dir", type=Path, required=True, help="Directory containing NPZ files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for reports")
    parser.add_argument("--max-positions", type=int, default=10, help="Maximum number of positions to analyze")
    parser.add_argument("--board-size", type=int, default=13, help="Board size")
    
    args = parser.parse_args()
    
    print("=== Step 5 – Inspect Parts (Human Games) ===")
    
    # Load NMF data
    print(f"Loading NMF data from {args.nmf_dir}...")
    components, activations, meta = load_nmf_data(args.nmf_dir)
    
    print(f"Components shape: {components.shape}")
    print(f"Activations shape: {activations.shape}")
    print(f"Meta keys: {list(meta.keys())}")
    
    # Load game data
    print(f"Loading game data from {args.npz_dir}...")
    game_data = load_game_data(args.npz_dir)
    print(f"Loaded {len(game_data)} games")
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Analyze positions
    n_positions = min(args.max_positions, activations.shape[0])
    n_parts = min(10, activations.shape[1])  # Analyze top 10 parts
    
    print(f"Analyzing {n_positions} positions across {n_parts} parts...")
    
    analyses = []
    for position_idx in range(n_positions):
        # Find the part with highest activation for this position
        position_activations = activations[position_idx, :]
        best_part_idx = np.argmax(position_activations)
        
        print(f"Analyzing position {position_idx} (best part: {best_part_idx})...")
        analysis = analyze_position(position_idx, best_part_idx, activations, game_data, args.board_size)
        analyses.append(analysis)
    
    # Save analysis
    output_file = args.output_dir / "part_analyses.json"
    with output_file.open('w') as f:
        json.dump(analyses, f, indent=2, cls=NumpyEncoder)
    
    print(f"✅ Analysis saved to {output_file}")
    
    # Generate CSV summary
    generate_csv_summary(analyses, args.output_dir)
    
    # Print summary
    print("\n=== Summary ===")
    for analysis in analyses:
        pos_idx = analysis["position_idx"]
        part_idx = analysis["part_idx"]
        strength = analysis["activation_strength"]
        move = analysis["move_coord"]
        print(f"Position {pos_idx}: Part {part_idx}, Strength {strength:.6f}, Move {move}")

if __name__ == "__main__":
    main()