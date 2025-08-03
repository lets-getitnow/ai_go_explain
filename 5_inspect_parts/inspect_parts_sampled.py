#!/usr/bin/env python3
"""
Step 5 – Inspect Parts (Sampled Positions Version)

Modified version that analyzes sampled positions from all parts.
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sgf
import pandas as pd

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
        print(f"Processing NPZ file: {npz_file.name}")
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
                   game_data: Dict[str, Any], board_size: int = 13, 
                   policy_outputs: np.ndarray = None, value_outputs: np.ndarray = None,
                   components: np.ndarray = None) -> Dict[str, Any]:
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
    # position_idx is the global position index, not the turn number in the game
    # Calculate the actual turn number within the game
    if len(moves) > 0:
        # Use modulo to get the turn number within the game
        turn_number = (position_idx % len(moves)) + 1  # +1 because turn numbers start at 1
        if turn_number > len(moves):
            turn_number = len(moves)  # Cap at the last move
        
        if turn_number > 0 and turn_number <= len(moves):
            color, coord = moves[turn_number - 1]  # turn_number 1 corresponds to move 0
            move_coord = coord.upper()
        else:
            move_coord = "Unknown"
    else:
        move_coord = "Unknown"
        turn_number = 1  # Default to turn 1 if no moves
    
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
    
    # Analyze policy outputs if available
    policy_analysis = {}
    if policy_outputs is not None:
        policy_logits = policy_outputs[position_idx]  # Shape: (6, 170)
        
        # Use the first policy head (most common)
        main_policy = policy_logits[0]  # Shape: (170,)
        
        # Convert logits to probabilities
        policy_probs = np.exp(main_policy - np.max(main_policy))  # Softmax
        policy_probs = policy_probs / np.sum(policy_probs)
        
        # Calculate policy entropy
        policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-10))
        
        # Get top moves
        top_indices = np.argsort(policy_probs)[-5:][::-1]  # Top 5 moves
        top_moves = []
        for idx in top_indices:
            if idx == 169:  # Pass move for 13x13 board (169 board positions + 1 pass)
                move = "PASS"
            else:
                # Convert to board coordinates for 13x13 board
                row = idx // 13
                col = idx % 13
                # Convert to Go coordinates (A1, B2, etc.) for 13x13
                move = f"{chr(ord('A') + col)}{row + 1}"
            top_moves.append({
                'move': move,
                'probability': float(policy_probs[idx]),
                'logit': float(main_policy[idx])
            })
        
        policy_analysis = {
            'entropy': float(policy_entropy),
            'confidence': float(np.max(policy_probs)),
            'top_moves': top_moves
        }
    
    # Analyze value outputs if available
    value_analysis = {}
    if value_outputs is not None:
        value_logits = value_outputs[position_idx]  # Shape: (6, 170)
        
        # Use the first value head
        main_value = value_logits[0]  # Shape: (170,)
        
        # Convert logits to probabilities
        value_probs = np.exp(main_value - np.max(main_value))  # Softmax
        value_probs = value_probs / np.sum(value_probs)
        
        value_analysis = {
            'max_value': float(np.max(value_probs)),
            'value_entropy': float(-np.sum(value_probs * np.log(value_probs + 1e-10)))
        }
    
    return {
        'position_idx': position_idx,
        'part_idx': part_idx,
        'activation_strength': activation_strength,
        'activation_percentile': activation_percentile,
        'sgf_content': sgf_content,
        'move_coord': move_coord,
        'turn_number': turn_number,
        'game_id': game_id,
        'total_moves': len(moves),
        'policy_analysis': policy_analysis,
        'value_analysis': value_analysis,
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
    print(f"✅ CSV summary saved to {csv_path}")

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Inspect NMF parts for sampled positions")
    parser.add_argument("--nmf-dir", required=True, type=Path, help="Directory containing NMF results")
    parser.add_argument("--npz-dir", required=True, type=Path, help="Directory containing NPZ files")
    parser.add_argument("--sampled-positions", required=True, type=Path, help="CSV file with sampled positions")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for analysis")
    parser.add_argument("--board-size", type=int, default=13, help="Board size")
    
    args = parser.parse_args()
    
    print("=== Step 5 – Inspect Parts (Sampled Positions) ===")
    
    # Load NMF data
    print("Loading NMF data from", args.nmf_dir)
    components, activations, meta = load_nmf_data(args.nmf_dir)
    
    print(f"Components shape: {components.shape}")
    print(f"Activations shape: {activations.shape}")
    print(f"Meta keys: {list(meta.keys())}")
    
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
    else:
        print("Policy and value outputs not found, analysis will be limited")
    
    # Load sampled positions
    print(f"Loading sampled positions from {args.sampled_positions}")
    sampled_df = pd.read_csv(args.sampled_positions)
    print(f"Loaded {len(sampled_df)} sampled positions")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze sampled positions
    analyses = []
    for _, row in sampled_df.iterrows():
        position_idx = int(row['position_idx'])
        part_idx = int(row['part_idx'])
        print(f"Analyzing position {position_idx} (Part {part_idx})...")
        
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