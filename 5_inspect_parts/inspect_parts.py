#!/usr/bin/env python3
"""
Step 5 – Inspect Parts (combined)

Single-entry script that:
1. Finds top-k strongest activations for each NMF component.
2. Saves the corresponding packed board tensors as *.npy.
3. Decodes the move played at each position.
4. Performs 3 types of analysis per position for NMF interpretation.
5. Clips the giant self-play .sgfs bundle so each position gets its own standalone
   SGF file containing the header + exact moves up to (and including) that turn.
6. Emits a consolidated CSV (`strong_positions_summary.csv`) linking everything.

STRICT CONTRACT (ZERO-FALLBACK):
• Abort with RuntimeError on any missing file or inconsistency.
• Never guess or substitute defaults.
"""
from __future__ import annotations

import bisect
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# ---------------------------------------------------------------------------
# JSON encoder for numpy types
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ---------------------------------------------------------------------------
# Constants – adjust here if your dataset path changes
# ---------------------------------------------------------------------------
BASE_DIR = (
    Path(__file__).resolve().parent / "../selfplay_out/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz"
).resolve()
TDATA_DIR = BASE_DIR / "tdata"
SGF_DIR = BASE_DIR / "sgfs"

N_TOP = 3            # strongest activations per part
BOARD_SIZE = 7
PASS_INDEX = BOARD_SIZE * BOARD_SIZE  # 49 on 7×7

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_activations_and_mapping() -> tuple[np.ndarray, List[str]]:
    """Load NMF activation matrix and global-index → .npz mapping file."""
    activ_path = Path(__file__).resolve().parent / "../4_nmf_parts/nmf_activations.npy"
    mapping_path = Path(__file__).resolve().parent / "../3_extract_activations/activations/pos_index_to_npz.txt"

    if not activ_path.exists():
        raise RuntimeError(f"Missing activations file {activ_path}")
    if not mapping_path.exists():
        raise RuntimeError(f"Missing mapping file {mapping_path}")

    activations = np.load(activ_path)
    with mapping_path.open("r") as f:
        pos_to_file = [ln.strip() for ln in f]

    return activations, pos_to_file


def strongest_indices_for_part(part_idx: int, activ: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k strongest activations for the given component."""
    part_act = activ[:, part_idx]
    top = np.argsort(part_act)[-k:][::-1]
    return top


def calc_file_and_local_idx(global_pos: int, mapping: List[str]) -> tuple[str, int]:
    """Given global index, derive .npz filename & position-within-file."""
    target_file = mapping[global_pos]
    local_idx = sum(1 for i in range(global_pos) if mapping[i] == target_file)
    return target_file, local_idx


def decode_move(policy_targets: np.ndarray) -> int:
    """Extract chosen move from channel-1 counts (argmax)."""
    return int(policy_targets[1].argmax())


def idx_to_coord(idx: int) -> str:
    if idx == PASS_INDEX:
        return "PASS"
    row, col = divmod(idx, BOARD_SIZE)
    return f"({row},{col})"


def split_sgfs_bundle(text: str) -> List[str]:
    """Exact parenthesis-balanced split of a KataGo .sgfs bundle."""
    games, buf, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        if depth:
            buf.append(ch)
        if ch == ")":
            depth -= 1
            if depth == 0 and buf:
                games.append("".join(buf))
                buf = []
    return games

# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_nmf_features(pos: Dict[str, Any], activations: np.ndarray, board_data: np.ndarray) -> Dict[str, Any]:
    """Analysis 1: Neural Network Feature Analysis"""
    part_idx = pos["part"]
    gpos = pos["global_pos"]
    
    # Current component activation strength
    activation_strength = float(activations[gpos, part_idx])
    
    # Activations in other components for this position
    other_activations = [float(activations[gpos, i]) for i in range(activations.shape[1])]
    
    # Channel importance - count bits set in each channel
    channel_activity = []
    for ch in range(board_data.shape[0]):
        bit_count = sum(bin(c).count('1') for c in board_data[ch])
        channel_activity.append(bit_count)
    
    # Find most active channels (top 5)
    top_channels = sorted(enumerate(channel_activity), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "activation_strength": activation_strength,
        "rank_in_component": pos["rank"],
        "activation_in_other_components": other_activations,
        "channel_activity": channel_activity,
        "top_active_channels": top_channels,
        "total_board_activity": sum(channel_activity)
    }


def analyze_go_patterns(pos: Dict[str, Any], policy_data: np.ndarray) -> Dict[str, Any]:
    """Analysis 2: Go-Specific Pattern Recognition"""
    move_idx = pos["move_idx"]
    coord = pos["coord"]
    turn_num = pos.get("turn", 0)
    
    # Basic move classification
    if move_idx == PASS_INDEX:
        move_type = "pass"
    else:
        row, col = divmod(move_idx, BOARD_SIZE)
        # Classify by board region
        if row <= 2 or row >= BOARD_SIZE - 3 or col <= 2 or col >= BOARD_SIZE - 3:
            if (row <= 2 and col <= 2) or (row <= 2 and col >= BOARD_SIZE - 3) or \
               (row >= BOARD_SIZE - 3 and col <= 2) or (row >= BOARD_SIZE - 3 and col >= BOARD_SIZE - 3):
                move_type = "corner"
            else:
                move_type = "side"
        else:
            move_type = "center"
    
    # Game phase estimation based on turn number
    if turn_num < 30:  # Adjusted for 7x7 board
        game_phase = "opening"
    elif turn_num < 60:
        game_phase = "middle_game"
    else:
        game_phase = "endgame"
    
    # Policy analysis
    policy_counts = policy_data[1].astype(int)
    total_counts = policy_counts.sum()
    
    # Calculate policy entropy
    probs = policy_counts / total_counts if total_counts > 0 else policy_counts
    probs = probs[probs > 0]  # Remove zeros for log calculation
    entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
    
    # Top policy moves
    top5_idx = policy_counts.argsort()[-5:][::-1]
    top_moves = []
    for idx in top5_idx:
        if policy_counts[idx] == 0:
            continue
        pct = 100.0 * policy_counts[idx] / total_counts if total_counts else 0
        top_moves.append({
            "coord": idx_to_coord(int(idx)), 
            "percentage": round(pct, 1), 
            "count": int(policy_counts[idx])
        })
    
    return {
        "move_type": move_type,
        "game_phase": game_phase,
        "turn_number": turn_num,
        "policy_entropy": float(entropy),
        "policy_confidence": float(policy_counts[move_idx] / total_counts) if total_counts > 0 else 0.0,
        "top_policy_moves": top_moves,
        "total_policy_counts": int(total_counts)
    }


def analyze_component_comparison(pos: Dict[str, Any], all_positions: List[Dict[str, Any]], activations: np.ndarray) -> Dict[str, Any]:
    """Analysis 3: Comparative Component Behavior"""
    part_idx = pos["part"]
    gpos = pos["global_pos"]
    
    # Find positions where other components are most active
    current_activation = activations[gpos, part_idx]
    
    # Component specialization: how unique is this activation?
    other_component_activations = [activations[gpos, i] for i in range(activations.shape[1]) if i != part_idx]
    max_other_activation = max(other_component_activations) if other_component_activations else 0.0
    
    uniqueness_score = current_activation / (current_activation + max_other_activation) if (current_activation + max_other_activation) > 0 else 0.0
    
    # Find similar positions (other high-activating positions for this component)
    component_activations = activations[:, part_idx]
    similar_position_indices = np.argsort(component_activations)[-10:][::-1]  # Top 10
    similar_positions = [int(idx) for idx in similar_position_indices if idx != gpos][:5]  # Exclude self, take top 5
    
    # Ranking across all positions for this component
    position_rank = np.sum(component_activations > current_activation) + 1
    
    return {
        "uniqueness_score": float(uniqueness_score),
        "similar_positions": similar_positions,
        "component_rank": int(position_rank),
        "max_other_component_activation": float(max_other_activation),
        "activation_percentile": float(100 * (1 - position_rank / len(component_activations)))
    }


# ---------------------------------------------------------------------------
# Main procedure
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Step 5 – Inspect Parts (combined) ===")

    # --- Stage 1: select strongest positions --------------------------------
    activ, mapping = load_activations_and_mapping()
    n_parts = activ.shape[1]
    print(f"Found {n_parts} parts; scanning top {N_TOP} positions each…")

    positions: List[Dict[str, Any]] = []  # accumulate metadata for later stages

    for part_idx in range(n_parts):
        top_idxs = strongest_indices_for_part(part_idx, activ, N_TOP)
        print(f"Part {part_idx}: global indices {list(top_idxs)}")

        for rank, gpos in enumerate(top_idxs, start=1):
            npz_file, local_idx = calc_file_and_local_idx(gpos, mapping)
            filename_only = Path(npz_file).name

            # Load board tensor & save .npy
            board_data = np.load(npz_file)["binaryInputNCHWPacked"][local_idx]
            npy_name = f"part{part_idx}_rank{rank}_pos{gpos}.npy"
            np.save(npy_name, board_data)

            positions.append(
                {
                    "part": part_idx,
                    "rank": rank,
                    "global_pos": gpos,
                    "npz_file": filename_only,
                    "pos_in_file": local_idx,
                    "board_npy": npy_name,
                }
            )

    print(f"→ Saved {len(positions)} board tensors\n")

    # --- Stage 2: decode moves ----------------------------------------------
    for pos in positions:
        npz_path = TDATA_DIR / pos["npz_file"]
        if not npz_path.exists():
            raise RuntimeError(f"Expected {npz_path} referenced by mapping but file is missing")

        data = np.load(npz_path)
        policy_slice = data["policyTargetsNCMove"][pos["pos_in_file"]]
        move_idx = decode_move(policy_slice)
        pos["move_idx"] = move_idx
        pos["coord"] = idx_to_coord(move_idx)

    print("Move decoding complete\n")

    # --- Stage 3: SGF clipping ----------------------------------------------
    sgf_candidates = sorted([p for p in SGF_DIR.glob("*.sgfs")])
    if not sgf_candidates:
        raise RuntimeError(f"No .sgfs bundle found in {SGF_DIR}")
    sgf_path = sgf_candidates[0]
    print(f"Using SGF bundle {sgf_path.name}")

    sgf_raw = sgf_path.read_text()
    games = split_sgfs_bundle(sgf_raw.strip())
    print(f"Bundle contains {len(games)} games")

    move_counts = [len(re.findall(r";[BW]\[", g)) for g in games]
    boundaries, cum = [], 0
    for cnt in move_counts:
        boundaries.append(cum)
        cum += cnt
    total_positions = cum
    print(f"Total move positions across all games: {total_positions}\n")

    for pos in positions:
        gpos = pos["global_pos"]
        if gpos >= total_positions:
            raise RuntimeError(f"Global pos {gpos} exceeds total {total_positions}")

        game_idx = bisect.bisect_right(boundaries, gpos) - 1
        turn_num = gpos - boundaries[game_idx]

        game_text = games[game_idx]
        header, *moves = game_text.split(";")
        clipped = ";".join([header] + moves[: turn_num + 1]) + ")"

        sgf_out = f"sgf_pos{gpos}.sgf"
        Path(sgf_out).write_text(clipped)
        pos["sgf_file"] = sgf_out
        pos["turn"] = turn_num

        # Console summary for immediate inspection
        print(
            f"Part {pos['part']} Rank {pos['rank']} | global {gpos} | turn {turn_num} | "
            f"coord {pos['coord']} | SGF {sgf_out} | board {pos['board_npy']}"
        )

    print("SGF clipping done – individual files written\n")

    # --- Stage 4: Comprehensive Analysis ------------------------------------
    print("=== Stage 4: Comprehensive Analysis ===")
    
    for pos in positions:
        gpos = pos["global_pos"]
        npz_path = TDATA_DIR / pos["npz_file"]
        data = np.load(npz_path)
        
        # Load board data and policy data
        board_data = data["binaryInputNCHWPacked"][pos["pos_in_file"]]
        policy_data = data["policyTargetsNCMove"][pos["pos_in_file"]]
        
        # Perform all three analyses
        nmf_analysis = analyze_nmf_features(pos, activ, board_data)
        go_analysis = analyze_go_patterns(pos, policy_data)
        comparison_analysis = analyze_component_comparison(pos, positions, activ)
        
        # Combine analyses
        comprehensive_analysis = {
            "position_info": {
                "part": int(pos["part"]),
                "rank": int(pos["rank"]),
                "global_position": int(gpos),
                "move_coordinate": pos["coord"],
                "turn_number": int(pos["turn"]),
                "npz_file": pos["npz_file"],
                "board_tensor_file": pos["board_npy"],
                "sgf_file": pos["sgf_file"]
            },
            "nmf_analysis": nmf_analysis,
            "go_pattern_analysis": go_analysis,
            "component_comparison": comparison_analysis
        }
        
        # Save analysis to JSON file
        analysis_file = f"analysis_pos{gpos}.json"
        with open(analysis_file, 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2, cls=NumpyEncoder)
        pos["analysis_file"] = analysis_file
        
        print(f"Analysis complete for Part {pos['part']} Rank {pos['rank']} (pos {gpos})")
    
    print("Comprehensive analysis complete\n")

    # --- Stage 5: consolidated CSV ------------------------------------------
    csv_path = "strong_positions_summary.csv"
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(
            [
                "part",
                "rank",
                "global_pos",
                "coord",
                "turn",
                "sgf_file",
                "board_npy",
                "analysis_file",
            ]
        )
        for p in positions:
            writer.writerow(
                [
                    p["part"],
                    p["rank"],
                    p["global_pos"],
                    p["coord"],
                    p["turn"],
                    p["sgf_file"],
                    p["board_npy"],
                    p["analysis_file"]
                ]
            )
    print(f"Summary written → {csv_path}\nAll tasks complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1) 