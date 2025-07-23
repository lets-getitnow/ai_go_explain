#!/usr/bin/env python3
"""
Step 5 – Inspect Parts (combined)

Single-entry script that:
1. Finds top-k strongest activations for each NMF component.
2. Saves the corresponding packed board tensors as *.npy.
3. Decodes the move played at each position.
4. Clips the giant self-play .sgfs bundle so each position gets its own standalone
   SGF file containing the header + exact moves up to (and including) that turn.
5. Emits a consolidated CSV (`strong_positions_summary.csv`) linking everything.

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

    # --- Stage 4: consolidated CSV ------------------------------------------
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
                ]
            )
    print(f"Summary written → {csv_path}\nAll tasks complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1) 