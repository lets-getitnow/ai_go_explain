# Step 5: Inspect Parts

**Goal**: For each NMF part, examine the board positions where it activates most strongly to identify interpretable Go patterns.

## Overview

This step takes the NMF components from step 4 and loads the actual board positions where each part shows strongest activation. As a Go expert, you'll examine these positions to look for common tactical or strategic patterns that the neural network has learned to recognize.

## Files

### Core Scripts

- **`inspect_parts.py`**
  - Single entry-point replacing the previous two scripts.
  - Selects top‐k strongest activations for every NMF part.
  - Saves board tensors (`part{N}_rank{R}_pos{GLOBAL}.npy`).
  - Decodes the move actually played and its board coordinate.
  - Clips the `.sgfs` bundle to a standalone SGF for *each* position (`sgf_pos{GLOBAL}.sgf`).
  - Outputs `strong_positions_summary.csv` linking part, rank, coord, turn, SGF file, and board‐npy file.

### Output Files

- **`part{N}_rank{R}_pos{GLOBAL}.npy`**: Raw board data for analysis
  - Shape: (22, 7) - KataGo's packed 19x19 board format
  - Channels 0-1: Black and white stone positions
  - Channels 2+: Game state information (recent moves, ko, etc.)

## Usage

```bash
cd 5_inspect_parts

# Everything now happens in one go
python3 inspect_parts.py
```

## Analysis Process

1. **Understand data structure**: Run `correlate_sgf.py` to understand how .npz positions map to SGF games and move numbers

2. **Load strongest positions**: For each of the 3 NMF parts, examine the top 3 most strongly activating board positions

3. **Extract board data**: Save the raw board states as .npy files for detailed analysis

4. **Pattern identification**: As a Go expert, examine the board positions to identify:
   - Tactical patterns (atari, ladders, nets)
   - Life and death situations
   - Connection/cutting patterns
   - Territorial patterns
   - Any other recurring Go concepts

5. **Part interpretation**: Determine if each part corresponds to a meaningful Go concept

**Note**: The correlation analysis revealed that `.npz` files contain actual move information in the `policyTargetsNCMove` array. This transforms the analysis from examining static board positions to understanding **what specific moves** the neural network associates with each pattern. This is essential for meaningful Go pattern interpretation.

## Expected Results

- **Interpretable parts**: Good parts should activate on positions sharing specific Go features
- **Part specialization**: Different parts should focus on different types of patterns
- **Sparsity issues**: Dense parts (using most channels) may be harder to interpret

## Current Limitations

From step 4 analysis:
- Low sparsity (parts use 95-98% of channels)
- Only 3 parts from 2725 positions
- Dense activation patterns may indicate need for parameter tuning

**Data Correlation Challenges**:
- Board positions extracted from .npz files lack game context
- Need mapping from position indices to SGF games and move numbers
- Multiple .npz files (4) vs single .sgfs file requires correlation analysis
- Without move context, pattern interpretation is severely limited

## Data Structure Discovery

Running `correlate_sgf.py` revealed the internal structure of KataGo training data:

### NPZ File Structure
Each `.npz` file contains training data with these key arrays:
- **`binaryInputNCHWPacked`** (1000, 22, 7): Board states in packed format
- **`globalInputNC`** (1000, 19): Game metadata (komi, turn, game state)
- **`policyTargetsNCMove`** (1000, 2, 50): **MOVE INFORMATION** - actual moves played
- **`qValueTargetsNCMove`** (1000, 3, 50): Q-values for all possible moves

### Move Decoding
For 7×7 Go:
- Move indices 0-48: Board positions (row×7 + col)
- Move index 49: Pass move
- High values in `policyTargetsNCMove` indicate the actual move played at that position

### Example Strong-Activation Positions
From NMF analysis, decoded move information:

**Part 0, Rank 1** (Global pos 1388):
- Move indices with high values: 22→515, 31→70, 37→333, 38→495
- Primary move: Index 38 (value 495) = board position (5,3)

**Part 1, Rank 1** (Global pos 1256): 
- Move indices with high values: 19→449 (primary move)
- Primary move: Index 19 = board position (2,5)

**Part 2, Rank 1** (Global pos 834):
- Move indices with high values: 48→415
- Primary move: Index 48 = board position (6,6) or pass

### SGF Correlation
- 200 games in single `.sgfs` file
- 4 NPZ files with 1000 positions each = 4000 total training positions
- Analysis uses 2725 positions (some filtering applied in earlier steps)

## Next Steps

**With Move Information Available:**
1. **Create move decoder** to convert indices to 7×7 board coordinates
2. **Analyze move patterns** for each NMF part's strong-activation positions
3. **Extract SGF context** to understand the tactical situation for each move
4. **Pattern identification** with move context:
   - What type of moves activate each part?
   - Are they corner moves, center fights, connection moves?
   - Do parts specialize by move type or board area?

**If clear patterns emerge:**
- Proceed to step 6 (add heuristics)
- Document the identified Go concepts for each interpretable part

**If patterns are unclear:**
- Consider returning to step 4 with different NMF parameters
- Increase sparsity regularization
- Try different numbers of components

**Important:** Now that we have move information, the analysis can focus on **what moves** the neural network considers important, not just static board positions. 