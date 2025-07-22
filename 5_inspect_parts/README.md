# Step 5: Inspect Parts

**Goal**: For each NMF part, examine the board positions where it activates most strongly to identify interpretable Go patterns.

## Overview

This step takes the NMF components from step 4 and loads the actual board positions where each part shows strongest activation. As a Go expert, you'll examine these positions to look for common tactical or strategic patterns that the neural network has learned to recognize.

## Files

### Core Scripts

- **`examine_boards.py`**: Main inspection script
  - Loads NMF activations from step 4
  - Maps global position indices to specific .npz files and positions within files
  - Extracts and saves the raw board data for manual analysis
  - Shows correlation between part activation strength and board positions

### Output Files

- **`part{N}_rank{R}_pos{GLOBAL}.npy`**: Raw board data for analysis
  - Shape: (22, 7) - KataGo's packed 19x19 board format
  - Channels 0-1: Black and white stone positions
  - Channels 2+: Game state information (recent moves, ko, etc.)

## Usage

```bash
cd 5_inspect_parts

# Extract board positions for analysis
python3 examine_boards.py
```

## Analysis Process

1. **Load strongest positions**: For each of the 3 NMF parts, examine the top 3 most strongly activating board positions

2. **Extract board data**: Save the raw board states as .npy files for detailed analysis

3. **Pattern identification**: As a Go expert, examine the board positions to identify:
   - Tactical patterns (atari, ladders, nets)
   - Life and death situations
   - Connection/cutting patterns
   - Territorial patterns
   - Any other recurring Go concepts

4. **Part interpretation**: Determine if each part corresponds to a meaningful Go concept

## Expected Results

- **Interpretable parts**: Good parts should activate on positions sharing specific Go features
- **Part specialization**: Different parts should focus on different types of patterns
- **Sparsity issues**: Dense parts (using most channels) may be harder to interpret

## Current Limitations

From step 4 analysis:
- Low sparsity (parts use 95-98% of channels)
- Only 3 parts from 2725 positions
- Dense activation patterns may indicate need for parameter tuning

## Next Steps

If clear patterns emerge:
- Proceed to step 6 (add heuristics)
- Document the identified Go concepts for each interpretable part

If patterns are unclear:
- Consider returning to step 4 with different NMF parameters
- Increase sparsity regularization
- Try different numbers of components 