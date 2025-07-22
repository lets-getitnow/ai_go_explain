# Step 4: NMF Parts Finder

**Goal**: Factor the pooled activations from step 3 into interpretable "parts" using Non-negative Matrix Factorization (NMF).

## Overview

This step takes the channel-averaged activations from step 3 and uses NMF to decompose them into a smaller number of meaningful components (parts). Each part represents a pattern in the neural network's internal representation that might correspond to Go concepts like ladders, atari, eyes, etc.

## Requirements

```bash
pip3 install scikit-learn
```

## Current Status

⚠️ **Limited Dataset**: We currently have only 4 positions from step 1, which significantly limits the analysis:

- **Recommended**: 50-70 parts from thousands of positions
- **Current**: 3 parts from 4 positions (NMF can't learn more components than samples)

For a full analysis, step 1 should collect thousands of varied board positions.

## Files

### Core Scripts

- **`run_nmf.py`**: Main factorization script
  - Loads pooled activations from step 3
  - Runs NMF with conservative component count
  - Saves components and activation patterns
  - Outputs reconstruction error metrics

- **`inspect_parts.py`**: Analysis and inspection script  
  - Shows which positions activate each part most strongly
  - Links parts back to original board positions
  - Generates human-readable summaries

### Output Files

- **`nmf_components.npy`**: The learned parts (shape: n_parts × 512 channels)
- **`nmf_activations.npy`**: Part activation per position (shape: 4 positions × n_parts)
- **`nmf_meta.json`**: Metadata about the factorization
- **`parts_summary.md`**: Human-readable analysis report

## Usage

```bash
cd 4_nmf_parts

# Run NMF factorization
python run_nmf.py

# Inspect the learned parts
python inspect_parts.py
```

## What to Look For

When inspecting parts, look for:

1. **Sparse activation patterns**: Good parts should activate strongly on few positions
2. **Channel clustering**: Parts should use distinct sets of channels
3. **Interpretable position patterns**: The positions that activate a part most strongly should share Go-related features

## Limitations with Small Dataset

With only 4 positions:
- Parts may not be meaningful/interpretable
- Statistical patterns are unreliable  
- Can't validate generalization to new positions

## Next Steps

1. **If patterns emerge**: Proceed to step 5 (add heuristics)
2. **If unclear**: Return to step 1 and collect more positions (aim for 1000+)
3. **Future improvement**: Replace NMF with sparse autoencoder for better feature learning

## Technical Details

- **Algorithm**: sklearn NMF with L1 regularization for sparsity
- **Non-negativity**: Enforced (activations were shifted positive in step 3)  
- **Reconstruction error**: Lower is better, reported in output
- **Random seed**: Fixed (42) for reproducible results 