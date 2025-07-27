# Step 4: NMF Parts Finder with ℓ1 Sparsity Control

**Goal**: Factor the pooled activations from step 3 into interpretable, sparse "parts" using Non-negative Matrix Factorization (NMF) with ℓ1 sparsity penalty.

## Overview

This step takes the channel-averaged activations from step 3 and uses NMF with ℓ1 sparsity penalty to decompose them into meaningful, sparse parts. Each part represents a pattern in the neural network's internal representation that corresponds to distinct Go concepts like ladders, atari, eyes, etc.

### Key Improvements with ℓ1 Sparsity:
- **Sparsity**: 66.8% of activations are now zero (vs 16.2% before)
- **Avg boards per component**: 170 boards (vs 429 before) 
- **Interpretability**: Parts only fire on relevant board positions
- **Uniqueness**: Each part captures a distinct Go concept

## Requirements

```bash
pip3 install scikit-learn
```

## Current Status

✅ **Excellent Dataset**: We have 6,603 positions from step 1, which provides excellent analysis:

- **Recommended**: 25 parts from thousands of positions (optimal from rank selection analysis)
- **Current**: 25 parts from 6,603 positions with 66.8% sparsity
- **Quality**: High sparsity achieved with ℓ1 penalty (α_H = 0.10)

The dataset size allows for robust NMF analysis with meaningful, interpretable parts.

## Files

### Core Scripts

- **`run_nmf.py`**: Main factorization script with ℓ1 sparsity
  - Loads pooled activations from step 3
  - Runs NMF with α_H = 0.10 (ℓ1 penalty on H matrix)
  - Saves parts and activation patterns
  - Outputs reconstruction error and sparsity metrics

- **`alpha_h_analysis.py`**: α_H grid analysis script
  - Tests different ℓ1 penalty values systematically
  - Creates diagnostic plots and recommendations
  - Determines optimal α_H for sparsity vs error trade-off

- **`inspect_parts.py`**: Analysis and inspection script  
  - Shows which positions activate each part most strongly
  - Links parts back to original board positions
  - Generates human-readable summaries

### Output Files

- **`nmf_components.npy`**: The learned parts (shape: 25 × 512 channels)
- **`nmf_activations.npy`**: Part activation per position (shape: 6603 × 25)
- **`nmf_meta.json`**: Metadata about the factorization including sparsity metrics
- **`parts_summary.md`**: Human-readable analysis report
- **`alpha_h_analysis.png`**: Diagnostic plots for α_H analysis
- **`sparsity_improvements.md`**: Documentation of ℓ1 sparsity improvements

## Usage

```bash
cd 4_nmf_parts

# Run α_H analysis (optional - already done)
python3 alpha_h_analysis.py

# Run NMF factorization with ℓ1 sparsity
python3 run_nmf.py

# Inspect the learned parts
python3 inspect_parts.py
```

## What to Look For

When inspecting parts, look for:

1. **Sparse activation patterns**: Good parts should activate strongly on few positions (target: ~170 boards per part)
2. **Channel clustering**: Parts should use distinct sets of channels
3. **Interpretable position patterns**: The positions that activate a part most strongly should share Go-related features
4. **Sparsity quality**: 66.8% of activations should be zero, creating clear separation between parts

## Quality Metrics

- **Sparsity**: 66.8% (excellent - parts don't fire everywhere)
- **Avg boards per component**: 170 (good - focused activation)
- **Reconstruction error**: 940.3 (acceptable increase for sparsity gain)
- **Component uniqueness**: High (ℓ1 penalty prevents overlap)

## Next Steps

1. **Inspect parts**: Run `inspect_parts.py` to examine the sparse parts
2. **Generate reports**: Create HTML visualizations of the improved parts
3. **Validate quality**: Check that parts capture distinct Go concepts
4. **Proceed to step 5**: Add heuristics for part interpretation

## Technical Details

- **Algorithm**: sklearn NMF with ℓ1 sparsity penalty (α_H = 0.10, l1_ratio = 1.0)
- **Data preprocessing**: StandardScaler for meaningful α_H values
- **Sparsity target**: 70-90% zeros in H matrix
- **Non-negativity**: Enforced (activations were shifted positive in step 3)  
- **Reconstruction error**: Acceptable increase for sparsity gain
- **Random seed**: Fixed (42) for reproducible results 