# 3×3 Grid Pooling Implementation

## Overview

This implementation replaces global average pooling with 3×3 grid pooling to preserve spatial information while maintaining computational efficiency for NMF analysis.

## Problem Solved

### Issues with Global Average Pooling:
- **Loses locality**: An eye at C-3 looks identical to one at G-5
- **Parts collapse into board-density themes**: Components become generic "stone count" detectors
- **No spatial meaning**: Can't distinguish corner ladders from center eyes

### How 3×3 Grid Pooling Fixes It:
- **Nine sub-tiles retain coarse coordinates**: "Upper-left ladder" differs from "lower-right ladder"
- **Components can specialize**: Eye-shapes often fire in center tiles, ladder paths near edges
- **Spatial meaning for pennies**: Only ×9 more dimensions vs ×49 blowup of full spatial

## Implementation Details

### Pooling Function
```python
def pool_3x3(x):            # x: (N, C, 7, 7)
    N, C, H, W = x.shape    # H=W=7
    # cell sizes: 3-2-2 split (top, mid, bottom)
    bins = [(0,3), (3,5), (5,7)]
    out = []
    for r0,r1 in bins:
        for c0,c1 in bins:
            out.append(x[:,:,r0:r1,c0:c1].mean(axis=(2,3)))   # (N,C)
    return np.concatenate(out, axis=1)   # (N, C*9)
```

### Data Shape Transformation
- **Before**: `(N_positions, C_channels)` - global average per channel
- **After**: `(N_positions, C_channels * 9)` - 9 spatial regions per channel

### Spatial Regions
```
┌─────────┬─────────┬─────────┐
│  0,0    │  0,1    │  0,2    │  ← Top row (0-3)
├─────────┼─────────┼─────────┤
│  1,0    │  1,1    │  1,2    │  ← Middle row (3-5)
├─────────┼─────────┼─────────┤
│  2,0    │  2,1    │  2,2    │  ← Bottom row (5-7)
└─────────┴─────────┴─────────┘
```

## Usage

### Step 1: Extract 3×3 Pooled Activations
```bash
cd 3_extract_activations
python run_3x3_extraction.py
```

### Step 2: Run NMF Analysis
```bash
cd 4_nmf_parts
python run_3x3_nmf.py
```

### Step 3: Validate Results
```bash
python validate_3x3_pooling.py
```

## Expected Improvements

### Spatial Specificity
- **Corner components**: Fire primarily in corner regions (0,0), (0,2), (2,0), (2,2)
- **Edge components**: Fire along edges (0,1), (1,0), (1,2), (2,1)
- **Center components**: Fire in center region (1,1)

### Go Pattern Recognition
- **Ladder patterns**: Often appear in corner/edge regions
- **Eye shapes**: Typically in center regions
- **Atari patterns**: Can be edge-specific or center-specific

### Reduced Noise
- **Less board-density themes**: Components specialize in spatial regions
- **Better interpretability**: Clear spatial activation patterns
- **More meaningful parts**: Each component has spatial context

## Validation Metrics

### Target Performance
- **Reconstruction R² drop**: ≤ 10% vs global average
- **Component uniqueness**: > 0.30 cosine distance
- **Sparsity**: 70-90% zeros in H matrix
- **Visual check**: ≥ 15 parts clearly local

### Diagnostics
```python
# Heat-map quick-look for spatial specificity
grid = H_comp.reshape(9, -1).mean(axis=1).reshape(3,3)
plt.imshow(grid, cmap='hot', interpolation='nearest')
```

## Troubleshooting

| Symptom | Suspected Cause | Fix |
|---------|----------------|-----|
| Parts still global "stone count" | Rotations not canonical | Ensure Black-to-play & rotate to put most recent move top-left |
| Components die (all-zero H row) | α_H too high in higher-dim setting | Halve α_H or raise rank by 5 |
| Reconstruction now terrible | k too small for 9× more dims | Add +10 components or lower α_H |

## Files Modified

### Core Implementation
- `3_extract_activations/extract_pooled_activations.py`: Updated pooling function
- `4_nmf_parts/run_nmf.py`: Updated metadata handling

### New Scripts
- `3_extract_activations/run_3x3_extraction.py`: Run 3×3 extraction
- `4_nmf_parts/run_3x3_nmf.py`: Run NMF with 3×3 data
- `4_nmf_parts/validate_3x3_pooling.py`: Validate results

### Output Files
- `spatial_patterns_3x3.png`: Visualization of spatial patterns
- `3x3_validation_report.json`: Comprehensive validation report

## Recommended Workflow

1. **Start with k=15**: Often a sweet spot for 3×3 on 3–5k boards
2. **Use existing α_H**: Keep your newly tuned α_H from global pooling
3. **Evaluate results**: Check spatial specificity and interpretability
4. **Scale up**: If parts improve, re-run on larger dataset

## Bottom Line

3×3 grid pooling is a one-hour code tweak that usually flips "noisy blobs" into "corner/edge/center-specific Go motifs." Couple it with the sparsity sweep you already implemented, and you'll know—today—whether spatial information was the missing ingredient.

## References

- Original analysis: User's detailed 3×3 pooling proposal
- NMF documentation: scikit-learn NMF implementation
- Go AI analysis: This project's methodology for analyzing neural network activations 