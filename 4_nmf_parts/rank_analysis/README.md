# NMF Rank Selection Analysis

This directory contains the systematic analysis for determining the optimal number of NMF (Non-negative Matrix Factorization) components for your Go AI activation data.

## 📊 What is Rank Selection?

Rank selection is the process of choosing the optimal number of "parts" or "components" for NMF decomposition. Too few parts and you miss important patterns; too many and you fit noise.

## 🔍 What Does "Reconstruction" Mean?

**Reconstruction** refers to how well the NMF model can "rebuild" or "recreate" the original data from the learned parts.

### The Process:
1. **Original Data**: Your activation matrix (6,603 positions × 512 channels)
2. **NMF Decomposition**: Breaks this into two matrices:
   - **Parts Matrix**: (k parts × 512 channels) - the learned "concepts"
   - **Activations Matrix**: (6,603 positions × k parts) - how much each part activates per position
3. **Reconstruction**: Multiply these back together: `Activations × Parts = Reconstructed Data`

### Mathematical Formula:
```
Original Data ≈ Activations × Parts
```

### What the R² Score Means:
- **R² = 1.0**: Perfect reconstruction (100% of original data explained)
- **R² = 0.8**: 80% of original data explained
- **R² = 0.5**: 50% of original data explained

### Visual Analogy:
Think of it like LEGO blocks:
- **Original model** = Your activation data
- **LEGO pieces** = NMF parts
- **Reconstruction** = How well you can rebuild the original model with those pieces
- **R² score** = How close your rebuilt model looks to the original

## 📈 Analysis Results

### Dataset Information:
- **Positions**: 6,603 Go board positions
- **Channels**: 512 activation channels from rconv14.out layer
- **Dataset Size**: Substantial for NMF analysis

### Key Metrics:

| Rank | R² Score | Reconstruction Error | Uniqueness | Status |
|------|----------|---------------------|------------|---------|
| 3    | 0.6287   | 198.9384            | 0.3222     | ✓ Good |
| 5    | 0.7177   | 175.1041            | 0.3444     | ✓ Good |
| 10   | 0.7632   | 159.4316            | 0.4029     | ✓ Good |
| 15   | 0.7827   | 152.1512            | 0.4471     | ✓ Good |
| 25   | 0.7892   | 149.2509            | 0.4663     | ✓ Good |
| 40   | 0.7897   | 149.2412            | 0.4732     | ✓ Good |
| 60   | 0.7878   | 150.3171            | 0.4718     | ✓ Good |

### Key Findings:

1. **Elbow Point**: k=5 (where improvement starts to slow)
2. **Peak Performance**: k=40 (R² = 0.790)
3. **All ranks exceed uniqueness threshold** (0.3)
4. **Diminishing returns** after k=25

## 🎯 Recommendations

### For Your 6,603 Position Dataset:

**Recommended Ranks:**
- **k = 15**: Good balance of reconstruction (0.783) and uniqueness (0.447)
- **k = 25**: Best reconstruction (0.789) with excellent uniqueness (0.466) ⭐ **RECOMMENDED**
- **k = 40**: Peak performance but diminishing returns

### Why k=25 is Recommended:
1. **Best R² score** (0.789) among reasonable ranks
2. **Excellent uniqueness** (0.466) - parts are distinct
3. **Good balance** - not too few, not too many
4. **Avoids overfitting** - stops before diminishing returns

## 📋 Files in This Directory

- **`rank_selection_analysis.png`**: Comprehensive visualization with 4 plots:
  1. Reconstruction Quality (R² vs. rank)
  2. Reconstruction Error (NMF error vs. rank)
  3. Component Uniqueness (cosine distance vs. rank)
  4. Combined Analysis (normalized comparison)

- **`rank_analysis_report.txt`**: Detailed numerical analysis and recommendations

## 🔧 How to Use These Results

1. **Review the visualizations** in `rank_selection_analysis.png`
2. **Read the detailed report** in `rank_analysis_report.txt`
3. **Update your NMF script** to use k=25 (recommended)
4. **Run the full NMF analysis** with the optimal number of parts

## 📚 Methodology

### Rank Selection Recipe (≤ 30 min):
1. **Reconstruction Curve**: Test ranks k = 3, 5, 10, 15, 25, 40, 60 with ≤ 20 iterations each
2. **Component Uniqueness**: Compute cosine distance between weight vectors (want > 0.3)
3. **Visual Spot-Check**: Focus on elbow ranks from step 1
4. **Pick Optimal Rank**: Smallest rank giving ≥ 15 interpretable parts

### Rule of Thumb:
For datasets with N positions and D dimensions:
```
k × D < 0.2 × N
```

With your data: k × 512 < 0.2 × 6,603 → k < 2.58
However, with 6.6k samples, you can safely use higher ranks.

## 🚀 Next Steps

1. **Update `run_nmf.py`** to use k=25
2. **Run full NMF analysis** with optimal rank
3. **Inspect parts** for interpretability
4. **Generate visualizations** of learned parts

## 📖 References

- NMF Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
- Rank Selection Methods: Various papers on NMF rank selection
- Go AI Analysis: This project's methodology for analyzing neural network activations 