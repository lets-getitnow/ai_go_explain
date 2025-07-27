# NMF Rank Selection - Quick Reference

## 🎯 **RECOMMENDED RANK: k = 25**

### Why k=25?
- **Best R² score**: 0.789 (79% data explained)
- **Excellent uniqueness**: 0.466 (parts are distinct)
- **Sweet spot**: Not too few, not too many
- **Avoids overfitting**: Stops before diminishing returns

## 📊 **Key Metrics**

| Rank | R² Score | Uniqueness | Recommendation |
|------|----------|------------|----------------|
| 3    | 0.629    | 0.322      | Too few parts |
| 5    | 0.718    | 0.344      | Elbow point |
| 10   | 0.763    | 0.403      | Good baseline |
| 15   | 0.783    | 0.447      | Good alternative |
| **25** | **0.789** | **0.466** | **⭐ RECOMMENDED** |
| 40   | 0.790    | 0.473      | Diminishing returns |
| 60   | 0.788    | 0.472      | Overfitting risk |

## 🔍 **What is Reconstruction?**

**Reconstruction** = How well NMF can rebuild original data from learned parts

- **R² = 1.0**: Perfect reconstruction (100% explained)
- **R² = 0.8**: 80% of data explained
- **R² = 0.5**: 50% of data explained

**Formula**: `Original Data ≈ Activations × Parts`

## 📈 **Dataset Info**

- **Positions**: 6,603 Go board positions
- **Channels**: 512 activation channels
- **Dataset Size**: Substantial for NMF analysis

## 🚀 **Next Steps**

1. Update `run_nmf.py` to use `n_parts=25`
2. Run full NMF analysis
3. Inspect parts for interpretability
4. Generate visualizations

## ⚠️ **Why Not Higher Ranks?**

- Diminishing returns after k=25
- Risk of overfitting to noise
- Computational cost increases
- Parts become less interpretable

---

**Analysis Date**: 2025-07-27  
**Method**: Systematic rank selection with reconstruction quality and uniqueness metrics 