# Step 3: Extract Pooled Activations

Extract neural network activations from KataGo model and pool them spatially for downstream analysis.

## Prerequisites
- KataGo model checkpoint file
- Position data (`.npz` files from Step 1)
- Layer selection from Step 2

## Basic Usage (Single Dataset)
```bash
python3 extract_pooled_activations.py \
  --positions-dir selfplay_out/ \
  --ckpt-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
  --output-dir activations/
```

## Multi-Variant Usage (Contextual Channel Detection)
For detecting channels sensitive to global inputs vs board shape:

```bash
# First generate variants (Step 1)
python3 1_collect_positions/generate_variants.py \
  --input-dir selfplay_out \
  --output-dir variants \
  --mode zero_global

# Then extract activations for all variants
python3 extract_pooled_activations.py \
  --variants-root variants \
  --ckpt-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
  --output-dir activations_variants/
```

This produces separate files for each variant:
- `pooled_rconv14.out__baseline.npy`
- `pooled_rconv14.out__zero_global.npy`

## Contextual Channel Detection
After extraction, run the contextual channel detector:

```bash
python3 contextual_channel_detector.py \
  --baseline activations_variants/pooled_rconv14.out__baseline.npy \
  --variant activations_variants/pooled_rconv14.out__zero_global.npy \
  --output contextual_mask.json \
  --threshold 0.1
```

This creates a JSON mask mapping channel IDs to "spatial" or "contextual" classifications for use in Step 4 NMF.

## Output Files
- `pooled_<layer>.npy` - Activation matrix (positions × channels)
- `pooled_meta.json` - Metadata about extraction
- `pos_index_to_npz.txt` - Mapping from matrix rows to source files

## Next Step
→ Go to `4_nmf_parts/` to factorize activations into interpretable parts. 