# Step 3: Extract Activations

Extract pooled activations from a chosen layer of KataGo's *general* neural network (trained for all board sizes) using the `.bin.gz` inference checkpoint.

## Prerequisites
- Completed Step 1 (positions in `selfplay_out/`)
- Completed Step 2 (`layer_selection.yml` exists)
- KataGo inference model file (`.bin.gz` from https://katagotraining.org/networks/)
- Python dependencies installed

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install torch numpy pyyaml
   ```

2. **Clone KataGo for PyTorch helper code:**
   ```bash
   git clone https://github.com/lightvector/KataGo.git
   ```

3. **Place the `.bin.gz` network** inside `models/` (e.g. `kata1-b28c512nbt-s9853922560-d5031756885.bin.gz`).  No conversion to `.ckpt` is needed – the script loads inference files directly.

4. *(Optional – recommended for macOS)* **Verify your PyTorch back-end:**
   ```bash
   # from inside 3_extract_activations/
   python verify_pytorch_device.py
   ```
   If the script reports
   ```text
   MPS built     : True
   MPS available : True
   ```
   then your Apple-Silicon GPU is usable – run the extractor with `--device mps` for faster execution. Otherwise default to `--device cpu` (or `cuda:0` on machines with an NVIDIA GPU).

## Extract Activations

```bash
cd 3_extract_activations
python extract_pooled_activations.py \
  --positions-dir ../selfplay_out \
  --model-path   ../models/kata1-b28c512nbt-s9853922560-d5031756885.bin.gz \
  --board-size   7 \
  --batch-size   512 \
  --device       mps        # mps (Apple Silicon), cpu, or cuda:0
```

## How It Works

The script:
1. **Loads the KataGo model** (supports `.bin.gz` inference files).
2. **Requests exactly one intermediate tensor** via KataGo's `ExtraOutputs` API using the layer name stored in `layer_selection.yml`.
3. **Processes 7 × 7 positions** in batches for efficiency.
4. **Spatially pools each channel** (mean across 7 × 7 spatial dimensions).
5. **Produces a non-negative, column-scaled matrix** suitable for NMF.

## Output

Creates `activations/` directory with:
```text
activations/
  pooled_<layer>.npy      # (N_positions, C) – main data matrix
  pooled_meta.json        # extraction metadata
  pos_index_to_npz.txt    # mapping row → original position file
```

## Troubleshooting

**"Layer '<name>' not found"**
- Run `python pick_layer.py --list` to view all layer names in your model.

**"Mismatched board shape"**
- Ensure your `.npz` files contain 7 × 7 tensors.

**"CUDA out of memory"**
- Lower `--batch-size` or switch to CPU.

## Next Step
→ Continue to Step 4 (NMF parts finding) with your extracted activation matrix. 