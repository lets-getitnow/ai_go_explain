# Step 1: Collect Positions

Generate varied 7×7 board positions using KataGo's self-play engine.

## Prerequisites
- KataGo binary installed
- Model file: `models/<latest_general_net>.bin.gz`  <!-- e.g. kata1-b28c512nbt-sXXXXX.bin.gz -->
- Config file: `selfplay7.cfg`

## Command
```bash
katago selfplay \
  -config selfplay.cfg \
  -models-dir models/ \
  -output-dir selfplay_out/ \
  -max-games-total 200
```

## Output
Creates `selfplay_out/` directory with the following structure:
```
selfplay_out/
├── log*.log (execution logs)
└── <latest_general_net>/
    ├── selfplay-*.cfg (generated config)
    ├── sgfs/
    │   └── *.sgfs (game records)
    └── tdata/
        └── *.npz (board positions)
```

The `.npz` files in `tdata/` contain the actual position data that will be used in Step 3 for activation extraction.

## Next Step
→ Go to `2_pick_layer/` to choose which network layer to analyze. 

## Optional: Generate Controlled Variants (Context Probes)
After collecting the raw `.npz` position files you can create *controlled variants* to probe which neural channels depend on global inputs (komi, move-history, ko state, etc.) vs pure board shape.

For the first probe we support `zero_global` – it zeroes out the entire `globalInputNC` vector while keeping `binaryInputNCHWPacked` intact.

```bash
python 1_collect_positions/generate_variants.py \
  --input-dir selfplay_out/ \
  --output-dir variants/ \
  --mode zero_global
```

This writes a parallel directory tree under `variants/zero_global/**` that mirrors the originals. Later, pass this directory to **Step 3** (`extract_pooled_activations.py --positions-dir variants/zero_global/`) to extract activations for the variant set.

> Tip: run the script **once per variant mode** you want to test (komi_sweep, ko_toggle, … once those modes are implemented).

### Variant Modes (road-map)
| Mode            | Purpose                              | Status |
|-----------------|--------------------------------------|--------|
| `zero_global`   | Zero out the entire global feature vector to isolate board-only channels | ✅ implemented |
| `komi_sweep`    | Vary komi ±N to identify komi-sensitive channels | 🚧 pending |
| `history_shuffle` | Randomise move history that leads to same final board | 🚧 pending |
| `ko_toggle`     | Flip a single ko capture to detect ko-aware channels | 🚧 pending |
| `board_holdout` | Feed identical board with *no* global changes to measure baseline variance | 🚧 pending |

Each variant produces its own subfolder under `variants/<mode>/…` keeping the original relative path, so downstream scripts can treat each probe as an independent "dataset slice".

## ✅ Successfully Completed: Variant Generation & Extraction

The variant generation workflow has been successfully tested and documented:

### Step 1: Generate Variants
```bash
# Create zero_global variant (zeros out globalInputNC)
python3 1_collect_positions/generate_variants.py \
  --input-dir selfplay_out \
  --output-dir variants \
  --mode zero_global

# Create baseline copy (for comparison)
cp -r selfplay_out variants/baseline
```

### Step 2: Extract Activations for All Variants
```bash
# Process both baseline and zero_global variants in one command
python3 3_extract_activations/extract_pooled_activations.py \
  --variants-root variants \
  --ckpt-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
  --output-dir 3_extract_activations/activations_variants
```

This produces:
- `pooled_rconv14.out__baseline.npy` (6603 × 4608)
- `pooled_rconv14.out__zero_global.npy` (6603 × 4608)
- Corresponding metadata and index files

The extraction script now supports `--variants-root` to automatically process all subdirectories as separate datasets, with each output file tagged with the variant name. 