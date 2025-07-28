# Step 1: Collect Positions

This step collects board positions from different sources for neural network analysis. There are three main approaches:

## 🎯 Three Starting Points

### 1. **Selfplay AI Games** (Original Pipeline)
Generate varied 7×7 board positions using KataGo's self-play engine.

#### Prerequisites
- KataGo binary installed
- Model file: `models/<latest_general_net>.bin.gz`  <!-- e.g. kata1-b28c512nbt-sXXXXX.bin.gz -->
- Config file: `selfplay7.cfg`

#### Command
```bash
katago selfplay \
  -config selfplay.cfg \
  -models-dir models/ \
  -output-dir selfplay_out/ \
  -max-games-total 200
```

#### Output
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

---

### 2. **Human Games** (New Pipeline)
Convert human SGF game files to KataGo-compatible format.

#### Prerequisites
- Human SGF files in `games/` directory
- Python with required dependencies

#### Command
```bash
python 1_collect_positions/convert_human_games.py \
  --input-dir games/go13/ \
  --output-dir human_games_output/ \
  --board-size 7
```

#### Output
Creates `.npz` files compatible with the activation extraction pipeline:
```
human_games_output/
└── npz_files/
    ├── pos_*.npz (converted board positions)
    └── metadata.json (game information)
```

#### Quick Start
For a complete human games pipeline, see `human_games_docs/`:
```bash
python human_games_docs/run_human_games_pipeline.py \
  --input-dir games/go13/ \
  --output-dir human_games_analysis/ \
  --model-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt
```

---

### 3. **Contextual Channels Analysis** (Advanced)
Generate controlled variants to probe neural channel dependencies on global inputs vs pure board shape.

#### Prerequisites
- Completed selfplay data collection (see approach #1)
- Python with required dependencies

#### Command
```bash
# Create zero_global variant (zeros out globalInputNC)
python 1_collect_positions/generate_variants.py \
  --input-dir selfplay_out \
  --output-dir variants \
  --mode zero_global

# Create baseline copy (for comparison)
cp -r selfplay_out variants/baseline
```

#### Output
Creates variant datasets for comparative analysis:
```
variants/
├── baseline/          # Original selfplay data
└── zero_global/      # Zeroed global inputs
    └── tdata/
        └── *.npz (modified positions)
```

#### Variant Modes (road-map)
| Mode            | Purpose                              | Status |
|-----------------|--------------------------------------|--------|
| `zero_global`   | Zero out the entire global feature vector to isolate board-only channels | ✅ implemented |
| `komi_sweep`    | Vary komi ±N to identify komi-sensitive channels | 🚧 pending |
| `history_shuffle` | Randomise move history that leads to same final board | 🚧 pending |
| `ko_toggle`     | Flip a single ko capture to detect ko-aware channels | 🚧 pending |
| `board_holdout` | Feed identical board with *no* global changes to measure baseline variance | 🚧 pending |

Each variant produces its own subfolder under `variants/<mode>/…` keeping the original relative path, so downstream scripts can treat each probe as an independent "dataset slice".

---

## 📊 Analysis Progress

### ✅ **Contextual Channels Analysis Completed**
See `CONTEXTUAL_CHANNELS_PROGRESS.md` for detailed progress on the contextual channels investigation.

### ✅ **Human Games Pipeline Implemented**
See `human_games_docs/HUMAN_GAMES_PIPELINE.md` for complete documentation on processing human SGF files.

### ✅ **Variant Generation & Extraction Tested**
The variant generation workflow has been successfully tested and documented:

#### Step 1: Generate Variants
```bash
# Create zero_global variant (zeros out globalInputNC)
python3 1_collect_positions/generate_variants.py \
  --input-dir selfplay_out \
  --output-dir variants \
  --mode zero_global

# Create baseline copy (for comparison)
cp -r selfplay_out variants/baseline
```

#### Step 2: Extract Activations for All Variants
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

---

## 🎯 Next Steps

After collecting positions using any of the three approaches above:

1. **For Selfplay/Human Games**: → Go to `2_pick_layer/` to choose which network layer to analyze
2. **For Contextual Channels**: → Use `3_extract_activations/extract_pooled_activations.py` with `--variants-root` to process all variants

## 📚 Related Documentation

- `CONTEXTUAL_CHANNELS_PROGRESS.md` - Detailed progress on contextual channels analysis
- `FINAL_SUMMARY.md` - Summary of findings and conclusions
- `human_games_docs/` - Complete human games pipeline documentation 