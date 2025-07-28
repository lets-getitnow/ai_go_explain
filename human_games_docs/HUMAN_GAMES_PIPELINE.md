# Human Games Analysis Pipeline

This document explains how to run the complete activation analysis pipeline (steps 1-5) on human SGF games instead of self-play data.

## Overview

The pipeline analyzes human Go games to understand what patterns the neural network learns from real human play. It follows the same 5-step process as the self-play pipeline:

1. **Convert SGF to NPZ**: Transform human SGF games into the format expected by the activation extraction pipeline
2. **Pick Layer**: Choose which neural network layer to analyze
3. **Extract Activations**: Run inference to get activation patterns for each position
4. **NMF Analysis**: Find interpretable parts using Non-negative Matrix Factorization
5. **Inspect Parts**: Generate detailed analysis and HTML reports

## Prerequisites

- KataGo installation with Python modules
- KataGo model checkpoint file (e.g., `models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt`)
- Human SGF games in a directory (e.g., `games/go13/`)

## Quick Start

Run the complete pipeline with one command:

```bash
python run_human_games_pipeline.py \
    --input-dir games/go13 \
    --output-dir human_games_analysis \
    --model-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt
```

This will:
1. Convert all SGF files in `games/go13/` to NPZ format
2. Use existing layer selection (or create new one if needed)
3. Extract activations from the chosen layer
4. Run NMF analysis to find ~50 interpretable parts
5. Generate HTML reports showing the strongest examples of each part

## Step-by-Step Process

### Step 1: Convert SGF to NPZ

Convert human SGF games to the NPZ format required by the pipeline:

```bash
python 1_collect_positions/convert_human_games.py \
    --input-dir games/go13 \
    --output-dir human_games_analysis/npz_files \
    --board-size 7
```

**What this does:**
- Parses each SGF file to extract all moves
- Creates board state tensors for each position
- Generates policy targets for each move
- Saves everything in NPZ format compatible with the existing pipeline

### Step 2: Pick Layer (Optional)

If you don't have an existing `layer_selection.yml`, run:

```bash
python 2_pick_layer/pick_layer.py \
    --model-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
    --board-size 7
```

### Step 3: Extract Activations

Extract activation patterns from the chosen layer:

```bash
python 3_extract_activations/extract_pooled_activations.py \
    --positions-dir human_games_analysis/npz_files \
    --ckpt-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
    --output-dir human_games_analysis/activations \
    --batch-size 256
```

### Step 4: Run NMF Analysis

Find interpretable parts using Non-negative Matrix Factorization:

```bash
python 4_nmf_parts/run_nmf.py \
    --activations-file human_games_analysis/activations/pooled_rconv14.out.npy \
    --output-dir human_games_analysis/nmf_parts \
    --num-components 50 \
    --max-iter 1000
```

### Step 5: Inspect Parts

Generate detailed analysis and HTML reports:

```bash
python 5_inspect_parts/inspect_parts.py \
    --activations-file human_games_analysis/activations/pooled_rconv14.out.npy \
    --nmf-components human_games_analysis/nmf_parts/nmf_components.npy \
    --nmf-activations human_games_analysis/nmf_parts/nmf_activations.npy \
    --output-dir human_games_analysis/inspect_parts \
    --num-positions-per-part 10

python 5_inspect_parts/generate_html_reports.py \
    --summary-file human_games_analysis/inspect_parts/strong_positions_summary.csv \
    --output-dir human_games_analysis/html_reports
```

## Output Structure

After running the pipeline, you'll have:

```
human_games_analysis/
├── npz_files/                    # Converted SGF games
│   ├── game1.npz
│   ├── game2.npz
│   └── ...
├── activations/                   # Extracted activations
│   ├── pooled_rconv14.out.npy
│   ├── pooled_meta.json
│   └── pos_index_to_npz.txt
├── nmf_parts/                    # NMF analysis results
│   ├── nmf_components.npy
│   ├── nmf_activations.npy
│   └── nmf_meta.json
├── inspect_parts/                # Detailed analysis
│   ├── strong_positions_summary.csv
│   └── output/
│       ├── pos_123/
│       │   ├── analysis.json
│       │   ├── board.npy
│       │   └── game.sgf
│       └── ...
└── html_reports/                 # Interactive HTML reports
    ├── index.html
    ├── pos_123_part1_rank1_analysis.html
    └── ...
```

## Key Differences from Self-Play Pipeline

### Data Source
- **Self-play**: Uses KataGo's self-play engine to generate positions
- **Human games**: Uses real human SGF games as the data source

### Position Generation
- **Self-play**: Each position is a training slice with search statistics
- **Human games**: Each position is a real game state at a specific move

### Analysis Focus
- **Self-play**: Understands what the AI learns from its own play
- **Human games**: Understands what the AI learns from human play patterns

## Troubleshooting

### Common Issues

1. **KataGo import errors**: Make sure KataGo is properly installed and the Python path is set correctly

2. **SGF parsing errors**: Some SGF files may have unusual formats. The converter will skip problematic files and continue with the rest.

3. **Memory issues**: For large datasets, reduce the batch size in step 3:
   ```bash
   --batch-size 128  # or even smaller
   ```

4. **Model compatibility**: Ensure your model checkpoint matches the expected input format for 7x7 boards.

### Debugging

- Check the console output for detailed error messages
- Look at the generated NPZ files to verify the conversion worked correctly
- Use the `--skip-conversion` flag to reuse existing NPZ files if the conversion step fails

## Advanced Usage

### Custom Board Sizes

The pipeline supports different board sizes (though 7x7 is most tested):

```bash
python run_human_games_pipeline.py \
    --input-dir games/go19 \
    --output-dir human_games_analysis_19x19 \
    --model-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
    --board-size 19
```

### Partial Pipeline Runs

Skip steps you've already completed:

```bash
# Skip conversion if NPZ files already exist
python run_human_games_pipeline.py \
    --input-dir games/go13 \
    --output-dir human_games_analysis \
    --model-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
    --skip-conversion

# Skip layer selection if layer_selection.yml exists
python run_human_games_pipeline.py \
    --input-dir games/go13 \
    --output-dir human_games_analysis \
    --model-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
    --skip-layer-pick
```

## Understanding the Results

The HTML reports show:
- **NMF Part Analysis**: Which neural network components fire strongest at each position
- **Go Pattern Analysis**: What Go concepts (moves, game phase, policy confidence) are present
- **Part Comparison**: How each part relates to others in the network

This helps understand:
- What patterns the AI recognizes in human play
- Which neural components are specialized for specific Go concepts
- How the AI's understanding differs from human intuition

## Next Steps

After running the pipeline, you can:
1. **Analyze specific parts**: Look at the HTML reports to understand what each NMF part represents
2. **Compare with self-play**: Run the same analysis on self-play data to see differences
3. **Extend the analysis**: Add more sophisticated Go pattern detection
4. **Train sparse autoencoders**: Replace NMF with more sophisticated feature learning methods 