# Documentation Index

This document provides an organized overview of all documentation in the ai_go_explain project.

## 📚 Core Documentation

### Getting Started
- **[../README.md](../README.md)** - Main project overview and quick start guide
- **[HUMAN_GAMES_PIPELINE.md](HUMAN_GAMES_PIPELINE.md)** - Complete guide for analyzing human SGF games
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - This file: organized documentation index

### Pipeline Documentation
- **[1_collect_positions/README.md](1_collect_positions/README.md)** - Position collection and variant generation
- **[2_pick_layer/README.md](2_pick_layer/README.md)** - Layer selection for analysis
- **[3_extract_activations/README.md](3_extract_activations/README.md)** - Activation extraction process
- **[4_nmf_parts/README.md](4_nmf_parts/README.md)** - NMF analysis and parts discovery
- **[5_inspect_parts/README.md](5_inspect_parts/README.md)** - Parts inspection and HTML report generation

## 🎯 Use Cases

### Self-Play Analysis
**Purpose**: Understand what the AI learns from its own play patterns.

**Pipeline**: 
1. Generate positions using KataGo self-play
2. Extract activations from chosen layer
3. Run NMF to find interpretable parts
4. Inspect parts to understand AI behavior

**Key Files**:
- `selfplay.cfg` - Self-play configuration
- `1_collect_positions/generate_variants.py` - Create controlled variants
- `3_extract_activations/contextual_channel_detector.py` - Detect global context sensitivity

### Human Games Analysis
**Purpose**: Understand what the AI learns from real human play patterns.

**Pipeline**:
1. Convert human SGF games to NPZ format
2. Extract activations from chosen layer
3. Run NMF to find interpretable parts
4. Inspect parts to understand human-AI differences

**Key Files**:
- `run_human_games_pipeline.py` - Complete pipeline runner
- `../1_collect_positions/convert_human_games.py` - SGF to NPZ converter
- `example_human_games_run.sh` - Example execution script

## 🔧 Scripts and Tools

### Pipeline Runners
- `run_human_games_pipeline.py` - Complete human games pipeline
- `example_human_games_run.sh` - Example shell script

### Conversion Tools
- `1_collect_positions/convert_human_games.py` - SGF to NPZ converter
- `1_collect_positions/generate_variants.py` - Create controlled variants

### Testing and Validation
- `test_human_games_conversion.py` - Test SGF conversion
- `../3_extract_activations/verify_pytorch_device.py` - Verify PyTorch setup
- `../4_nmf_parts/rank_analysis/` - NMF rank selection analysis

### Analysis Tools
- `3_extract_activations/contextual_channel_detector.py` - Detect global context sensitivity
- `5_inspect_parts/inspect_parts.py` - Detailed parts analysis
- `5_inspect_parts/generate_html_reports.py` - HTML report generation

## 📊 Output Formats

### NPZ Files
- `binaryInputNCHWPacked` - Packed binary board representation
- `globalInputNC` - Global features (komi, move number, etc.)
- `policyTargetsNCMove` - Policy targets for each position

### Activation Files
- `pooled_<layer>.npy` - Pooled activations matrix (N_positions × C_channels)
- `pooled_meta.json` - Metadata about extraction process
- `pos_index_to_npz.txt` - Mapping from positions to source files

### NMF Results
- `nmf_components.npy` - Component weights (C_channels × N_components)
- `nmf_activations.npy` - Component activations (N_positions × N_components)
- `nmf_meta.json` - NMF parameters and metadata

### HTML Reports
- Interactive Go boards with Besogo
- NMF part analysis with activation strengths
- Go pattern analysis (moves, game phase, policy confidence)
- Part comparison and uniqueness scores

## 🧪 Testing and Validation

### Conversion Testing
```bash
python test_human_games_conversion.py
```

### Pipeline Validation
```bash
# Test SGF conversion
python 1_collect_positions/convert_human_games.py \
    --input-dir games/go13 \
    --output-dir test_output \
    --board-size 7

# Test activation extraction
python 3_extract_activations/extract_pooled_activations.py \
    --positions-dir test_output \
    --ckpt-path models/your-model.ckpt \
    --output-dir test_activations
```

### Device Verification
```bash
python 3_extract_activations/verify_pytorch_device.py
```

## 📈 Analysis Examples

### Contextual Channel Detection
```bash
python 3_extract_activations/contextual_channel_detector.py \
    --baseline-file activations/pooled_rconv14.out__baseline.npy \
    --variant-file activations/pooled_rconv14.out__zero_global.npy \
    --output-file contextual_analysis.json
```

### NMF Rank Analysis
```bash
python 4_nmf_parts/rank_analysis/run_rank_analysis.py \
    --activations-file activations/pooled_rconv14.out.npy \
    --output-dir nmf_parts/rank_analysis
```

### HTML Report Generation
```bash
python 5_inspect_parts/generate_html_reports.py \
    --summary-file inspect_parts/strong_positions_summary.csv \
    --output-dir html_reports
```

## 🔍 Understanding Results

### NMF Parts
- **Components**: Neural network patterns that fire together
- **Activations**: How strongly each pattern is present in each position
- **Ranking**: Positions ordered by activation strength within each part

### Go Pattern Analysis
- **Move Type**: Normal play, pass, or resign
- **Game Phase**: Opening, middle-game, or endgame
- **Policy Confidence**: How certain the AI is about the move
- **Policy Entropy**: How spread out the AI's move probabilities are

### Part Comparison
- **Uniqueness Score**: How distinct each part is from others
- **Part Rank**: Ordering by average activation strength
- **Max Other Activation**: Highest activation from other parts at each position

## 🚀 Quick Reference

### Self-Play Pipeline
```bash
# Generate positions
katago selfplay -config selfplay.cfg -models-dir models/ -output-dir selfplay_out/

# Extract activations
python 3_extract_activations/extract_pooled_activations.py \
    --positions-dir selfplay_out/ \
    --ckpt-path models/your-model.ckpt \
    --output-dir activations/

# Run NMF
python 4_nmf_parts/run_nmf.py \
    --activations-file activations/pooled_rconv14.out.npy \
    --output-dir nmf_parts/

# Inspect parts
python 5_inspect_parts/inspect_parts.py \
    --activations-file activations/pooled_rconv14.out.npy \
    --nmf-components nmf_parts/nmf_components.npy \
    --nmf-activations nmf_parts/nmf_activations.npy \
    --output-dir inspect_parts/
```

### Human Games Pipeline
```bash
# Complete pipeline
python run_human_games_pipeline.py \
    --input-dir games/go13 \
    --output-dir human_games_analysis \
    --model-path models/your-model.ckpt
```

## 📝 Contributing

### Adding New Analysis Methods
1. Create new script in appropriate directory
2. Follow existing naming conventions
3. Add documentation to relevant README files
4. Update this documentation index

### Extending the Pipeline
1. Modify conversion scripts for new data formats
2. Add new analysis steps to pipeline runners
3. Update HTML report templates for new visualizations
4. Test with both self-play and human games data

## 🔗 External Resources

- **[KataGo Repository](https://github.com/lightvector/KataGo)** - Neural network implementation
- **[Stanford Sparse Autoencoder Paper](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)** - Theoretical foundation
- **[Besogo](https://github.com/kaorahi/besogo)** - Go board visualization library 