# Human Games Analysis Documentation

This directory contains all documentation and tools for analyzing human SGF games using the ai_go_explain pipeline.

## 📁 Contents

### Documentation
- **[HUMAN_GAMES_PIPELINE.md](HUMAN_GAMES_PIPELINE.md)** - Complete guide for human games analysis
- **[README.md](README.md)** - This file: overview of human games documentation

### Pipeline Tools
- **[run_human_games_pipeline.py](run_human_games_pipeline.py)** - Complete pipeline runner
- **[example_human_games_run.sh](example_human_games_run.sh)** - Example shell script
- **[test_human_games_conversion.py](test_human_games_conversion.py)** - Test SGF conversion

## 🚀 Quick Start

### One-Command Pipeline
```bash
python3 human_games_docs/run_human_games_pipeline.py \
    --input-dir games/go13 \
    --output-dir test/human_games_analysis \
    --model-path models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt \
    --max-files 1000 \
    --processor mps \
    --board-size 13
```

### Example Script
```bash
./human_games_docs/example_human_games_run.sh
```

### Test Conversion
```bash
python human_games_docs/test_human_games_conversion.py
```

## 📚 Documentation Structure

### Main Documentation
- **[../README.md](../README.md)** - Main project overview with quick start and common fixes
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete documentation index
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Detailed troubleshooting guide

### Human Games Specific
- **[HUMAN_GAMES_PIPELINE.md](HUMAN_GAMES_PIPELINE.md)** - Detailed human games guide

## 🔧 Pipeline Steps

1. **Convert SGF to NPZ**: Transform human SGF games into the format expected by the activation extraction pipeline
2. **Pick Layer**: Choose which neural network layer to analyze
3. **Extract Activations**: Run inference to get activation patterns for each position
4. **Run NMF**: Find interpretable parts using Non-negative Matrix Factorization
5. **Inspect Parts**: Generate detailed analysis and HTML reports

## 📊 Output Structure

```
test/human_games_analysis/
├── npz_files/           # Converted SGF games
├── activations/          # Extracted activations
├── nmf_parts/           # NMF analysis results
├── inspect_parts/       # Detailed analysis
└── html_reports/        # Interactive HTML reports
```

## 🧪 Testing

### Test Setup
```bash
# Test conversion
python human_games_docs/test_human_games_conversion.py

# Test device
python ../3_extract_activations/verify_pytorch_device.py

# Test imports
python -c "import katago, torch; print('Setup OK')"
```

### Test Individual Steps
```bash
# Test SGF conversion
python ../1_collect_positions/convert_human_games.py \
    --input-dir games/go13 \
    --output-dir test_output \
    --board-size 7

# Test activation extraction
python ../3_extract_activations/extract_pooled_activations.py \
    --positions-dir test_output \
    --ckpt-path your_model.ckpt \
    --output-dir test_activations \
    --batch-size 8
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

### HTML Reports
- Interactive Go boards with Besogo
- NMF part analysis with activation strengths
- Go pattern analysis (moves, game phase, policy confidence)
- Part comparison and uniqueness scores

## 🐛 Troubleshooting

### Common Issues
- **Import Errors**: Check KataGo and PyTorch installation
- **Memory Issues**: Reduce batch size or use CPU
- **File Not Found**: Check model and SGF file paths
- **Conversion Errors**: Test with individual files

### Debugging Commands
```bash
# Check NPZ files
python -c "import numpy as np; data=np.load('file.npz'); print(list(data.keys()))"

# Check activations
python -c "import numpy as np; act=np.load('pooled_rconv14.out.npy'); print(f'Shape: {act.shape}')"

# Check NMF results
python -c "import numpy as np; comp=np.load('nmf_components.npy'); print(f'Components: {comp.shape}')"
```

## 📞 Help Resources

- **[HUMAN_GAMES_PIPELINE.md](HUMAN_GAMES_PIPELINE.md)** - Detailed human games guide
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Detailed troubleshooting guide
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete documentation index
- **[../README.md](../README.md)** - Main project overview with quick start

## 🎯 Key Benefits

- **Same Analysis**: Uses the exact same pipeline as self-play data
- **Human Insights**: Understands what patterns the AI learns from real human play
- **Interactive Reports**: HTML visualizations with Go boards and detailed analysis
- **Flexible**: Supports different board sizes and can skip steps if needed 