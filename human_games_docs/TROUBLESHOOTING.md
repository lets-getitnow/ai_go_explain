# Troubleshooting Guide

This guide helps you resolve common issues when running the ai_go_explain pipeline.

## 🚨 Common Issues

### Import Errors

#### KataGo Import Issues
**Error**: `ImportError: No module named 'katago'`

**Solution**:
```bash
# Make sure KataGo is cloned in the project directory
git clone https://github.com/lightvector/KataGo.git

# Set PYTHONPATH to include KataGo
export PYTHONPATH="${PYTHONPATH}:$(pwd)/KataGo/python"

# Or add to your shell profile
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)/KataGo/python"' >> ~/.bashrc
```

#### PyTorch Import Issues
**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
# Install PyTorch
pip install torch torchvision torchaudio

# For CUDA support (if you have a GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### File Not Found Errors

#### Model File Missing
**Error**: `FileNotFoundError: models/your-model.ckpt`

**Solution**:
1. Download a KataGo model checkpoint
2. Place it in the `models/` directory
3. Update the path in your command

```bash
# Example model download
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s9584861952-d4960414494.bin.gz
gunzip kata1-b28c512nbt-s9584861952-d4960414494.bin.gz
mkdir -p models/kata1-b28c512nbt-s9584861952-d4960414494
mv kata1-b28c512nbt-s9584861952-d4960414494.bin models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt
```

#### SGF Files Missing
**Error**: `No SGF files found in games/go13/`

**Solution**:
```bash
# Create the directory and add SGF files
mkdir -p games/go13
# Add your .sgf files to games/go13/
```

### Memory Issues

#### Out of Memory During Activation Extraction
**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python 3_extract_activations/extract_pooled_activations.py \
    --positions-dir your_positions \
    --ckpt-path your_model.ckpt \
    --output-dir activations \
    --batch-size 64  # Try smaller values: 32, 16, 8
```

#### Large Dataset Memory Issues
**Error**: `MemoryError` during NMF

**Solution**:
```bash
# Use fewer components
python 4_nmf_parts/run_nmf.py \
    --activations-file activations/pooled_rconv14.out.npy \
    --output-dir nmf_parts \
    --num-components 25  # Reduce from 50
```

### Conversion Issues

#### SGF Parsing Errors
**Error**: `Warning: Invalid coordinate format`

**Solution**:
- Check SGF file format
- Some SGF files may have unusual coordinate systems
- The converter will skip problematic files and continue

#### NPZ Format Issues
**Error**: `KeyError: binaryInputNCHWPacked missing`

**Solution**:
```bash
# Test the conversion
python test_human_games_conversion.py

# Check NPZ file contents
python -c "import numpy as np; data=np.load('your_file.npz'); print(list(data.keys()))"
```

### Device Issues

#### CUDA Device Not Found
**Error**: `RuntimeError: CUDA device not found`

**Solution**:
```bash
# Check available devices
python 3_extract_activations/verify_pytorch_device.py

# Force CPU usage
python 3_extract_activations/extract_pooled_activations.py \
    --positions-dir your_positions \
    --ckpt-path your_model.ckpt \
    --output-dir activations \
    --device cpu
```

#### GPU Memory Issues
**Error**: `CUDA out of memory`

**Solution**:
```bash
# Use CPU instead
--device cpu

# Or reduce batch size
--batch-size 32
```

## 🔧 Debugging Steps

### 1. Test Basic Setup
```bash
# Test KataGo import
python -c "import katago; print('KataGo import successful')"

# Test PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test device availability
python 3_extract_activations/verify_pytorch_device.py
```

### 2. Test SGF Conversion
```bash
# Test with a single file
python 1_collect_positions/convert_human_games.py \
    --input-dir games/go13 \
    --output-dir test_output \
    --board-size 7

# Check output
ls test_output/
python -c "import numpy as np; data=np.load('test_output/your_file.npz'); print(data.keys())"
```

### 3. Test Activation Extraction
```bash
# Test with a small subset
python 3_extract_activations/extract_pooled_activations.py \
    --positions-dir test_output \
    --ckpt-path your_model.ckpt \
    --output-dir test_activations \
    --batch-size 8
```

### 4. Test NMF
```bash
# Test with fewer components
python 4_nmf_parts/run_nmf.py \
    --activations-file test_activations/pooled_rconv14.out.npy \
    --output-dir test_nmf \
    --num-components 10
```

## 📊 Performance Optimization

### For Large Datasets
```bash
# Use smaller batch sizes
--batch-size 32

# Use fewer NMF components
--num-components 25

# Use CPU if GPU memory is limited
--device cpu
```

### For Faster Processing
```bash
# Use GPU if available
--device cuda

# Increase batch size (if memory allows)
--batch-size 256

# Use more NMF components for better analysis
--num-components 50
```

## 🐛 Common Error Messages

### Import Errors
```
ImportError: No module named 'katago'
```
→ Install KataGo and set PYTHONPATH

```
ModuleNotFoundError: No module named 'torch'
```
→ Install PyTorch

### File Errors
```
FileNotFoundError: models/your-model.ckpt
```
→ Download and place model file

```
No SGF files found in games/go13/
```
→ Add SGF files to the directory

### Memory Errors
```
RuntimeError: CUDA out of memory
```
→ Reduce batch size or use CPU

```
MemoryError
```
→ Use fewer NMF components

### Format Errors
```
KeyError: binaryInputNCHWPacked missing
```
→ Check NPZ file format

```
ValueError: Unexpected board shape
```
→ Check board size configuration

## 🔍 Advanced Debugging

### Check NPZ File Contents
```python
import numpy as np

# Load and inspect NPZ file
with np.load('your_file.npz') as data:
    print("Keys:", list(data.keys()))
    for key, value in data.items():
        print(f"{key}: {value.shape} {value.dtype}")
```

### Check Activation File
```python
import numpy as np

# Load activations
activations = np.load('pooled_rconv14.out.npy')
print(f"Shape: {activations.shape}")
print(f"Min: {activations.min()}, Max: {activations.max()}")
print(f"Mean: {activations.mean()}, Std: {activations.std()}")
```

### Check NMF Results
```python
import numpy as np

# Load NMF results
components = np.load('nmf_components.npy')
activations = np.load('nmf_activations.npy')

print(f"Components shape: {components.shape}")
print(f"Activations shape: {activations.shape}")
print(f"Component sparsity: {(components == 0).mean():.2%}")
```

## 📞 Getting Help

### Before Asking for Help
1. Run the test script: `python test_human_games_conversion.py`
2. Check the troubleshooting steps above
3. Gather error messages and system information

### Documentation Links
- **[HUMAN_GAMES_PIPELINE.md](HUMAN_GAMES_PIPELINE.md)** - Complete human games guide
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete documentation index
- **[../README.md](../README.md)** - Main project overview with quick start

### Useful Information to Include
- Operating system and Python version
- Error messages and stack traces
- File paths and directory structure
- Hardware specifications (CPU, GPU, RAM)

### System Information Commands
```bash
# Python version
python --version

# PyTorch version
python -c "import torch; print(torch.__version__)"

# GPU information
nvidia-smi

# System memory
free -h

# Directory structure
ls -la
tree -L 2
``` 