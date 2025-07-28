#!/usr/bin/env python3
"""
Run 3x3 Grid Pooling Extraction

This script runs the updated activation extraction with 3x3 grid pooling
instead of global average pooling. This preserves spatial information
while still reducing dimensionality for NMF analysis.

The 3x3 pooling:
- Splits 7x7 activations into 9 regions: (0,3), (3,5), (5,7) for each dimension
- Averages each region to get 9 values per channel
- Concatenates all regions: (N, C*9) instead of (N, C)

This preserves coarse spatial information while avoiding the 49x blowup
of full spatial dimensions.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run 3x3 pooling extraction."""
    print("=== 3x3 Grid Pooling Activation Extraction ===")
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Check if we have the required files
    model_path = project_root / "models" / "kata1-b28c512nbt-s9584861952-d4960414494" / "model.ckpt"
    positions_dir = project_root / "selfplay_out"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the KataGo model is available")
        sys.exit(1)
    
    if not positions_dir.exists():
        print(f"❌ Positions directory not found: {positions_dir}")
        print("Please run step 1 to collect positions first")
        sys.exit(1)
    
    # Run the extraction with 3x3 pooling
    print("🔄 Running 3x3 grid pooling extraction...")
    print(f"📁 Model: {model_path}")
    print(f"📁 Positions: {positions_dir}")
    
    cmd = [
        sys.executable, "extract_pooled_activations.py",
        "--positions-dir", str(positions_dir),
        "--ckpt-path", str(model_path),
        "--batch-size", "256",
        "--device", "cpu"
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True, text=True)
        print("✅ 3x3 pooling extraction completed successfully!")
        print("📊 Output files created in activations/ directory")
        print("📊 New data shape: (N_positions, C*9) instead of (N_positions, C)")
        print("📊 Spatial information preserved in 3x3 grid")
    except subprocess.CalledProcessError as e:
        print(f"❌ Extraction failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 