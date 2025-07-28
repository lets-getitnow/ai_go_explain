#!/usr/bin/env python3
"""
Run NMF Analysis with 3x3 Grid Pooling

This script runs NMF analysis on the new 3x3 pooled activation data.
The 3x3 pooling preserves spatial information while still being suitable
for NMF analysis.

Key differences from global average pooling:
- Data shape: (N_positions, C*9) instead of (N_positions, C)
- Spatial information preserved in 9 sub-regions
- Components can specialize in spatial regions (corner, edge, center)
- Better interpretability for Go-specific patterns

Expected improvements:
- Parts that fire in specific board regions (corner ladders, center eyes)
- Reduced "board-density" themes
- More meaningful spatial patterns
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run NMF analysis with 3x3 pooled data."""
    print("=== NMF Analysis with 3x3 Grid Pooling ===")
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Check if 3x3 pooled data exists
    activations_path = project_root / "3_extract_activations" / "activations" / "pooled_rconv14.out.npy"
    meta_path = project_root / "3_extract_activations" / "activations" / "pooled_meta.json"
    
    if not activations_path.exists():
        print(f"❌ 3x3 pooled activations not found: {activations_path}")
        print("Please run the 3x3 extraction first:")
        print("  cd 3_extract_activations && python run_3x3_extraction.py")
        sys.exit(1)
    
    if not meta_path.exists():
        print(f"❌ 3x3 pooled metadata not found: {meta_path}")
        print("Please run the 3x3 extraction first")
        sys.exit(1)
    
    # Check metadata to confirm 3x3 pooling
    import json
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    pooling_method = meta.get("pooling_method", "unknown")
    if pooling_method != "3x3_grid":
        print(f"⚠️  Warning: Pooling method is '{pooling_method}', expected '3x3_grid'")
        print("This may not be 3x3 pooled data")
    
    print(f"✅ Found 3x3 pooled data: {pooling_method}")
    print(f"📊 Original channels: {meta.get('original_channels', 'unknown')}")
    print(f"📊 Pooled channels: {meta.get('pooled_channels', 'unknown')}")
    
    # Run NMF analysis
    print("\n🔄 Running NMF analysis with 3x3 pooled data...")
    
    cmd = [
        sys.executable, "run_nmf.py"
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True, capture_output=True, text=True)
        print("✅ NMF analysis completed successfully!")
        print("📊 Results saved in 4_nmf_parts/ directory")
        print("📊 Expected improvements:")
        print("  - Spatial-specific components (corner, edge, center)")
        print("  - Better Go pattern recognition")
        print("  - Reduced board-density themes")
    except subprocess.CalledProcessError as e:
        print(f"❌ NMF analysis failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 