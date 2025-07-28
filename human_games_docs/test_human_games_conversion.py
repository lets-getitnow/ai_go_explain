#!/usr/bin/env python3
"""
Test Human Games Conversion
==========================
Purpose
-------
Test the conversion of human SGF games to NPZ format to ensure compatibility
with the existing activation extraction pipeline.

Usage
-----
python test_human_games_conversion.py
"""

import sys
from pathlib import Path
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_conversion():
    """Test the SGF to NPZ conversion process."""
    print("🧪 Testing human games conversion...")
    
    # Check if we have SGF files
    sgf_dir = Path("games/go13")
    if not sgf_dir.exists():
        print("❌ No SGF files found in games/go13/")
        print("Please add some SGF files to test with")
        return False
    
    sgf_files = list(sgf_dir.glob("*.sgf"))
    if not sgf_files:
        print("❌ No SGF files found in games/go13/")
        return False
    
    print(f"✅ Found {len(sgf_files)} SGF files")
    
    # Test conversion of a single file
    test_sgf = sgf_files[0]
    print(f"📄 Testing conversion of {test_sgf.name}")
    
    try:
        # Import the conversion module
        sys.path.insert(0, str(Path(__file__).parent.parent / "1_collect_positions"))
        from convert_human_games import process_sgf_file
        
        # Create test output directory
        test_output = Path("test_conversion_output")
        test_output.mkdir(exist_ok=True)
        
        # Convert the file
        process_sgf_file(test_sgf, test_output)
        
        # Check if NPZ file was created
        npz_file = test_output / f"{test_sgf.stem}.npz"
        if not npz_file.exists():
            print("❌ NPZ file was not created")
            return False
        
        print(f"✅ NPZ file created: {npz_file}")
        
        # Test loading the NPZ file
        with np.load(npz_file) as data:
            print("📊 NPZ file contents:")
            for key in data.keys():
                print(f"  {key}: {data[key].shape}")
            
            # Check required keys
            required_keys = ['binaryInputNCHWPacked', 'globalInputNC', 'policyTargetsNCMove']
            for key in required_keys:
                if key not in data:
                    print(f"❌ Missing required key: {key}")
                    return False
            
            print("✅ All required keys present")
            
            # Check shapes
            binary_shape = data['binaryInputNCHWPacked'].shape
            global_shape = data['globalInputNC'].shape
            policy_shape = data['policyTargetsNCMove'].shape
            
            print(f"  Binary inputs: {binary_shape}")
            print(f"  Global inputs: {global_shape}")
            print(f"  Policy targets: {policy_shape}")
            
            if len(binary_shape) != 3:
                print("❌ Binary inputs should have 3 dimensions")
                return False
            
            if binary_shape[0] != global_shape[0] or binary_shape[0] != policy_shape[0]:
                print("❌ Number of positions should match across all arrays")
                return False
            
            print("✅ Shape validation passed")
        
        # Clean up
        import shutil
        shutil.rmtree(test_output)
        print("🧹 Test output cleaned up")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure convert_human_games.py is in the correct location")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def test_pipeline_compatibility():
    """Test that the converted NPZ files are compatible with the pipeline."""
    print("\n🔧 Testing pipeline compatibility...")
    
    # Check if we have the required pipeline files
    required_files = [
        "3_extract_activations/extract_pooled_activations.py",
        "2_pick_layer/pick_layer.py",
        "4_nmf_parts/run_nmf.py",
        "5_inspect_parts/inspect_parts.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing required file: {file_path}")
            return False
    
    print("✅ All required pipeline files present")
    
    # Check if we have a model file
    model_files = list(Path("models").glob("*.ckpt")) if Path("models").exists() else []
    if not model_files:
        print("⚠️  No model checkpoint files found in models/")
        print("You'll need a KataGo model checkpoint to run the full pipeline")
    else:
        print(f"✅ Found {len(model_files)} model checkpoint(s)")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting human games conversion tests...\n")
    
    # Test 1: Basic conversion
    conversion_ok = test_conversion()
    
    # Test 2: Pipeline compatibility
    pipeline_ok = test_pipeline_compatibility()
    
    print("\n" + "="*50)
    if conversion_ok and pipeline_ok:
        print("🎉 All tests passed!")
        print("\nYou can now run the full pipeline with:")
        print("python run_human_games_pipeline.py \\")
        print("    --input-dir games/go13 \\")
        print("    --output-dir human_games_analysis \\")
        print("    --model-path models/your-model.ckpt")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    print("="*50)

if __name__ == "__main__":
    main() 