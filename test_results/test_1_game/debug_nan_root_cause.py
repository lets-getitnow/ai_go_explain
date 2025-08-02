#!/usr/bin/env python3
"""
Debug the root cause of NaN activations in the KataGo model.
"""

import torch
import numpy as np
from pathlib import Path
from katago.train.load_model import load_model
from katago.train.model_pytorch import ExtraOutputs

def test_model_loading():
    """Test model loading with different pos_len values."""
    print("=== Testing Model Loading ===")
    
    model_path = "models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt"
    
    # Test with pos_len=13 (human games board size)
    print("\n1. Testing with pos_len=13:")
    try:
        model, swa_model, other_state_dict = load_model(
            model_path, use_swa=False, device="cpu", pos_len=13, verbose=False
        )
        print("✅ Model loaded successfully with pos_len=13")
        
        # Test forward pass
        dummy_binary = torch.randn(1, 22, 13, 13)
        dummy_global = torch.randn(1, 19)
        
        with torch.no_grad():
            output = model(dummy_binary, dummy_global)
            print(f"✅ Forward pass successful, output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"   Output tuple length: {len(output)}")
                for i, out in enumerate(output):
                    print(f"   Output[{i}] shape: {out.shape}")
                    print(f"   Output[{i}] stats: min={torch.min(out):.6f}, max={torch.max(out):.6f}, mean={torch.mean(out):.6f}")
                    print(f"   Output[{i}] NaN count: {torch.isnan(out).sum()}")
            else:
                print(f"   Output shape: {output.shape}")
                print(f"   Output stats: min={torch.min(output):.6f}, max={torch.max(output):.6f}, mean={torch.mean(output):.6f}")
                print(f"   NaN count: {torch.isnan(output).sum()}")
            
    except Exception as e:
        print(f"❌ Failed with pos_len=13: {e}")
    
    # Test with pos_len=19 (model's native size)
    print("\n2. Testing with pos_len=19:")
    try:
        model, swa_model, other_state_dict = load_model(
            model_path, use_swa=False, device="cpu", pos_len=19, verbose=False
        )
        print("✅ Model loaded successfully with pos_len=19")
        
        # Test forward pass with 13x13 input
        dummy_binary = torch.randn(1, 22, 13, 13)
        dummy_global = torch.randn(1, 19)
        
        with torch.no_grad():
            output = model(dummy_binary, dummy_global)
            print(f"✅ Forward pass successful, output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"   Output tuple length: {len(output)}")
                for i, out in enumerate(output):
                    print(f"   Output[{i}] shape: {out.shape}")
                    print(f"   Output[{i}] stats: min={torch.min(out):.6f}, max={torch.max(out):.6f}, mean={torch.mean(out):.6f}")
                    print(f"   Output[{i}] NaN count: {torch.isnan(out).sum()}")
            else:
                print(f"   Output shape: {output.shape}")
                print(f"   Output stats: min={torch.min(output):.6f}, max={torch.max(output):.6f}, mean={torch.mean(output):.6f}")
                print(f"   NaN count: {torch.isnan(output).sum()}")
            
    except Exception as e:
        print(f"❌ Failed with pos_len=19: {e}")

def test_extra_outputs():
    """Test ExtraOutputs API with different scenarios."""
    print("\n=== Testing ExtraOutputs API ===")
    
    model_path = "models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt"
    
    # Load model with pos_len=19
    model, swa_model, other_state_dict = load_model(
        model_path, use_swa=False, device="cpu", pos_len=19, verbose=False
    )
    model.eval()
    
    # Test with random data
    print("\n1. Testing with random data:")
    dummy_binary = torch.randn(1, 22, 13, 13)
    dummy_global = torch.randn(1, 19)
    
    extra = ExtraOutputs(requested=["rconv14.out"])
    extra.no_grad = True
    
    with torch.no_grad():
        _ = model(dummy_binary, dummy_global, extra_outputs=extra)
        
        if "rconv14.out" in extra.returned:
            act = extra.returned["rconv14.out"]
            print(f"✅ ExtraOutputs successful with random data")
            print(f"   Activation shape: {act.shape}")
            print(f"   Activation stats: min={torch.min(act):.6f}, max={torch.max(act):.6f}, mean={torch.mean(act):.6f}")
            print(f"   NaN count: {torch.isnan(act).sum()}")
        else:
            print("❌ ExtraOutputs failed - no activations returned")
    
    # Test with real NPZ data
    print("\n2. Testing with real NPZ data:")
    npz_file = "human_games_analysis/npz_files/2015-03-06T16:25:13.507Z_k5m7o9gtv63k.npz"
    
    if Path(npz_file).exists():
        data = np.load(npz_file)
        
        # Correctly unpack binary input
        binary_packed = data["binaryInputNCHWPacked"][:1]  # Shape: (1, 22, 22)
        binary_unpacked = np.unpackbits(binary_packed, axis=2)  # Shape: (1, 22, 176)
        
        # Reshape to (1, 22, 13, 13) - take only the first 169 bits (13*13)
        binary_reshaped = binary_unpacked[:, :, :169].reshape(1, 22, 13, 13)
        
        global_input = torch.from_numpy(data["globalInputNC"][:1]).float()
        
        print(f"   Binary packed shape: {binary_packed.shape}")
        print(f"   Binary unpacked shape: {binary_unpacked.shape}")
        print(f"   Binary reshaped shape: {binary_reshaped.shape}")
        print(f"   Global input shape: {global_input.shape}")
        
        binary_tensor = torch.from_numpy(binary_reshaped).float()
        
        extra = ExtraOutputs(requested=["rconv14.out"])
        extra.no_grad = True
        
        with torch.no_grad():
            _ = model(binary_tensor, global_input, extra_outputs=extra)
            
            if "rconv14.out" in extra.returned:
                act = extra.returned["rconv14.out"]
                print(f"✅ ExtraOutputs successful with real data")
                print(f"   Activation shape: {act.shape}")
                print(f"   Activation stats: min={torch.min(act):.6f}, max={torch.max(act):.6f}, mean={torch.mean(act):.6f}")
                print(f"   NaN count: {torch.isnan(act).sum()}")
            else:
                print("❌ ExtraOutputs failed - no activations returned")
    else:
        print(f"❌ NPZ file not found: {npz_file}")

def test_batch_processing():
    """Test batch processing to see if the issue is batch-specific."""
    print("\n=== Testing Batch Processing ===")
    
    model_path = "models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt"
    
    # Load model with pos_len=19
    model, swa_model, other_state_dict = load_model(
        model_path, use_swa=False, device="cpu", pos_len=19, verbose=False
    )
    model.eval()
    
    npz_file = "human_games_analysis/npz_files/2015-03-06T16:25:13.507Z_k5m7o9gtv63k.npz"
    
    if Path(npz_file).exists():
        data = np.load(npz_file)
        
        # Test first 5 positions individually
        for i in range(5):
            print(f"\nTesting position {i}:")
            
            # Correctly unpack and reshape binary input
            binary_packed = data["binaryInputNCHWPacked"][i:i+1]
            binary_unpacked = np.unpackbits(binary_packed, axis=2)
            binary_reshaped = binary_unpacked[:, :, :169].reshape(1, 22, 13, 13)
            
            binary_input = torch.from_numpy(binary_reshaped).float()
            global_input = torch.from_numpy(data["globalInputNC"][i:i+1]).float()
            
            print(f"   Binary input shape: {binary_input.shape}")
            print(f"   Global input shape: {global_input.shape}")
            
            extra = ExtraOutputs(requested=["rconv14.out"])
            extra.no_grad = True
            
            with torch.no_grad():
                _ = model(binary_input, global_input, extra_outputs=extra)
                
                if "rconv14.out" in extra.returned:
                    act = extra.returned["rconv14.out"]
                    print(f"   ✅ Activation shape: {act.shape}")
                    print(f"   Activation stats: min={torch.min(act):.6f}, max={torch.max(act):.6f}, mean={torch.mean(act):.6f}")
                    print(f"   NaN count: {torch.isnan(act).sum()}")
                else:
                    print("   ❌ No activations returned")

def main():
    test_model_loading()
    test_extra_outputs()
    test_batch_processing()

if __name__ == "__main__":
    main()