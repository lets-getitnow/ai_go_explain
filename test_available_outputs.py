#!/usr/bin/env python3
"""Test script to see what outputs are available from KataGo model."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add KataGo python path
sys.path.insert(0, str(Path(__file__).parent / "KataGo" / "python"))

from katago.train.model_pytorch import ExtraOutputs
from katago.train.load_model import load_model

def test_available_outputs():
    """Test what outputs are available from the KataGo model."""
    
    # Load the model
    model_path = Path("models/kata1-b28c512nbt-s9584861952-d4960414494/model.ckpt")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    # Load model using KataGo's infrastructure
    try:
        # Temporarily monkey-patch torch.load for PyTorch 2.6 compatibility
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        
        try:
            model, swa_model, other_state_dict = load_model(
                checkpoint_file=str(model_path),
                use_swa=False,
                device="cpu",
                pos_len=19,
                verbose=False
            )
        finally:
            torch.load = original_torch_load
        
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create dummy input in the correct format
    batch_size = 1
    board_size = 19
    num_channels = 22  # Standard KataGo input channels
    
    # Create dummy binary input in 4D format (N, C, H, W)
    binary_input = np.random.rand(batch_size, num_channels, board_size, board_size).astype(np.float32)
    global_input = np.random.randn(batch_size, 19).astype(np.float32)
    
    # Convert to tensors
    binary_tensor = torch.from_numpy(binary_input)
    global_tensor = torch.from_numpy(global_input)
    
    # Try different output names
    output_names = [
        'rconv14.out',  # The actual layer being used
        'trunkfinal',   # Final trunk output
        'policy_output',
        'value_output', 
        'policy_head_output',
        'value_head_output',
        'trunk_output',
        'trunk_block_9_output',
        'rconv14_output',
        'policy_head',
        'value_head'
    ]
    
    print("Testing available outputs...")
    
    for output_name in output_names:
        try:
            extra = ExtraOutputs(requested=[output_name])
            extra.no_grad = True
            
            # Run forward pass
            with torch.no_grad():
                _ = model(binary_tensor, global_tensor, extra_outputs=extra)
            
            if output_name in extra.returned:
                print(f"✅ {output_name}: {extra.returned[output_name].shape}")
            else:
                print(f"❌ {output_name}: Not available")
                
        except Exception as e:
            print(f"❌ {output_name}: Error - {e}")
    
    # Look for policy and value outputs in the available list
    print("\nSearching for policy and value outputs in available layers...")
    available_outputs = [
        'rconv1.normactconvp.out', 'rconv1.blockstack.0.normactconv1.out', 'rconv1.blockstack.0.normactconv2.out', 'rconv1.blockstack.0.out', 'rconv1.blockstack.1.normactconv1.out', 'rconv1.blockstack.1.normactconv2.out', 'rconv1.blockstack.1.out', 'rconv1.normactconvq.out', 'rconv1.out', 'rconv2.normactconvp.out', 'rconv2.blockstack.0.normactconv1.out', 'rconv2.blockstack.0.normactconv2.out', 'rconv2.blockstack.0.out', 'rconv2.blockstack.1.normactconv1.out', 'rconv2.blockstack.1.normactconv2.out', 'rconv2.blockstack.1.out', 'rconv2.normactconvq.out', 'rconv2.out', 'rconv3.normactconvp.out', 'rconv3.blockstack.0.normactconv1.out', 'rconv3.blockstack.0.normactconv2.out', 'rconv3.blockstack.0.out', 'rconv3.blockstack.1.normactconv1.out', 'rconv3.blockstack.1.normactconv2.out', 'rconv3.blockstack.1.out', 'rconv3.normactconvq.out', 'rconv3.out', 'rconv4.normactconvp.out', 'rconv4.blockstack.0.normactconv1.out', 'rconv4.blockstack.0.normactconv2.out', 'rconv4.blockstack.0.out', 'rconv4.blockstack.1.normactconv1.out', 'rconv4.blockstack.1.normactconv2.out', 'rconv4.blockstack.1.out', 'rconv4.normactconvq.out', 'rconv4.out', 'rconv5.normactconvp.out', 'rconv5.blockstack.0.normactconv1.out', 'rconv5.blockstack.0.normactconv2.out', 'rconv5.blockstack.0.out', 'rconv5.blockstack.1.normactconv1.out', 'rconv5.blockstack.1.normactconv2.out', 'rconv5.blockstack.1.out', 'rconv5.normactconvq.out', 'rconv5.out', 'rconv6.normactconvp.out', 'rconv6.blockstack.0.normactconv1.out', 'rconv6.blockstack.0.normactconv2.out', 'rconv6.blockstack.0.out', 'rconv6.blockstack.1.normactconv1.out', 'rconv6.blockstack.1.normactconv2.out', 'rconv6.blockstack.1.out', 'rconv6.normactconvq.out', 'rconv6.out', 'rconv7.normactconvp.out', 'rconv7.blockstack.0.normactconv1.out', 'rconv7.blockstack.0.normactconv2.out', 'rconv7.blockstack.0.out', 'rconv7.blockstack.1.normactconv1.out', 'rconv7.blockstack.1.normactconv2.out', 'rconv7.blockstack.1.out', 'rconv7.normactconvq.out', 'rconv7.out', 'rconv8.normactconvp.out', 'rconv8.blockstack.0.normactconv1.out', 'rconv8.blockstack.0.normactconv2.out', 'rconv8.blockstack.0.out', 'rconv8.blockstack.1.normactconv1.out', 'rconv8.blockstack.1.normactconv2.out', 'rconv8.blockstack.1.out', 'rconv8.normactconvq.out', 'rconv8.out', 'rconv9.normactconvp.out', 'rconv9.blockstack.0.normactconv1.out', 'rconv9.blockstack.0.normactconv2.out', 'rconv9.blockstack.0.out', 'rconv9.blockstack.1.normactconv1.out', 'rconv9.blockstack.1.normactconv2.out', 'rconv9.blockstack.1.out', 'rconv9.normactconvq.out', 'rconv9.out', 'rconv10.normactconvp.out', 'rconv10.blockstack.0.normactconv1.out', 'rconv10.blockstack.0.normactconv2.out', 'rconv10.blockstack.0.out', 'rconv10.blockstack.1.normactconv1.out', 'rconv10.blockstack.1.normactconv2.out', 'rconv10.blockstack.1.out', 'rconv10.normactconvq.out', 'rconv10.out', 'rconv11.normactconvp.out', 'rconv11.blockstack.0.normactconv1.out', 'rconv11.blockstack.0.normactconv2.out', 'rconv11.blockstack.0.out', 'rconv11.blockstack.1.normactconv1.out', 'rconv11.blockstack.1.normactconv2.out', 'rconv11.blockstack.1.out', 'rconv11.normactconvq.out', 'rconv11.out', 'rconv12.normactconvp.out', 'rconv12.blockstack.0.normactconv1.out', 'rconv12.blockstack.0.normactconv2.out', 'rconv12.blockstack.0.out', 'rconv12.blockstack.1.normactconv1.out', 'rconv12.blockstack.1.normactconv2.out', 'rconv12.blockstack.1.out', 'rconv12.normactconvq.out', 'rconv12.out', 'rconv13.normactconvp.out', 'rconv13.blockstack.0.normactconv1.out', 'rconv13.blockstack.0.normactconv2.out', 'rconv13.blockstack.0.out', 'rconv13.blockstack.1.normactconv1.out', 'rconv13.blockstack.1.normactconv2.out', 'rconv13.blockstack.1.out', 'rconv13.normactconvq.out', 'rconv13.out', 'rconv14.normactconvp.out', 'rconv14.blockstack.0.normactconv1.out', 'rconv14.blockstack.0.normactconv2.out', 'rconv14.blockstack.0.out', 'rconv14.blockstack.1.normactconv1.out', 'rconv14.blockstack.1.normactconv2.out', 'rconv14.blockstack.1.out', 'rconv14.normactconvq.out', 'rconv14.out', 'rconv15.normactconvp.out', 'rconv15.blockstack.0.normactconv1.out', 'rconv15.blockstack.0.normactconv2.out', 'rconv15.blockstack.0.out', 'rconv15.blockstack.1.normactconv1.out', 'rconv15.blockstack.1.normactconv2.out', 'rconv15.blockstack.1.out', 'rconv15.normactconvq.out', 'rconv15.out', 'rconv16.normactconvp.out', 'rconv16.blockstack.0.normactconv1.out', 'rconv16.blockstack.0.normactconv2.out', 'rconv16.blockstack.0.out', 'rconv16.blockstack.1.normactconv1.out', 'rconv16.blockstack.1.normactconv2.out', 'rconv16.blockstack.1.out', 'rconv16.normactconvq.out', 'rconv16.out', 'rconv17.normactconvp.out', 'rconv17.blockstack.0.normactconv1.out', 'rconv17.blockstack.0.normactconv2.out', 'rconv17.blockstack.0.out', 'rconv17.blockstack.1.normactconv1.out', 'rconv17.blockstack.1.normactconv2.out', 'rconv17.blockstack.1.out', 'rconv17.normactconvq.out', 'rconv17.out', 'rconv18.normactconvp.out', 'rconv18.blockstack.0.normactconv1.out', 'rconv18.blockstack.0.normactconv2.out', 'rconv18.blockstack.0.out', 'rconv18.blockstack.1.normactconv1.out', 'rconv18.blockstack.1.normactconv2.out', 'rconv18.blockstack.1.out', 'rconv18.normactconvq.out', 'rconv18.out', 'rconv19.normactconvp.out', 'rconv19.blockstack.0.normactconv1.out', 'rconv19.blockstack.0.normactconv2.out', 'rconv19.blockstack.0.out', 'rconv19.blockstack.1.normactconv1.out', 'rconv19.blockstack.1.normactconv2.out', 'rconv19.blockstack.1.out', 'rconv19.normactconvq.out', 'rconv19.out', 'rconv20.normactconvp.out', 'rconv20.blockstack.0.normactconv1.out', 'rconv20.blockstack.0.normactconv2.out', 'rconv20.blockstack.0.out', 'rconv20.blockstack.1.normactconv1.out', 'rconv20.blockstack.1.normactconv2.out', 'rconv20.blockstack.1.out', 'rconv20.normactconvq.out', 'rconv20.out', 'rconv21.normactconvp.out', 'rconv21.blockstack.0.normactconv1.out', 'rconv21.blockstack.0.normactconv2.out', 'rconv21.blockstack.0.out', 'rconv21.blockstack.1.normactconv1.out', 'rconv21.blockstack.1.normactconv2.out', 'rconv21.blockstack.1.out', 'rconv21.normactconvq.out', 'rconv21.out', 'rconv22.normactconvp.out', 'rconv22.blockstack.0.normactconv1.out', 'rconv22.blockstack.0.normactconv2.out', 'rconv22.blockstack.0.out', 'rconv22.blockstack.1.normactconv1.out', 'rconv22.blockstack.1.normactconv2.out', 'rconv22.blockstack.1.out', 'rconv22.normactconvq.out', 'rconv22.out', 'rconv23.normactconvp.out', 'rconv23.blockstack.0.normactconv1.out', 'rconv23.blockstack.0.normactconv2.out', 'rconv23.blockstack.0.out', 'rconv23.blockstack.1.normactconv1.out', 'rconv23.blockstack.1.normactconv2.out', 'rconv23.blockstack.1.out', 'rconv23.normactconvq.out', 'rconv23.out', 'rconv24.normactconvp.out', 'rconv24.blockstack.0.normactconv1.out', 'rconv24.blockstack.0.normactconv2.out', 'rconv24.blockstack.0.out', 'rconv24.blockstack.1.normactconv1.out', 'rconv24.blockstack.1.normactconv2.out', 'rconv24.blockstack.1.out', 'rconv24.normactconvq.out', 'rconv24.out', 'rconv25.normactconvp.out', 'rconv25.blockstack.0.normactconv1.out', 'rconv25.blockstack.0.normactconv2.out', 'rconv25.blockstack.0.out', 'rconv25.blockstack.1.normactconv1.out', 'rconv25.blockstack.1.normactconv2.out', 'rconv25.blockstack.1.out', 'rconv25.normactconvq.out', 'rconv25.out', 'rconv26.normactconvp.out', 'rconv26.blockstack.0.normactconv1.out', 'rconv26.blockstack.0.normactconv2.out', 'rconv26.blockstack.0.out', 'rconv26.blockstack.1.normactconv1.out', 'rconv26.blockstack.1.normactconv2.out', 'rconv26.blockstack.1.out', 'rconv26.normactconvq.out', 'rconv26.out', 'rconv27.normactconvp.out', 'rconv27.blockstack.0.normactconv1.out', 'rconv27.blockstack.0.normactconv2.out', 'rconv27.blockstack.0.out', 'rconv27.blockstack.1.normactconv1.out', 'rconv27.blockstack.1.normactconv2.out', 'rconv27.blockstack.1.out', 'rconv27.normactconvq.out', 'rconv27.out', 'rconv28.normactconvp.out', 'rconv28.blockstack.0.normactconv1.out', 'rconv28.blockstack.0.normactconv2.out', 'rconv28.blockstack.0.out', 'rconv28.blockstack.1.normactconv1.out', 'rconv28.blockstack.1.normactconv2.out', 'rconv28.blockstack.1.out', 'rconv28.normactconvq.out', 'rconv28.out', 'trunkfinal']
    
    policy_layers = [layer for layer in available_outputs if 'policy' in layer.lower()]
    value_layers = [layer for layer in available_outputs if 'value' in layer.lower()]
    
    print(f"Policy-related layers: {policy_layers}")
    print(f"Value-related layers: {value_layers}")
    
    # Try with no specific requested outputs to see what's available
    try:
        extra = ExtraOutputs(requested=[])
        extra.no_grad = True
        
        with torch.no_grad():
            _ = model(binary_tensor, global_tensor, extra_outputs=extra)
        
        print(f"\nAll available outputs: {extra.available}")
        
    except Exception as e:
        print(f"Error getting all available outputs: {e}")
    
    # Try to get policy and value outputs from model forward pass
    print("\nTesting model forward pass return values...")
    try:
        with torch.no_grad():
            outputs = model(binary_tensor, global_tensor)
        
        print(f"Model forward pass returned: {type(outputs)}")
        if isinstance(outputs, (list, tuple)):
            print(f"Number of outputs: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"Output {i}: {type(output)}")
                if isinstance(output, (list, tuple)):
                    print(f"  Number of sub-outputs: {len(output)}")
                    for j, sub_output in enumerate(output):
                        print(f"    Sub-output {j}: {type(sub_output)}, shape: {sub_output.shape if hasattr(sub_output, 'shape') else 'N/A'}")
                elif hasattr(output, 'shape'):
                    print(f"  Shape: {output.shape}")
                else:
                    print(f"  Value: {output}")
        elif hasattr(outputs, 'shape'):
            print(f"Single output shape: {outputs.shape}")
        else:
            print(f"Output: {outputs}")
            
    except Exception as e:
        print(f"Error in model forward pass: {e}")

if __name__ == "__main__":
    test_available_outputs() 