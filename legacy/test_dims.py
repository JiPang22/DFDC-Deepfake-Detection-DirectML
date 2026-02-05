
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_model import RPPGDeepfakeModel

def test():
    device = torch.device('cpu')
    model = RPPGDeepfakeModel().to(device)
    
    # Input: (B, C, T, H, W)
    B, C, T, H, W = 32, 1, 30, 128, 128
    inputs = torch.randn(B, C, T, H, W).to(device)
    
    print(f"Testing with input shape: {inputs.shape}")
    try:
        output = model(inputs)
        print(f"Output shape: {output.shape}")
        if output.shape == (B, 1):
            print("SUCCESS: Dimensions match.")
        else:
            print(f"FAILURE: Output shape mismatch. Expected {(B, 1)}, got {output.shape}")
    except Exception as e:
        print(f"FAILURE: Exception during forward pass: {e}")

if __name__ == "__main__":
    test()
