"""
INT8 Dynamic Quantization Script

Applies PyTorch dynamic quantization to reduce model size by 30-60% with minimal accuracy loss.
Targets nn.Linear layers (ViT attention, MLP) with INT8 weights. CNN Conv2d layers remain FP32.

Usage:
    python compress_int8_dynamic.py SRC=best_model.pth DST=best_model_int8.pth
    
Expected:
    - Reduction: 30-60% (depends on Linear vs Conv2d ratio)
    - Accuracy: 0.5-3% drop (typically <2% for vision models)
    - Speed: 1.2-2.0× faster on CPU
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Import model architecture
from model import HybridCNNViT


def load_model_from_checkpoint(checkpoint_path: str, num_classes: int = 9):
    """Load model and weights from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model architecture
    model = HybridCNNViT(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Detected wrapped format (model_state_dict key)")
    else:
        state_dict = checkpoint
        print("Detected direct state_dict format")
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model


def compress_to_int8(src_path: str, dst_path: str, num_classes: int = 9):
    """
    Apply dynamic INT8 quantization to model and save.
    
    Dynamic quantization:
    - Converts nn.Linear layer weights to INT8
    - Activations remain FP32, converted to INT8 at runtime
    - Best for CPU inference (no GPU support)
    
    Args:
        src_path: Path to original checkpoint
        dst_path: Path to save quantized model
        num_classes: Number of output classes (default: 9 for banana diseases)
    """
    # Load original model
    model = load_model_from_checkpoint(src_path, num_classes)
    
    # Apply dynamic quantization to Linear layers
    print("\nApplying dynamic INT8 quantization...")
    print("Targeting: nn.Linear layers (ViT QKV, MLP, classifier)")
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8
    )
    
    print("✅ Quantization complete")
    
    # Save quantized model (must save entire module, not just state_dict)
    print(f"\nSaving quantized model to {dst_path}...")
    torch.save(quantized_model, dst_path)
    
    # Report file sizes
    src_size = Path(src_path).stat().st_size / (1024**3)  # GB
    dst_size = Path(dst_path).stat().st_size / (1024**3)  # GB
    reduction = (1 - dst_size / src_size) * 100
    
    print(f"\n✅ Compression complete!")
    print(f"   Original: {src_size:.2f} GB")
    print(f"   Quantized: {dst_size:.2f} GB")
    print(f"   Reduction: {reduction:.1f}%")
    print(f"\nSaved to: {dst_path}")
    
    # Note about loading
    print("\n⚠️  Loading quantized models:")
    print("   Use: model = torch.load('best_model_int8.pth')")
    print("   NOT: model.load_state_dict(...)")


if __name__ == "__main__":
    # Parse command-line arguments
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            args[key.upper()] = value
    
    src = args.get("SRC", "best_model.pth")
    dst = args.get("DST", "best_model_int8.pth")
    num_classes = int(args.get("CLASSES", "9"))
    
    if not Path(src).exists():
        print(f"❌ Error: Source file '{src}' not found")
        print(f"Usage: python compress_int8_dynamic.py SRC=<input.pth> DST=<output.pth> [CLASSES=9]")
        sys.exit(1)
    
    compress_to_int8(src, dst, num_classes)
