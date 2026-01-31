"""
FP16 Model Compression Script

Converts PyTorch model weights to half precision (float16) for 50% size reduction.
FP16 weights can still be loaded and used on CPU by converting back to float32 at runtime.

Usage:
    python compress_fp16.py SRC=best_model.pth DST=best_model_fp16.pth
    
Expected:
    - Input: 1.13 GB (float32 weights)
    - Output: ~565 MB (float16 weights)
    - Accuracy: No loss (inference still uses float32 computation)
"""

import torch
import sys
from pathlib import Path


def compress_to_fp16(src_path: str, dst_path: str):
    """
    Load checkpoint, convert all float32 tensors to float16, and save.
    
    Args:
        src_path: Path to original checkpoint (.pth file)
        dst_path: Path to save FP16 compressed checkpoint
    """
    print(f"Loading checkpoint from {src_path}...")
    checkpoint = torch.load(src_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Detected wrapped format (model_state_dict key)")
    else:
        state_dict = checkpoint
        print("Detected direct state_dict format")
    
    # Convert all float tensors to half precision
    compressed_state_dict = {}
    fp32_count = 0
    fp16_count = 0
    
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            compressed_state_dict[key] = tensor.half()
            fp32_count += 1
        else:
            # Keep non-float32 tensors as-is (e.g., int64 for batch norm stats)
            compressed_state_dict[key] = tensor
            fp16_count += 1
    
    print(f"Converted {fp32_count} float32 tensors to float16")
    print(f"Kept {fp16_count} non-float32 tensors unchanged")
    
    # Save in the same format as input
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        output_checkpoint = checkpoint.copy()
        output_checkpoint["model_state_dict"] = compressed_state_dict
        torch.save(output_checkpoint, dst_path)
    else:
        torch.save(compressed_state_dict, dst_path)
    
    # Report file sizes
    src_size = Path(src_path).stat().st_size / (1024**3)  # GB
    dst_size = Path(dst_path).stat().st_size / (1024**3)  # GB
    reduction = (1 - dst_size / src_size) * 100
    
    print(f"\n✅ Compression complete!")
    print(f"   Original: {src_size:.2f} GB")
    print(f"   Compressed: {dst_size:.2f} GB")
    print(f"   Reduction: {reduction:.1f}%")
    print(f"\nSaved to: {dst_path}")


if __name__ == "__main__":
    # Parse command-line arguments: SRC=... DST=...
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            args[key.upper()] = value
    
    src = args.get("SRC", "best_model.pth")
    dst = args.get("DST", "best_model_fp16.pth")
    
    if not Path(src).exists():
        print(f"❌ Error: Source file '{src}' not found")
        print(f"Usage: python compress_fp16.py SRC=<input.pth> DST=<output.pth>")
        sys.exit(1)
    
    compress_to_fp16(src, dst)
