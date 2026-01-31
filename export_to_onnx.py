"""
Export HybridCNNViT PyTorch model to ONNX for mobile inference.

Steps:
1. Load model weights from best_model.pth
2. Create dummy input (1,3,224,224)
3. torch.onnx.export with opset 17
4. Validate output parity (max absolute diff) vs TorchScript Lite if available

Produces: banana_model.onnx in assets/models (to be copied manually or scripted)
"""
import os
import torch
from model import HybridCNNViT

NUM_CLASSES = 9
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
OUTPUT_ONNX = os.path.join(os.path.dirname(__file__), 'banana_model.onnx')
ASSETS_DEST = os.path.join(os.path.dirname(__file__), '..', 'assets', 'models', 'banana_model.onnx')

def load_model():
    model = HybridCNNViT(num_classes=NUM_CLASSES)
    state = torch.load(WEIGHTS_PATH, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected}")
    model.eval()
    return model

def export():
    model = load_model()
    dummy = torch.randn(1,3,224,224)
    print('[info] Exporting to ONNX...')
    torch.onnx.export(
        model,
        dummy,
        OUTPUT_ONNX,
        input_names=['input'],
        output_names=['logits'],
        opset_version=17,
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(OUTPUT_ONNX)/1e6
    print(f'[done] ONNX model written to {OUTPUT_ONNX} size={size_mb:.2f}MB')
    try:
        os.makedirs(os.path.dirname(ASSETS_DEST), exist_ok=True)
        import shutil
        shutil.copy2(OUTPUT_ONNX, ASSETS_DEST)
        print(f'[copy] Copied ONNX to assets: {ASSETS_DEST}')
    except Exception as e:
        print(f'[copy] Failed to copy ONNX to assets: {e}')

    try:
        # Optional TorchScript Lite parity check if file present
        lite_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'models', 'banana_model_mobile.ptl')
        if os.path.isfile(lite_path):
            ts = torch.jit.load(lite_path)
            with torch.no_grad():
                torch_out = ts(dummy).detach()
                import onnxruntime as ort
                ort_sess = ort.InferenceSession(OUTPUT_ONNX, providers=['CPUExecutionProvider'])
                ort_out = ort_sess.run(['logits'], {'input': dummy.numpy()})[0]
                max_diff = (torch_out.numpy() - ort_out).abs().max()
                print(f'[parity] Max abs diff TorchScript Lite vs ONNX: {max_diff:.6f}')
        else:
            print('[parity] TorchScript Lite file not found; skipped diff check.')
    except Exception as e:
        print(f'[parity] Skipped parity verification due to error: {e}')

if __name__ == '__main__':
    export()
