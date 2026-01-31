"""
Model loading and inference helpers

- Downloads the model from cloud storage if missing
- Loads HybridCNNViT weights (supports FP16 checkpoints)
- Converts FP16 → FP32 for stable CPU inference
- Performs image inference and returns top predictions
"""

import os
import urllib.request
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import HybridCNNViT

# ---------------- Configuration ----------------
DEVICE = torch.device("cpu")
NUM_CLASSES = 9

# These must be provided via environment variables in cloud deployment
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "/tmp/best_model_fp16.pth")
CHECKPOINT_URL = os.environ.get("MODEL_URL")

CLASS_NAMES = [
    "Anthracnose",
    "Banana Fruit-Scarring Beetle",
    "Banana Skipper Damage",
    "Banana Split Peel",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana",
    "Healthy Banana leaf",
    "Panama Wilt Disease",
]

# ---------------- Transforms ----------------
base_inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ---------------- Utilities ----------------
def _ensure_checkpoint_available():
    """
    Ensure the model checkpoint exists locally.
    If not, download it from MODEL_URL.
    """
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH

    if not CHECKPOINT_URL:
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH} and MODEL_URL is not set."
        )

    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or ".", exist_ok=True)
    print(f"Downloading model from: {CHECKPOINT_URL}")

    try:
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=300) as resp, open(
            CHECKPOINT_PATH, "wb"
        ) as out:
            chunk_size = 1024 * 1024  # 1 MB
            downloaded = 0
            total = int(resp.headers.get("Content-Length", 0)) or None

            while True:
                data = resp.read(chunk_size)
                if not data:
                    break
                out.write(data)
                downloaded += len(data)

                if total:
                    pct = (downloaded / total) * 100
                    print(
                        f"Downloaded {downloaded / 1_048_576:.1f} MB "
                        f"({pct:.1f}%)",
                        end="\r",
                    )

        print("\nModel download complete.")
        size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
        print(f"Model size on disk: {size_mb:.1f} MB")

    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

    return CHECKPOINT_PATH

# ---------------- Model Loading ----------------
def load_model():
    """
    Instantiate the model and load weights.

    Supports:
    - FP16 checkpoints (compressed)
    - FP32 checkpoints

    FP16 weights are converted to FP32 for safe CPU inference.
    """
    ckpt_path = _ensure_checkpoint_available()
    print(f"Loading model from {ckpt_path} on {DEVICE}...")

    model = HybridCNNViT(NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # Support both checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Detect FP16 checkpoint
    first_param = next(iter(state_dict.values()))
    is_fp16 = first_param.dtype == torch.float16

    if is_fp16:
        print("Detected FP16 checkpoint. Converting to FP32 for CPU inference...")
        model = model.half()
        model.load_state_dict(state_dict)
        model = model.float()
    else:
        model.load_state_dict(state_dict)

    model.eval()
    print("Model loaded successfully.")
    return model

# ---------------- Inference ----------------
def predict_image(model, pil_image: Image.Image, top_k: int = 3):
    """
    Perform inference on a PIL image.

    Returns:
    {
        "top1": {"label": str, "probability": float},
        "topK": [{"label": str, "probability": float}, ...]
    }
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    input_tensor = (
        base_inference_transform(pil_image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits[0], dim=0)

    sorted_indices = torch.argsort(probs, descending=True)

    top1 = {
        "label": CLASS_NAMES[sorted_indices[0]],
        "probability": probs[sorted_indices[0]].item(),
    }

    topk = [
        {
            "label": CLASS_NAMES[i],
            "probability": probs[i].item(),
        }
        for i in sorted_indices[:top_k]
    ]

    return {
        "top1": top1,
        "topK": topk,
    }
