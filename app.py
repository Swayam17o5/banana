"""
FastAPI service for Banana Disease Detection
- Lazy-loads model (cloud safe)
- Accepts image at /predict
- Returns top predictions with confidence thresholding
"""

from fastapi import FastAPI, UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
import io
import inference

# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(title="Banana Disease Detector API")

# -------------------------------------------------
# CORS (allow mobile apps / Expo / React Native)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Lazy model loading (CRITICAL FOR CLOUD)
# -------------------------------------------------
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        print("Loading model for first request...")
        MODEL = inference.load_model()
    return MODEL

# -------------------------------------------------
# Config
# -------------------------------------------------
UNKNOWN_CONFIDENCE_THRESHOLD = float(
    os.environ.get("UNKNOWN_THRESHOLD", "0.40")
)

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "Banana Disease Detector API is running",
        "model_loaded": MODEL is not None,
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
    }

# -------------------------------------------------
# Optional UI
# -------------------------------------------------
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

@app.get("/ui", response_class=HTMLResponse)
def upload_ui():
    try:
        return FileResponse("static/index.html")
    except Exception:
        return """
        <!doctype html>
        <title>Banana Disease Detector</title>
        <p>UI not found.</p>
        <p>POST an image to <code>/predict</code></p>
        """

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = get_model()
        result = inference.predict_image(model, image)

        top1 = result["top1"]
        topK = result["topK"]

        # Apply confidence threshold
        if top1["probability"] < UNKNOWN_CONFIDENCE_THRESHOLD:
            top1 = {
                "label": "No disease found",
                "probability": top1["probability"],
            }
            topK = [top1]

        return {
            "top1": top1,
            "topK": topK,
            "threshold": UNKNOWN_CONFIDENCE_THRESHOLD,
        }

    except Exception as e:
        print("Inference error:", e)
        return {"error": "Prediction failed"}
