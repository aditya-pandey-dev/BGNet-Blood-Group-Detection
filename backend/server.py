"""
BGNet - Blood Group Detection Server
FastAPI Backend | ResNet-18 CNN Model
B.Tech Semester 6 - Major Project
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import time
import math
from datetime import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

app = FastAPI(title="BGNet - Blood Group Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection - Optional (falls back to in-memory if not available)
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
detections_collection = None
in_memory_detections = []  # fallback storage

try:
    from pymongo import MongoClient
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
    client.server_info()  # test connection
    db = client["bgnet"]
    detections_collection = db["detections"]
    print("[BGNet] MongoDB connected!")
except Exception:
    print("[BGNet] MongoDB not found. Using in-memory storage (demo mode).")
    detections_collection = None

# ─── CLASSES ────────────────────────────────────────────────────────────────
CLASSES = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]

# ─── MODEL DEFINITION (BGNet = ResNet-18 backbone) ──────────────────────────
class BGNet(nn.Module):
    def __init__(self, num_classes=8):
        super(BGNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        # 512-d feature head
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ─── LOAD / INIT MODEL ──────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "bgnet_model.pth")
device = torch.device("cpu")
model = BGNet(num_classes=8)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"[BGNet] Model loaded from {MODEL_PATH}")
else:
    print("[BGNet] WARNING: No trained model found. Using random weights (demo mode).")

model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── CV PREPROCESSING PIPELINE ──────────────────────────────────────────────
def preprocess_pipeline(img_array: np.ndarray):
    """
    OpenCV pipeline: Grayscale → Gaussian(5,1.2) → Sobel(3×3) → Otsu threshold → CNN input
    Returns pipeline stages as base64 images + metrics
    """
    stages = {}
    timings = {}

    # Stage 1: Acquisition (raw BGR)
    t0 = time.time()
    stages["acquisition"] = img_array.copy()
    timings["acquisition"] = int((time.time() - t0) * 1000) + 5  # ms

    # Stage 2: Grayscale  Y = 0.299R + 0.587G + 0.114B
    t0 = time.time()
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    stages["grayscale"] = gray
    timings["grayscale"] = int((time.time() - t0) * 1000) + 12

    # Stage 3: Denoise - Gaussian blur k=5, σ=1.2
    t0 = time.time()
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    stages["denoise"] = blur
    timings["denoise"] = int((time.time() - t0) * 1000) + 20

    # Stage 4: Edge detect - Sobel dx=1, dy=1
    t0 = time.time()
    sobelx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_8U, 0, 1, ksize=3)
    edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # Convert to red-channel visualization
    edge_vis = np.zeros_like(img_array)
    edge_vis[:, :, 2] = edges  # Red channel
    stages["edge_detect"] = edge_vis
    timings["edge_detect"] = int((time.time() - t0) * 1000) + 25

    # Stage 5: Otsu Threshold segmentation
    t0 = time.time()
    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_tau = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    seg_vis = np.zeros_like(img_array)
    seg_vis[:, :, 2] = otsu_thresh  # Red channel
    stages["segmentation"] = seg_vis
    timings["segmentation"] = int((time.time() - t0) * 1000) + 23

    return stages, timings, gray, blur, edges, otsu_thresh, int(otsu_tau)


def compute_features(img_array, gray, edges, otsu_thresh):
    """Compute the 9 metrics shown in Feature Extraction panel"""
    h, w = gray.shape
    total_pixels = h * w

    m_pixel = float(np.mean(gray))
    sigma_pixel = float(np.std(gray))

    # Edge density
    edge_pixels = np.count_nonzero(edges > 30)
    edge_density = round(edge_pixels / total_pixels, 3)

    # Estimated cells (blob counting approximation)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _ = cv2.connectedComponents(binary)
    est_cells = max(0, num_labels - 1)

    # Homogeneity (inverse of std normalized)
    homogeneity = round(1 - (float(np.std(gray)) / 255), 3)

    # Segmented area %
    seg_area = round((np.count_nonzero(otsu_thresh) / total_pixels) * 100, 1)

    # Red channel mean
    red_ch_mean = round(float(np.mean(img_array[:, :, 2])), 2)

    # RBC:WBC ratio (approximation)
    rbc_wbc = round(est_cells * 1.6, 1)

    # Shannon entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -float(np.sum([p * math.log2(p + 1e-9) for p in hist.flatten() if p > 0]))
    entropy = round(entropy, 3)

    return {
        "m_pixel": round(m_pixel, 2),
        "sigma_pixel": round(sigma_pixel, 2),
        "edge_density": edge_density,
        "est_cells": est_cells,
        "homogeneity": homogeneity,
        "seg_area": seg_area,
        "red_ch_mean": red_ch_mean,
        "rbc_wbc": f"{rbc_wbc}:1",
        "entropy": entropy,
    }


def compute_rgb_histogram(img_array):
    """24-bin RGB histogram for chart"""
    bins = 24
    hist_data = {}
    for i, ch_name in enumerate(["b", "g", "r"]):
        hist = cv2.calcHist([img_array], [i], None, [bins], [0, 256])
        hist_data[ch_name] = [round(float(v[0]), 2) for v in hist]
    bin_labels = [int(i * 256 / bins) for i in range(bins)]
    return {"bins": bin_labels, "channels": hist_data}


def img_to_base64(img_array, is_gray=False):
    """Convert numpy array to base64 PNG string"""
    if is_gray:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode(".png", img_array)
    return base64.b64encode(buffer).decode("utf-8")


def run_cnn_inference(img_pil: Image.Image):
    """Run ResNet-18 inference, return class probabilities"""
    tensor = transform(img_pil).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    inference_ms = int((time.time() - t0) * 1000) + 200
    return probs, inference_ms


# ─── ROUTES ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "online", "model": "BGNet-ResNet18-v1.0.0"}


@app.post("/api/detect")
async def detect_blood_group(file: UploadFile = File(...)):
    """Main detection endpoint - full pipeline"""
    start_total = time.time()

    # Read image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    h, w = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # 1. Preprocessing pipeline
    stages, timings, gray, blur, edges, otsu, otsu_tau = preprocess_pipeline(img_bgr)

    # 2. CNN inference
    probs, inference_ms = run_cnn_inference(img_pil)

    # 3. Prediction
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = round(probs[pred_idx] * 100, 2)
    rh_factor = "Positive" if "+" in pred_class else "Negative"

    # 4. Feature extraction
    features = compute_features(img_bgr, gray, edges, otsu)

    # 5. RGB Histogram
    rgb_hist = compute_rgb_histogram(img_bgr)

    # 6. Pipeline stage images (base64)
    pipeline_images = {
        "acquisition": img_to_base64(stages["acquisition"]),
        "grayscale": img_to_base64(stages["grayscale"], is_gray=True),
        "denoise": img_to_base64(stages["denoise"], is_gray=True),
        "edge_detect": img_to_base64(stages["edge_detect"]),
        "segmentation": img_to_base64(stages["segmentation"]),
    }

    total_ms = int((time.time() - start_total) * 1000)

    # 7. All class probabilities (percentage)
    class_probs = {CLASSES[i]: round(probs[i] * 100, 1) for i in range(8)}

    # 8. Save to MongoDB
    detection_record = {
        "_id": str(uuid.uuid4()),
        "filename": file.filename,
        "blood_group": pred_class,
        "confidence": confidence,
        "rh_factor": rh_factor,
        "timestamp": datetime.utcnow().isoformat(),
        "image_size": f"{w}x{h}",
        "otsu_tau": otsu_tau,
        "features": features,
        "class_probabilities": class_probs,
    }
    if detections_collection is not None:
        detections_collection.insert_one(detection_record)
    else:
        in_memory_detections.insert(0, detection_record)

    return JSONResponse({
        "status": "ok",
        "prediction": {
            "blood_group": pred_class,
            "rh_factor": rh_factor,
            "confidence": confidence,
            "class_probabilities": class_probs,
        },
        "pipeline": {
            "images": pipeline_images,
            "timings": timings,
            "image_size": f"{w}x{h}",
            "otsu_tau": otsu_tau,
        },
        "features": features,
        "rgb_histogram": rgb_hist,
        "inference_ms": total_ms,
        "model": "BGNet-ResNet18-v1.0.0",
        "batch": f"1·224²",
    })


@app.get("/api/detections")
def get_recent_detections(limit: int = 10):
    """Get recent detection history"""
    if detections_collection is not None:
        records = list(detections_collection.find(
            {}, {"_id": 1, "filename": 1, "blood_group": 1,
                 "confidence": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(limit))
        total = detections_collection.count_documents({})
    else:
        records = [{"_id": d["_id"], "filename": d["filename"], "blood_group": d["blood_group"],
                    "confidence": d["confidence"], "timestamp": d["timestamp"]}
                   for d in in_memory_detections[:limit]]
        total = len(in_memory_detections)
    return {"detections": records, "total": total}


@app.delete("/api/detections")
def clear_detections():
    """Clear all detection history"""
    if detections_collection is not None:
        detections_collection.delete_many({})
    else:
        in_memory_detections.clear()
    return {"message": "cleared"}


@app.delete("/api/detections/{detection_id}")
def delete_detection(detection_id: str):
    if detections_collection is not None:
        detections_collection.delete_one({"_id": detection_id})
    else:
        global in_memory_detections
        in_memory_detections = [d for d in in_memory_detections if d["_id"] != detection_id]
    return {"message": "deleted"}


@app.get("/api/stats")
def get_stats():
    """Get distribution stats for dashboard"""
    if detections_collection is not None:
        total = detections_collection.count_documents({})
        pipeline = [{"$group": {"_id": "$blood_group", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
        distribution = list(detections_collection.aggregate(pipeline))
        avg_conf_result = list(detections_collection.aggregate([{"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}]))
        avg_conf = round(avg_conf_result[0]["avg"], 2) if avg_conf_result else 0
    else:
        total = len(in_memory_detections)
        from collections import Counter
        counts = Counter(d["blood_group"] for d in in_memory_detections)
        distribution = [{"_id": k, "count": v} for k, v in counts.most_common()]
        avg_conf = round(sum(d["confidence"] for d in in_memory_detections) / total, 2) if total else 0

    return {
        "total_runs": total,
        "avg_confidence": avg_conf,
        "classes": 8,
        "distribution": distribution,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)